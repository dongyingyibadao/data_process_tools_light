from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lightworkbench import lerobot_converter as converter


def pose(x: float = 0.0, y: float = 0.0) -> dict:
    return {
        "left_eef_pose": {"position": [x, y, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
        "right_eef_pose": {"position": [2 * x, y, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
        "head_pose": {"position": [0.0, 0.0, 1.5], "rotation": [0.0, 0.0, 0.0, 1.0]},
    }


def record(index: int, streams: dict[str, str], control=None) -> dict:
    return {
        "frame_idx": index,
        "t_ns": 1_000_000_000 + index,
        "joints": {
            "position": [float(index)] * len(converter.JOINT_NAMES),
            "velocity": [0.0] * len(converter.JOINT_NAMES),
            "torque": [0.0] * len(converter.JOINT_NAMES),
        },
        "robot_state": {"state": 2},
        "current_eef_pose": pose(float(index), float(index) / 2),
        "target_eef_pose": pose(float(index) + 0.1),
        "current_height_z": {"height_z": 0.0},
        "target_height_z": {"height_z": 0.1},
        "control": control,
        "videos": {
            name: {"path": path, "frame_id": index, "is_repeat": False}
            for name, path in streams.items()
        },
    }


def make_source(tmp_path: Path, number: int = 1, *, width: int = 48,
                extra_streams: tuple[str, ...] = ("rgbd_head_depth", "head_right")) -> converter.SourceEpisode:
    episode_path = tmp_path / f"episode_{number:06d}"
    episode_path.mkdir(parents=True)
    names = (*converter.REQUIRED_VIDEO_STREAMS, *extra_streams)
    videos = {}
    stream_paths = {}
    for name in names:
        suffix = ".mkv" if converter.is_depth_stream(name) else ".mp4"
        path = episode_path / "videos" / name / f"episode_{number:06d}{suffix}"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"{name}-{number}".encode())
        stream_paths[name] = path.relative_to(episode_path).as_posix()
        videos[name] = converter.VideoSource(
            name, path, width, 32, 30.0, 3, converter.is_depth_stream(name)
        )
    records = [record(index, stream_paths) for index in range(3)]
    (episode_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [
            {"_type": "session_header", "fps_target": 30, "episode_id": number},
            *records,
            {"_type": "session_footer", "aborted": False},
        ]) + "\n",
        encoding="utf-8",
    )
    (episode_path / "task_meta.json").write_text('{"description":"source task"}\n', encoding="utf-8")
    return converter.SourceEpisode(
        episode_path, number, "stored task", 30,
        {"_type": "session_header", "fps_target": 30, "episode_id": number},
        records, videos,
    )


def test_import_does_not_load_optional_heavy_dependencies() -> None:
    code = """
import sys
import lightworkbench.lerobot_converter
for name in ('numpy', 'av', 'cv2', 'torch', 'lerobot'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_action_is_fixed_20d_and_uses_frame_t_flat_commands() -> None:
    current = record(0, {}, {
        "commands.SET_LEFT_GRIPPER_SPEED": 0.0,
        "commands.SET_LEFT_FORCE": 2.5,
        "commands.SET_RIGHT_GRIPPER_SPEED": -3.0,
    })
    following = record(1, {})
    action = converter.build_action(current, following)
    assert converter.ACTION_NAMES[6:10] == [
        "left.gripper_speed", "left.gripper_force",
        "left.gripper_speed_valid", "left.gripper_force_valid",
    ]
    assert len(action) == 20
    assert action[:6] == pytest.approx([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert action[6:10] == [0.0, 2.5, 1.0, 1.0]
    assert action[10:16] == pytest.approx([2.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert action[16:20] == [-3.0, 0.0, 1.0, 0.0]


def test_action_distinguishes_missing_from_real_zero_and_accepts_nested_commands() -> None:
    missing = converter.build_action(record(0, {}, None), record(1, {}))
    assert missing[6:10] == [0.0, 0.0, 0.0, 0.0]
    assert missing[16:20] == [0.0, 0.0, 0.0, 0.0]

    nested = converter.build_action(
        record(0, {}, {"commands": {"SET_LEFT_GRIPPER_SPEED": 0, "SET_RIGHT_FORCE": 0}}),
        record(1, {}),
    )
    assert nested[6:10] == [0.0, 0.0, 1.0, 0.0]
    assert nested[16:20] == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "4"])
def test_non_finite_or_non_numeric_present_command_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        converter.build_action(
            record(0, {}, {"commands.SET_LEFT_FORCE": value}), record(1, {})
        )


def test_all_active_rgb_depth_and_extra_streams_become_features(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    features = converter.build_features(source.videos)
    assert list(key for key in features if key.startswith("observation.images.")) == [
        "observation.images.rgbd_head_color",
        "observation.images.hand_left",
        "observation.images.hand_right",
        "observation.images.head_right",
        "observation.images.rgbd_head_depth",
    ]
    depth = features["observation.images.rgbd_head_depth"]
    assert depth["shape"] == (32, 48, 1)
    assert depth["info"]["is_depth_map"] is True
    assert features["action"]["shape"] == (20,)
    assert features["action"]["names"]["axes"] == converter.ACTION_NAMES
    assert "joint_position.Joint_Left_Gripper" in features["observation.state"]["names"]["axes"]


def test_load_source_episode_preserves_all_valid_streams_and_ignores_empty_reference(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    rows = [json.loads(line) for line in (source.path / "manifest.jsonl").read_text().splitlines()]
    for row in rows:
        if "frame_idx" in row:
            row["videos"]["unused_camera"] = {
                "path": None, "frame_id": None, "is_repeat": False
            }
    (source.path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    def probe(path: Path, stream: str) -> converter.VideoSource:
        return converter.VideoSource(stream, path, 48, 32, 30.0, 3, False)

    loaded = converter.load_source_episode(source.path, "english task", video_probe=probe)
    assert loaded.task == "english task"
    assert set(loaded.videos) == set(source.videos)
    assert loaded.videos["rgbd_head_depth"].is_depth is True
    assert "unused_camera" not in loaded.videos


def test_last_frame_has_no_action() -> None:
    source = converter.SourceEpisode(Path("episode_1"), 1, "task", 30, {}, [{}, {}, {}], {})
    assert converter.converted_episode_length(source) == 2
    with pytest.raises(ValueError, match="at least two"):
        converter.converted_episode_length(replace(source, records=[{}]))


def test_schema_baseline_uses_largest_group_and_first_on_tie(tmp_path: Path) -> None:
    first = make_source(tmp_path / "a", 1, width=48)
    second = make_source(tmp_path / "b", 2, width=64)
    third = make_source(tmp_path / "c", 3, width=64)
    accepted, skipped = converter.select_compatible_episodes([first, second, third])
    assert [item.source_episode_id for item in accepted] == [2, 3]
    assert [(item.source_episode_id, reason) for item, reason in skipped] == [(1, "schema_outlier")]
    accepted, _ = converter.select_compatible_episodes([first, second])
    assert accepted == [first]


def test_state_rejects_legacy_14d_output(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / converter.CONVERSION_STATE_FILENAME).write_text(
        json.dumps({"version": 1, "conversion_config": {"action_dim": 14}, "episodes": []}),
        encoding="utf-8",
    )
    with pytest.raises(converter.StateConflictError, match="20-D action schema"):
        converter.load_conversion_state(root)


def test_incremental_state_rejects_config_revision_missing_rename_and_duplicates(tmp_path: Path) -> None:
    task = tmp_path / "task"
    first = make_source(task, 1)
    second = make_source(task, 2)
    config = converter.ConverterConfig()
    state = converter.new_conversion_state(
        task, "autolife/task", config, converter.schema_for_episode(first), first.task
    )
    state["episodes"].append(converter._state_entry(first, 0))

    assert converter.validate_incremental_state(
        state, [first, second], source_task=task, repo_id="autolife/task", config=config
    ) == [second]
    with pytest.raises(converter.StateConflictError, match="settings conflict"):
        converter.validate_incremental_state(
            state, [first, second], source_task=task, repo_id="autolife/task",
            config=converter.ConverterConfig(video_crf=24),
        )
    changed_task = replace(second, task="不同语言")
    with pytest.raises(converter.StateConflictError, match="stored task language"):
        converter.validate_incremental_state(
            state, [first, changed_task], source_task=task, repo_id="autolife/task", config=config
        )
    with pytest.raises(converter.StateConflictError, match="missing"):
        converter.validate_incremental_state(
            state, [second], source_task=task, repo_id="autolife/task", config=config
        )
    renamed = replace(first, path=first.path.with_name("episode_000099"))
    with pytest.raises(converter.StateConflictError, match="changed name"):
        converter.validate_incremental_state(
            state, [renamed, second], source_task=task, repo_id="autolife/task", config=config
        )
    first.records[0]["t_ns"] += 1
    (first.path / "manifest.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(converter.StateConflictError, match="source changed"):
        converter.validate_incremental_state(
            state, [first, second], source_task=task, repo_id="autolife/task", config=config
        )
    duplicate = replace(second, source_episode_id=first.source_episode_id)
    with pytest.raises(converter.StateConflictError, match="duplicate source episode id"):
        converter.validate_incremental_state(
            state, [first, duplicate], source_task=task, repo_id="autolife/task", config=config
        )


def test_initial_publish_uses_sibling_staging_and_preserves_preflight_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = tmp_path / "task"
    sources = [make_source(task, 1), make_source(task, 2)]
    output = tmp_path / "dataset"
    output.mkdir()
    report = b'{"status":"preflight"}\n'
    (output / "conversion_report.json").write_bytes(report)
    calls = []

    monkeypatch.setattr(converter, "_heavy_runtime", lambda: {})

    def append_one(runtime, root, repo_id, episode, config, *, create):
        root.mkdir(parents=True, exist_ok=True)
        calls.append((root, episode.source_episode_id, create))
        return len(calls) - 1

    monkeypatch.setattr(converter, "_append_one", append_one)
    result = converter.convert_task(
        sources, output, "autolife/task", source_task=task,
    )
    assert result.created is True
    assert result.appended_episode_indices == (0, 1)
    assert calls[0][0].parent == output.parent and calls[0][0] != output
    assert [item[2] for item in calls] == [True, False]
    assert (output / "conversion_report.json").read_bytes() == report
    state = converter.load_conversion_state(output)
    assert state is not None
    assert state["conversion_config"]["action_dim"] == 20
    assert state["stored_task"] == "stored task"
    assert [item["source_episode_id"] for item in state["episodes"]] == [1, 2]
    assert not list(tmp_path.glob(".dataset.staging-*"))


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lerobot") is None,
    reason="optional LeRobot conversion environment is not installed",
)
def test_optional_lerobot_dependency_is_discoverable() -> None:
    runtime = converter._heavy_runtime()
    assert runtime["LeRobotDataset"] is not None
