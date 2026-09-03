from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
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


def test_dual_actions_use_exact_axes_next_state_and_joint_grippers() -> None:
    current = record(0, {}, {"commands.SET_LEFT_FORCE": float("nan")})
    following = record(1, {})
    qpos = [100.0 + index for index in range(23)]
    following["joints"]["position"] = qpos

    hybrid = converter.build_action(current, following, action_mode=converter.BODY_JOINT_EEF)
    whole = converter.build_action(current, following, action_mode=converter.WHOLE_BODY_JOINT)

    assert len(hybrid) == 21
    assert hybrid[:4] == qpos[:4]
    assert hybrid[4:10] == pytest.approx([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert hybrid[10] == 118.0
    assert hybrid[11:17] == pytest.approx([2.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert hybrid[17] == 119.0
    assert hybrid[18:] == [120.0, 121.0, 122.0]
    assert whole == qpos
    assert converter.BODY_JOINT_EEF_ACTION_NAMES[10] == "left.gripper"
    assert converter.BODY_JOINT_EEF_ACTION_NAMES[17] == "right.gripper"


def test_terminal_actions_hold_joints_and_zero_eef_delta() -> None:
    terminal = record(7, {}, {"commands.SET_RIGHT_FORCE": "ignored"})
    qpos = [-10.0 + index * 0.5 for index in range(23)]
    terminal["joints"]["position"] = qpos

    hybrid = converter.build_action(terminal, action_mode=converter.BODY_JOINT_EEF)
    whole = converter.build_action(terminal, action_mode=converter.WHOLE_BODY_JOINT)

    assert hybrid[:4] == qpos[:4]
    assert hybrid[4:10] == [0.0] * 6
    assert hybrid[10] == qpos[18]
    assert hybrid[11:17] == [0.0] * 6
    assert hybrid[17] == qpos[19]
    assert hybrid[18:] == qpos[20:23]
    assert whole == qpos


def test_relative_pose_delta_is_expressed_in_current_local_frame() -> None:
    half = math.sqrt(0.5)
    current = [0.0, 0.0, 0.0, 0.0, 0.0, half, half]
    following = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    delta = converter.relative_pose_delta(current, following)

    assert delta[:3] == pytest.approx([0.0, -1.0, 0.0], abs=1e-7)
    assert delta[3:] == pytest.approx([0.0, 0.0, math.pi / 2], abs=1e-7)
    assert converter.relative_pose_delta(current, [*current[:3], 0.0, 0.0, -half, -half]) == pytest.approx(
        [0.0] * 6, abs=1e-7
    )


def test_make_frame_keeps_shared_38d_observation_and_terminal_source_fields() -> None:
    np = pytest.importorskip("numpy")
    source = converter.SourceEpisode(
        Path("episode_000007"), 7, "english task", 30, {}, [record(3, {})], {},
    )
    source.records[0]["joints"]["position"] = [float(index) for index in range(23)]

    whole = converter._make_frame(source, 0, {}, np, converter.WHOLE_BODY_JOINT)
    hybrid = converter._make_frame(source, 0, {}, np, converter.BODY_JOINT_EEF)

    assert whole["observation.state"].shape == (38,)
    assert np.array_equal(whole["observation.state"], hybrid["observation.state"])
    assert int(whole["source.frame_index"][0]) == 3
    assert int(whole["source.timestamp_ns"][0]) == 1_000_000_003
    assert int(whole["source.episode_id"][0]) == 7
    assert whole["action"].shape == (23,)
    assert hybrid["action"].shape == (21,)


def test_training_features_are_three_rgb_streams_with_mode_specific_actions(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    features = converter.build_features(source.videos, converter.BODY_JOINT_EEF)
    assert list(key for key in features if key.startswith("observation.images.")) == [
        "observation.images.rgbd_head_color",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]
    assert features["action"]["shape"] == (21,)
    assert features["action"]["names"]["axes"] == converter.BODY_JOINT_EEF_ACTION_NAMES
    whole = converter.build_features(source.videos, converter.WHOLE_BODY_JOINT)
    assert whole["action"]["shape"] == (23,)
    assert "joint_position.Joint_Left_Gripper" in features["observation.state"]["names"]["axes"]
    assert features["observation.state"]["shape"] == (38,)
    assert converter.JOINT_NAMES[-3:] == [
        "Joint_Neck_Yaw", "Joint_Neck_Pitch", "Joint_Neck_Roll",
    ]


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


def test_load_source_episode_default_task_is_normalized_english_title(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source.path / "task_meta.json").write_text(
        json.dumps({"task_title": "pick_the_egg_tart", "task_description": "拿起蛋挞"}),
        encoding="utf-8",
    )

    loaded = converter.load_source_episode(
        source.path,
        video_probe=lambda path, stream: source.videos[stream],
    )

    assert loaded.task == "pick the egg tart"


def test_virtual_source_loads_full_videos_and_decoder_skips_removed_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(tmp_path)
    source_ids = [0, 2, 4]
    rows = [json.loads(line) for line in (source.path / "manifest.jsonl").read_text().splitlines()]
    for record_row, source_id in zip(rows[1:4], source_ids, strict=True):
        for entry in record_row["videos"].values():
            entry["source_frame_id"] = source_id
    (source.path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (source.path / "CUT_INFO.json").write_text(json.dumps({
        "formatVersion": 2,
        "videoMaterialization": "full_source",
        "sourceFrames": 5,
    }), encoding="utf-8")

    def probe(path: Path, stream: str) -> converter.VideoSource:
        return converter.VideoSource(stream, path, 48, 32, 30.0, 5, False)

    loaded = converter.load_source_episode(source.path, "english task", video_probe=probe)
    assert {video.frames for video in loaded.videos.values()} == {5}

    calls: dict[str, list[int]] = {}

    class Image:
        def __init__(self, shape):
            self.shape = shape

    class Reader:
        def __init__(self, video, _runtime):
            self.video = video
            calls[video.stream] = []

        def read_at(self, frame_index):
            calls[self.video.stream].append(frame_index)
            channels = 1 if self.video.is_depth else 3
            return Image((self.video.height, self.video.width, channels))

        def read(self):
            raise AssertionError("virtual video should not require an end-of-file read")

        def close(self):
            pass

    class Dataset:
        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

    monkeypatch.setattr(converter, "_VideoReader", Reader)
    monkeypatch.setattr(converter, "_make_frame", lambda *_args: {"frame": len(dataset.frames)})
    dataset = Dataset()
    converter._add_frames(dataset, loaded, {"np": object()})

    assert len(dataset.frames) == 3
    assert calls and all(indices == source_ids for indices in calls.values())


def test_all_source_frames_are_retained_including_single_frame_episode() -> None:
    source = converter.SourceEpisode(Path("episode_1"), 1, "task", 30, {}, [{}, {}, {}], {})
    assert converter.converted_episode_length(source) == 3
    assert converter.converted_episode_length(replace(source, records=[{}])) == 1


def test_hybrid_action_rewrite_updates_embedded_huggingface_schema(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    pa = pytest.importorskip("pyarrow")
    np = pytest.importorskip("numpy")
    from datasets.arrow_dataset import update_metadata_with_features

    first = make_source(tmp_path / "task_a")
    second = make_source(tmp_path / "task_b")
    second.records[1]["joints"]["position"] = [200.0 + index for index in range(23)]
    owner_features = datasets.Features({
        "episode_index": datasets.Value("int64"),
        "source.episode_id": datasets.Value("int64"),
        "source.frame_index": datasets.Value("int64"),
        "action": datasets.Sequence(datasets.Value("float32"), length=23),
    })
    table = datasets.Dataset.from_dict(
        {
            "episode_index": [0, 1],
            "source.episode_id": [1, 1],
            "source.frame_index": [0, 0],
            "action": [[0.0] * 23, [0.0] * 23],
        },
        features=owner_features,
    ).data.table
    hybrid_features = datasets.Features({
        "episode_index": datasets.Value("int64"),
        "source.episode_id": datasets.Value("int64"),
        "source.frame_index": datasets.Value("int64"),
        "action": datasets.Sequence(datasets.Value("float32"), length=21),
    })

    rewritten, actions, _ = converter._replace_action_column(
        {"pa": pa, "np": np, "update_metadata_with_features": update_metadata_with_features},
        table,
        {0: first, 1: second},
        hybrid_features,
    )

    metadata = json.loads(rewritten.schema.metadata[b"huggingface"])
    assert rewritten.schema.field("action").type.list_size == 21
    assert metadata["info"]["features"]["action"]["length"] == 21
    assert actions.shape == (2, 21)
    assert actions[0].tolist() == pytest.approx(
        converter.build_action(first.records[0], first.records[1], action_mode=converter.BODY_JOINT_EEF)
    )
    assert actions[1].tolist() == pytest.approx(
        converter.build_action(second.records[0], second.records[1], action_mode=converter.BODY_JOINT_EEF)
    )
    assert actions[0, 10] != actions[1, 10]


def test_video_encoding_mode_accepts_parallel_without_changing_default() -> None:
    assert converter.ConverterConfig().video_encoding_mode == "sequential"
    assert converter.ConverterConfig(video_encoding_mode="parallel").video_encoding_mode == "parallel"
    with pytest.raises(ValueError, match="sequential, parallel, or streaming"):
        converter.ConverterConfig(video_encoding_mode="invalid")


def test_video_workers_are_operational_and_do_not_change_incremental_state() -> None:
    config = converter.ConverterConfig(video_workers=5)
    assert config.video_workers == 5
    assert "video_workers" not in config.state_value()
    with pytest.raises(ValueError, match="positive"):
        converter.ConverterConfig(video_workers=0)


def test_streaming_video_workers_above_three_run_training_and_auxiliary_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(
        tmp_path,
        extra_streams=("rgbd_head_depth", "head_left", "head_right"),
    )
    rendezvous = threading.Barrier(2)
    worker_counts: dict[str, int] = {}
    committed: list[list[dict]] = []

    def append_one(
        runtime, root, repo_id, episode, config, *, create, action_mode, dataset_opened,
    ):
        worker_counts["training"] = config.video_workers
        dataset_opened()
        rendezvous.wait(timeout=2)
        return 0

    def encode_auxiliary(runtime, root, index, episode, config):
        worker_counts["auxiliary"] = config.video_workers
        rendezvous.wait(timeout=2)
        return [
            {"episode_index": index, "stream": stream}
            for stream in sorted(set(episode.videos) - set(converter.TRAINING_VIDEO_STREAMS))
        ]

    monkeypatch.setattr(converter, "_append_one", append_one)
    monkeypatch.setattr(converter, "_encode_auxiliary_episode", encode_auxiliary)
    monkeypatch.setattr(
        converter, "_commit_auxiliary_rows",
        lambda runtime, root, index, episode, rows: committed.append(list(rows)),
    )
    monkeypatch.setattr(
        converter, "source_episode_signature", lambda episode: {"digest": "stable"},
    )

    index = converter._append_episode_outputs(
        {}, tmp_path / "output", "local/test", source,
        converter.ConverterConfig(video_encoding_mode="streaming", video_workers=5),
        create=True, expected_index=0,
    )

    assert index == 0
    assert worker_counts == {"training": 3, "auxiliary": 2}
    assert len(committed) == 1
    assert {row["stream"] for row in committed[0]} == {
        "rgbd_head_depth", "head_left", "head_right",
    }


def test_three_video_workers_keep_training_and_auxiliary_groups_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        converter, "_append_one",
        lambda *args, **kwargs: calls.append("training") or 0,
    )
    monkeypatch.setattr(
        converter, "_ensure_auxiliary_episode",
        lambda *args, **kwargs: calls.append("auxiliary"),
    )
    monkeypatch.setattr(
        converter, "_encode_auxiliary_episode",
        lambda *args, **kwargs: pytest.fail("sequential path encoded auxiliary directly"),
    )

    converter._append_episode_outputs(
        {}, tmp_path / "output", "local/test", source,
        converter.ConverterConfig(video_encoding_mode="streaming", video_workers=3),
        create=True, expected_index=0,
    )

    assert calls == ["training", "auxiliary"]


def test_parallel_auxiliary_does_not_start_when_training_dataset_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(
        tmp_path,
        extra_streams=("rgbd_head_depth", "head_left", "head_right"),
    )
    auxiliary_calls: list[bool] = []
    monkeypatch.setattr(
        converter, "_append_one",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("open failed")),
    )
    monkeypatch.setattr(
        converter, "_encode_auxiliary_episode",
        lambda *args, **kwargs: auxiliary_calls.append(True) or [],
    )
    monkeypatch.setattr(
        converter, "source_episode_signature", lambda episode: {"digest": "stable"},
    )

    with pytest.raises(RuntimeError, match="open failed"):
        converter._append_episode_outputs(
            {}, tmp_path / "output", "local/test", source,
            converter.ConverterConfig(video_encoding_mode="streaming", video_workers=6),
            create=True, expected_index=0,
        )

    assert auxiliary_calls == []


def test_open_dataset_enables_async_image_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(tmp_path)
    calls: list[dict] = []

    class Dataset:
        @classmethod
        def create(cls, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(converter, "_encoders", lambda *args: (None, None))
    converter._open_dataset(
        {"LeRobotDataset": Dataset},
        tmp_path / "output",
        "local/test",
        source,
        converter.ConverterConfig(video_workers=3),
        create=True,
    )

    assert calls[0]["image_writer_threads"] == 3


def test_open_dataset_streaming_queue_holds_the_complete_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(tmp_path)
    calls: list[dict] = []

    class Dataset:
        @classmethod
        def create(cls, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(converter, "_encoders", lambda *args: (None, None))
    converter._open_dataset(
        {"LeRobotDataset": Dataset},
        tmp_path / "output",
        "local/test",
        source,
        converter.ConverterConfig(
            video_encoding_mode="streaming", encoder_queue_maxsize=1,
        ),
        create=True,
    )

    assert calls[0]["streaming_encoding"] is True
    assert calls[0]["encoder_queue_maxsize"] == len(source.records) + 1
    assert calls[0]["image_writer_threads"] == 0


def test_auxiliary_runtime_threads_are_explicitly_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    cv2 = type("CV2", (), {"setNumThreads": lambda _self, value: calls.append(("cv2", value))})()
    pa = type(
        "PyArrow",
        (),
        {
            "set_cpu_count": lambda _self, value: calls.append(("pa_cpu", value)),
            "set_io_thread_count": lambda _self, value: calls.append(("pa_io", value)),
        },
    )()
    monkeypatch.setenv("DATA_AUTOPRO_AUX_THREADS", "2")
    monkeypatch.delitem(converter.sys.modules, "torch", raising=False)

    converter._configure_auxiliary_threads(cv2, pa)

    assert calls == [("cv2", 2), ("pa_cpu", 2), ("pa_io", 2)]


@pytest.mark.parametrize("value", ("0", "invalid"))
def test_auxiliary_runtime_threads_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv("DATA_AUTOPRO_AUX_THREADS", value)
    with pytest.raises(ValueError, match="positive integer"):
        converter._configure_auxiliary_threads(object(), object())


@pytest.mark.parametrize(
    ("mode", "expected_parallel"),
    (("sequential", False), ("parallel", True), ("streaming", False)),
)
def test_append_one_selects_requested_video_encoding_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_parallel: bool,
) -> None:
    source = make_source(tmp_path)
    calls: list[bool] = []

    class Dataset:
        meta = type("Meta", (), {"total_episodes": 0})()

        def save_episode(self, *, parallel_encoding: bool) -> None:
            calls.append(parallel_encoding)

        def finalize(self) -> None:
            pass

    monkeypatch.setattr(converter, "_open_dataset", lambda *args, **kwargs: Dataset())
    monkeypatch.setattr(converter, "_add_frames", lambda *args, **kwargs: None)
    monkeypatch.setattr(converter, "_verify_episode", lambda *args, **kwargs: None)

    index = converter._append_one(
        {}, tmp_path / "output", "autolife/task", source,
        converter.ConverterConfig(video_encoding_mode=mode), create=True,
    )

    assert index == 0
    assert calls == [expected_parallel]


def test_schema_baseline_uses_largest_group_and_first_on_tie(tmp_path: Path) -> None:
    first = make_source(tmp_path / "a", 1, width=48)
    second = make_source(tmp_path / "b", 2, width=64)
    third = make_source(tmp_path / "c", 3, width=64)
    accepted, skipped = converter.select_compatible_episodes([first, second, third])
    assert [item.source_episode_id for item in accepted] == [2, 3]
    assert [(item.source_episode_id, reason) for item, reason in skipped] == [(1, "schema_outlier")]
    accepted, _ = converter.select_compatible_episodes([first, second])
    assert accepted == [first]


def test_auxiliary_schema_differences_do_not_reject_training_compatible_episode(tmp_path: Path) -> None:
    first = make_source(tmp_path / "a", 1)
    second = make_source(tmp_path / "b", 2)
    second_videos = dict(second.videos)
    depth = second_videos["rgbd_head_depth"]
    second_videos["rgbd_head_depth"] = replace(depth, width=96)
    second = replace(second, videos=second_videos)

    accepted, skipped = converter.select_compatible_episodes([first, second])

    assert accepted == [first, second]
    assert skipped == []


def test_empty_auxiliary_set_still_creates_shareable_index(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = make_source(tmp_path / "source", extra_streams=())
    root = tmp_path / "owner"

    converter._ensure_auxiliary_episode(
        {"pa": pa, "pq": pq}, root, 0, source, converter.ConverterConfig(),
    )

    index = root / converter.AUXILIARY_INDEX_PATH
    assert index.is_file()
    assert pq.read_table(index).num_rows == 0


def test_state_rejects_legacy_14d_output(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / converter.CONVERSION_STATE_FILENAME).write_text(
        json.dumps({"version": 1, "conversion_config": {"action_dim": 14}, "episodes": []}),
        encoding="utf-8",
    )
    with pytest.raises(converter.StateConflictError, match="v2/20-D output"):
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


def test_merged_state_uses_relative_paths_for_duplicate_ids_and_incremental_validation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    first = replace(make_source(source_root / "task_a", 1), task="task a")
    second = replace(make_source(source_root / "task_b", 1), task="task b")
    third = replace(make_source(source_root / "task_c", 1), task="task c")
    config = converter.ConverterConfig()
    state = converter.new_conversion_state(
        source_root,
        "autolife/whole_body_joint",
        config,
        converter.schema_for_episode(first),
        None,
        dataset_layout=converter.MERGED_DATASET_LAYOUT,
    )

    assert converter.validate_incremental_state(
        state,
        [first, second],
        source_task=source_root,
        repo_id="autolife/whole_body_joint",
        config=config,
        dataset_layout=converter.MERGED_DATASET_LAYOUT,
    ) == [first, second]
    converter._append_state_episode(
        state, first, 0, dataset_layout=converter.MERGED_DATASET_LAYOUT,
        source_root=source_root,
    )
    converter._append_state_episode(
        state, second, 1, dataset_layout=converter.MERGED_DATASET_LAYOUT,
        source_root=source_root,
    )

    assert [entry["source_relative_path"] for entry in state["episodes"]] == [
        "task_a/episode_000001", "task_b/episode_000001",
    ]
    assert [entry["stored_task"] for entry in state["episodes"]] == ["task a", "task b"]
    assert state["stored_tasks"] == ["task a", "task b"]
    assert converter.validate_incremental_state(
        state,
        [first, second],
        source_task=source_root,
        repo_id="autolife/whole_body_joint",
        config=config,
        dataset_layout=converter.MERGED_DATASET_LAYOUT,
    ) == []
    assert converter.validate_incremental_state(
        state,
        [first, second, third],
        source_task=source_root,
        repo_id="autolife/whole_body_joint",
        config=config,
        dataset_layout=converter.MERGED_DATASET_LAYOUT,
    ) == [third]

    common = {
        "source_task": source_root,
        "repo_id": "autolife/whole_body_joint",
        "config": config,
        "dataset_layout": converter.MERGED_DATASET_LAYOUT,
    }
    with pytest.raises(converter.StateConflictError, match="missing"):
        converter.validate_incremental_state(state, [first], **common)
    with pytest.raises(converter.StateConflictError, match="changed stored task"):
        converter.validate_incremental_state(
            state, [replace(first, task="changed"), second], **common,
        )
    with pytest.raises(converter.StateConflictError, match="changed numeric id"):
        converter.validate_incremental_state(
            state, [first, replace(second, source_episode_id=2)], **common,
        )
    with pytest.raises(converter.StateConflictError, match="duplicate source relative path"):
        converter.validate_incremental_state(
            state, [first, replace(second, path=first.path)], **common,
        )
    renamed = replace(second, path=source_root / "task_renamed" / second.path.name)
    with pytest.raises(converter.StateConflictError, match="missing"):
        converter.validate_incremental_state(state, [first, renamed], **common)


def test_merged_recovery_uses_pending_relative_path_when_numeric_ids_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    first = replace(make_source(source_root / "task_a", 1), task="task a")
    second = replace(make_source(source_root / "task_b", 1), task="task b")
    output = tmp_path / "output"
    output.mkdir()
    config = converter.ConverterConfig()
    state = converter.new_conversion_state(
        source_root, "autolife/whole_body_joint", config,
        converter.schema_for_episode(first), None,
        dataset_layout=converter.MERGED_DATASET_LAYOUT,
    )
    converter._append_state_episode(
        state, first, 0, dataset_layout=converter.MERGED_DATASET_LAYOUT,
        source_root=source_root,
    )
    converter._write_state(output, state)
    converter._mark_pending_episode(
        output, state, second, 1, dataset_layout=converter.MERGED_DATASET_LAYOUT,
        source_root=source_root,
    )
    verified: list[Path] = []
    monkeypatch.setattr(converter, "_dataset_episode_count", lambda *args: 2)
    monkeypatch.setattr(
        converter, "_verify_episode",
        lambda runtime, root, repo_id, index, episode, action_mode, video_workers: verified.append(episode.path),
    )
    monkeypatch.setattr(converter, "_ensure_auxiliary_episode", lambda *args: None)

    recovered = converter._recover_uncommitted_state(
        {}, output, "autolife/whole_body_joint", state, [first, second], config,
        dataset_layout=converter.MERGED_DATASET_LAYOUT, source_root=source_root,
    )

    assert recovered == [1]
    assert verified == [second.path]
    assert state["episodes"][1]["source_relative_path"] == "task_b/episode_000001"
    assert "pending_episode" not in state


def test_convert_merged_bundle_uses_shared_roots_layout_and_path_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    episodes = [
        replace(make_source(source_root / "task_a", 1), task="task a"),
        replace(make_source(source_root / "task_b", 1), task="task b"),
    ]
    bundle_root = tmp_path / "bundle"
    calls: dict[str, tuple] = {}

    def owner_convert(items, output_root, repo_id, **kwargs):
        calls["owner"] = (output_root, repo_id, kwargs)
        state = converter.new_conversion_state(
            kwargs["source_task"], repo_id, kwargs["config"],
            converter.schema_for_episode(items[0]), None,
            dataset_layout=kwargs["dataset_layout"],
        )
        for index, episode in enumerate(items):
            converter._append_state_episode(
                state, episode, index, dataset_layout=converter.MERGED_DATASET_LAYOUT,
                source_root=source_root,
            )
        return converter.ConversionResult(
            output_root, True, (), (0, 1), (), (), state,
        )

    def hybrid_derive(items, owner_root, hybrid_root, repo_id, **kwargs):
        calls["hybrid"] = (owner_root, hybrid_root, repo_id, kwargs)
        owner_state = calls["owner_result"].state
        state = converter.new_conversion_state(
            source_root, repo_id, kwargs["config"], owner_state["schema"], None,
            converter.BODY_JOINT_EEF, os.path.relpath(owner_root, start=hybrid_root),
            converter.MERGED_DATASET_LAYOUT,
        )
        state["episodes"] = [dict(entry) for entry in owner_state["episodes"]]
        return converter.ConversionResult(
            hybrid_root, True, (), (0, 1), (), (), state,
        )

    def owner_wrapper(*args, **kwargs):
        result = owner_convert(*args, **kwargs)
        calls["owner_result"] = result
        return result

    monkeypatch.setattr(converter, "convert_task", owner_wrapper)
    monkeypatch.setattr(converter, "derive_hybrid_dataset", hybrid_derive)

    result = converter.convert_merged_bundle(
        episodes, bundle_root, "autolife", source_root=source_root,
    )

    owner_root, owner_repo, owner_kwargs = calls["owner"]
    hybrid_owner, hybrid_root, hybrid_repo, hybrid_kwargs = calls["hybrid"]
    assert owner_root == bundle_root / converter.WHOLE_BODY_JOINT
    assert hybrid_owner == owner_root
    assert hybrid_root == bundle_root / converter.BODY_JOINT_EEF
    assert owner_repo == "autolife/whole_body_joint"
    assert hybrid_repo == "autolife/body_joint_eef"
    assert owner_kwargs["dataset_layout"] == converter.MERGED_DATASET_LAYOUT
    assert owner_kwargs["source_task"] == source_root.resolve()
    assert hybrid_kwargs["source_task"] == source_root.resolve()
    assert [entry["source_relative_path"] for entry in result.ledger["episodes"]] == [
        "task_a/episode_000001", "task_b/episode_000001",
    ]
    assert result.ledger["stored_tasks"] == ["task a", "task b"]
    assert all(
        status["status"] == "committed" and status["outcome"] == "appended"
        for entry in result.ledger["episodes"] for status in entry["modes"].values()
    )
    assert (bundle_root / ".bundle/bundle_state.json").is_file()


def test_merged_hybrid_noop_avoids_heavy_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = tmp_path / "source"
    source = replace(make_source(source_root / "task_a", 1), task="task a")
    owner = tmp_path / converter.WHOLE_BODY_JOINT
    hybrid = tmp_path / converter.BODY_JOINT_EEF
    (owner / "videos").mkdir(parents=True)
    (owner / "auxiliary").mkdir()
    hybrid.mkdir()
    config = converter.ConverterConfig()
    owner_state = converter.new_conversion_state(
        source_root, "autolife/whole_body_joint", config,
        converter.schema_for_episode(source), None,
        dataset_layout=converter.MERGED_DATASET_LAYOUT,
    )
    converter._append_state_episode(
        owner_state, source, 0, dataset_layout=converter.MERGED_DATASET_LAYOUT,
        source_root=source_root,
    )
    converter._write_state(owner, owner_state)
    shared_owner = os.path.relpath(owner, start=hybrid)
    hybrid_state = converter.new_conversion_state(
        source_root, "autolife/body_joint_eef", config, owner_state["schema"], None,
        converter.BODY_JOINT_EEF, shared_owner, converter.MERGED_DATASET_LAYOUT,
    )
    hybrid_state["episodes"] = [dict(owner_state["episodes"][0])]
    hybrid_state["stored_tasks"] = [source.task]
    converter._write_state(hybrid, hybrid_state)
    (hybrid / "videos").symlink_to(
        os.path.relpath(owner / "videos", start=hybrid), target_is_directory=True,
    )
    (hybrid / "auxiliary").symlink_to(
        os.path.relpath(owner / "auxiliary", start=hybrid), target_is_directory=True,
    )
    monkeypatch.setattr(
        converter, "_heavy_runtime",
        lambda: pytest.fail("no-op hybrid derivation loaded heavy dependencies"),
    )

    result = converter.derive_hybrid_dataset(
        [source], owner, hybrid, "autolife/body_joint_eef",
        source_task=source_root, config=config,
    )

    assert result.existing_episode_indices == (0,)
    assert result.appended_episode_indices == ()


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

    def append_one(runtime, root, repo_id, episode, config, *, create, action_mode):
        root.mkdir(parents=True, exist_ok=True)
        calls.append((root, episode.source_episode_id, create))
        return len(calls) - 1

    monkeypatch.setattr(converter, "_append_one", append_one)
    monkeypatch.setattr(converter, "_ensure_auxiliary_episode", lambda *args, **kwargs: None)
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
    assert state["conversion_config"]["action_dim"] == 23
    assert state["action_mode"] == converter.WHOLE_BODY_JOINT
    assert state["stored_task"] == "stored task"
    assert [item["source_episode_id"] for item in state["episodes"]] == [1, 2]
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_promotion_validates_all_consumers_before_exposing_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_task = tmp_path / "source"
    source = make_source(source_task, 1)
    owner = tmp_path / converter.WHOLE_BODY_JOINT / "task"
    (owner / "meta").mkdir(parents=True)
    info_path = owner / "meta/info.json"
    info_path.write_text('{"features":{"sentinel":{}}}\n', encoding="utf-8")
    state = converter.new_conversion_state(
        source_task,
        "autolife/owner",
        converter.ConverterConfig(),
        converter.schema_for_episode(source),
        source.task,
    )
    state["episodes"].append(converter._state_entry(source, 0))
    converter.atomic_write_json(owner / converter.CONVERSION_STATE_FILENAME, state)
    signature = converter.source_episode_signature(source)["digest"]
    monkeypatch.setattr(converter, "_heavy_runtime", lambda: {})
    monkeypatch.setattr(
        converter,
        "_read_auxiliary_rows",
        lambda runtime, root: [{
            "episode_index": 0,
            "source_episode_id": 1,
            "stream": "head_right",
            "frame_count": 3,
            "source_signature": signature,
            "width": 48,
            "height": 32,
            "fps": 30.0,
            "is_depth": False,
            "feature_info_json": "{}",
        }],
    )
    original = info_path.read_bytes()

    with pytest.raises(converter.StateConflictError, match="consumer is missing"):
        converter.promote_aux_videos(
            owner, ["head_right"], consumer_roots=[tmp_path / "missing-consumer"],
        )

    assert info_path.read_bytes() == original


def test_promotion_rejects_consumer_with_different_episode_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_task = tmp_path / "source"
    source = make_source(source_task, 1)
    owner = tmp_path / converter.WHOLE_BODY_JOINT / "task"
    consumer = tmp_path / converter.BODY_JOINT_EEF / "task"
    for root in (owner, consumer):
        (root / "meta").mkdir(parents=True)
        (root / "videos").mkdir()
    (owner / "meta/info.json").write_text('{"features":{}}\n', encoding="utf-8")
    owner_state = converter.new_conversion_state(
        source_task, "autolife/owner", converter.ConverterConfig(),
        converter.schema_for_episode(source), source.task,
    )
    owner_state["episodes"].append(converter._state_entry(source, 0))
    converter.atomic_write_json(owner / converter.CONVERSION_STATE_FILENAME, owner_state)
    consumer_state = converter.new_conversion_state(
        source_task, "autolife/consumer", converter.ConverterConfig(),
        converter.schema_for_episode(source), source.task, converter.BODY_JOINT_EEF,
        os.path.relpath(owner, start=consumer),
    )
    consumer_entry = converter._state_entry(source, 0)
    consumer_entry["source_episode_id"] = 99
    consumer_state["episodes"].append(consumer_entry)
    converter.atomic_write_json(consumer / converter.CONVERSION_STATE_FILENAME, consumer_state)
    (consumer / "videos").rmdir()
    (consumer / "videos").symlink_to(
        os.path.relpath(owner / "videos", start=consumer), target_is_directory=True,
    )
    signature = converter.source_episode_signature(source)["digest"]
    monkeypatch.setattr(converter, "_heavy_runtime", lambda: {})
    monkeypatch.setattr(
        converter, "_read_auxiliary_rows",
        lambda runtime, root: [{
            "episode_index": 0, "source_episode_id": 1, "stream": "head_right",
            "frame_count": 3, "source_signature": signature, "width": 48,
            "height": 32, "fps": 30.0, "is_depth": False, "feature_info_json": "{}",
        }],
    )

    with pytest.raises(converter.StateConflictError, match="metadata differs"):
        converter.promote_aux_videos(owner, ["head_right"], consumer_roots=[consumer])


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("source_relative_path", "other/episode_000001"), ("stored_task", "different task")),
)
def test_merged_promotion_rejects_consumer_path_or_per_episode_task_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, replacement: str,
) -> None:
    source_root = tmp_path / "source"
    source = replace(make_source(source_root / "task_a", 1), task="task a")
    owner = tmp_path / converter.WHOLE_BODY_JOINT
    consumer = tmp_path / converter.BODY_JOINT_EEF
    (owner / "videos").mkdir(parents=True)
    consumer.mkdir()
    (consumer / "videos").symlink_to(
        os.path.relpath(owner / "videos", start=consumer), target_is_directory=True,
    )
    config = converter.ConverterConfig()
    owner_state = converter.new_conversion_state(
        source_root, "autolife/whole_body_joint", config,
        converter.schema_for_episode(source), None,
        dataset_layout=converter.MERGED_DATASET_LAYOUT,
    )
    converter._append_state_episode(
        owner_state, source, 0, dataset_layout=converter.MERGED_DATASET_LAYOUT,
        source_root=source_root,
    )
    converter._write_state(owner, owner_state)
    consumer_state = converter.new_conversion_state(
        source_root, "autolife/body_joint_eef", config, owner_state["schema"], None,
        converter.BODY_JOINT_EEF, os.path.relpath(owner, start=consumer),
        converter.MERGED_DATASET_LAYOUT,
    )
    consumer_entry = dict(owner_state["episodes"][0])
    consumer_entry[field] = replacement
    consumer_state["episodes"] = [consumer_entry]
    consumer_state["stored_tasks"] = [str(consumer_entry["stored_task"])]
    converter._write_state(consumer, consumer_state)
    signature = converter.source_episode_signature(source)["digest"]
    monkeypatch.setattr(converter, "_heavy_runtime", lambda: {})
    monkeypatch.setattr(
        converter, "_read_auxiliary_rows",
        lambda runtime, root: [{
            "episode_index": 0, "source_episode_id": 1, "stream": "head_right",
            "frame_count": 3, "source_signature": signature, "width": 48,
            "height": 32, "fps": 30.0, "is_depth": False, "feature_info_json": "{}",
        }],
    )

    with pytest.raises(converter.StateConflictError, match=f"metadata differs from owner at {field}"):
        converter.promote_aux_videos(owner, ["head_right"], consumer_roots=[consumer])


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lerobot") is None,
    reason="optional LeRobot conversion environment is not installed",
)
def test_optional_lerobot_dependency_is_discoverable() -> None:
    runtime = converter._heavy_runtime()
    assert runtime["LeRobotDataset"] is not None
