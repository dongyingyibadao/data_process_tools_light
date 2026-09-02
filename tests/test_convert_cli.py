from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from lightworkbench import cli
from lightworkbench.validation import (
    EpisodeValidation,
    FINGERPRINT_VERSION,
    TIMESTAMP_REWRITE_VERSION,
    output_fingerprint,
    validate_episode,
)


STREAMS = ("rgbd_head_color", "hand_left", "hand_right", "rgbd_head_depth")


def fake_probe(path: Path) -> dict:
    return {
        "valid": path.is_file(),
        "frames": 3,
        "fps": 30.0,
        "width": 64,
        "height": 48,
        "codec": "ffv1" if "depth" in path.as_posix() else "h264",
        "pixelFormat": "gray16le" if "depth" in path.as_posix() else "yuv420p",
    }


def make_cleaned_episode(
    root: Path,
    relative: str,
    *,
    task_title: str = "close_fridge",
    description: str = "关闭冰箱",
    ranges: list[list[int]] | None = None,
    source_frames: int = 3,
    rewrite_version: int | None = TIMESTAMP_REWRITE_VERSION,
) -> Path:
    episode = root / relative
    episode.mkdir(parents=True)
    paths = {
        name: f"videos/{name}/{episode.name}{'.mkv' if 'depth' in name else '.mp4'}"
        for name in STREAMS
    }
    for value in paths.values():
        target = episode / value
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic-video")
    rows = [{
        "_type": "session_header",
        "fps_target": 30,
        "episode_id": int(episode.name.split("_")[-1]),
        "task_title": task_title,
        "task_description": description,
    }]
    for index in range(3):
        control = {}
        if index == 0:
            control["commands.SET_RIGHT_GRIPPER_SPEED"] = 0.0
        rows.append({
            "frame_idx": index,
            "t_wall": 10 + index / 30,
            "t_ns": 10_000_000_000 + index,
            "task": {"task_title": task_title, "episode_id": int(episode.name.split("_")[-1])},
            "videos": {
                name: {"path": value, "frame_id": index, "is_repeat": False}
                for name, value in paths.items()
            },
            "joints": {
                "position": [float(index)] * 23,
                "velocity": [0.0] * 23,
                "torque": [0.0] * 23,
            },
            "current_eef_pose": {
                name: {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
                for name in ("left_eef_pose", "right_eef_pose", "head_pose")
            },
            "target_eef_pose": {
                name: {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
                for name in ("left_eef_pose", "right_eef_pose", "head_pose")
            },
            "current_height_z": {"height_z": 0.0},
            "target_height_z": {"height_z": 0.0},
            "robot_state": {"state": 2},
            "control": control,
        })
    (episode / "manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8"
    )
    (episode / "task_meta.json").write_text(json.dumps({
        "task_title": task_title, "task_description": description,
    }, ensure_ascii=False), encoding="utf-8")
    removed = ranges or []
    output_frames = source_frames - sum(end - start for start, end in removed)
    audit = {
        "operationId": "operation",
        "revision": 1,
        "mode": "trim" if removed else "no_trim",
        "sourceRoot": str(root / "unmounted-source"),
        "episode": relative,
        "removedRanges": removed,
        "sourceFrames": source_frames,
        "outputFrames": output_frames,
        "sourceToken": "0" * 64,
        "completedAtUtc": "2026-08-30T00:00:00Z",
        "fingerprintVersion": FINGERPRINT_VERSION,
        "outputFingerprint": output_fingerprint(episode),
    }
    if rewrite_version is not None:
        audit["timestampRewriteVersion"] = rewrite_version
    (episode / "CUT_INFO.json").write_text(json.dumps(audit), encoding="utf-8")
    return episode


def refresh_audit_fingerprint(episode: Path) -> None:
    path = episode / "CUT_INFO.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["outputFingerprint"] = output_fingerprint(episode)
    path.write_text(json.dumps(audit), encoding="utf-8")


def test_parallel_preflight_is_bounded_and_preserves_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"episode_{index:06d}" for index in (10, 2, 3, 4)]
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0
    require_source_values: list[bool] = []

    def fake_validate(
        input_root: Path, episode: Path, *, require_source: bool,
    ) -> EpisodeValidation:
        del input_root
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            require_source_values.append(require_source)
        try:
            barrier.wait(timeout=2)
            if episode == paths[0]:
                time.sleep(0.05)
            return EpisodeValidation(episode, episode.name, "task")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(cli, "validate_episode", fake_validate)
    monkeypatch.setitem(cli.RESOURCES, "budget", 2)

    checked, workers = cli._validate_episodes(
        tmp_path,
        paths,
        require_source=True,
        requested_workers=10,
    )

    assert workers == 2
    assert max_active == 2
    assert require_source_values == [True] * len(paths)
    assert [item.path for item in checked] == paths


def test_strict_validation_fingerprint_control_mask_and_source_policy(tmp_path: Path) -> None:
    root = tmp_path / "cleaned"
    relative = "289/date/task/episode_000001"
    episode = make_cleaned_episode(root, relative)
    checked = validate_episode(root, episode, video_probe=fake_probe)
    assert checked.valid
    assert checked.warnings == ["source_root_unavailable"]
    assert set(checked.videos) == set(STREAMS)
    assert checked.videos["rgbd_head_depth"].is_depth
    assert checked.control_counts["commands.SET_RIGHT_GRIPPER_SPEED"] == 1
    assert checked.control_coverage["commands.SET_RIGHT_GRIPPER_SPEED"]["ratio"] == pytest.approx(1 / 3)

    strict = validate_episode(root, episode, require_source=True, video_probe=fake_probe)
    assert not strict.valid
    assert "source_root_unavailable" in strict.reasons

    (episode / "task_meta.json").write_text("{}", encoding="utf-8")
    tampered = validate_episode(root, episode, video_probe=fake_probe)
    assert "output_fingerprint_mismatch" in tampered.reasons


def test_validation_accepts_a_strict_cleaned_subroot(tmp_path: Path) -> None:
    root = tmp_path / "cleaned"
    episode = make_cleaned_episode(root, "289/date/task/episode_000001")

    checked = validate_episode(root / "289", episode, video_probe=fake_probe)

    assert checked.valid
    assert checked.relative_path == "date/task/episode_000001"
    assert checked.warnings == ["source_root_unavailable"]


def test_validation_rejects_missing_converter_fields_and_bad_video_frame_id(tmp_path: Path) -> None:
    root = tmp_path / "cleaned"
    episode = make_cleaned_episode(root, "task/episode_000001")
    manifest = episode / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[1].pop("target_eef_pose")
    rows[1]["videos"]["hand_left"]["frame_id"] = 99
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    refresh_audit_fingerprint(episode)

    checked = validate_episode(root, episode, video_probe=fake_probe)

    assert "invalid_target_eef_pose:0" in checked.reasons
    assert "video_frame_id_mismatch:hand_left:0" in checked.reasons


def test_merged_preflight_skips_aborted_session_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    aborted = make_cleaned_episode(root, "day/task/episode_000001")
    manifest = aborted / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "_type": "session_footer", "aborted": True, "reason": "FAILURE",
        }) + "\n")
    refresh_audit_fingerprint(aborted)
    make_cleaned_episode(root, "day/task/episode_000002")
    output = tmp_path / "merged"
    monkeypatch.setattr(
        "lightworkbench.validation.probe_video",
        lambda path, decoded=True: fake_probe(path),
    )

    code = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
        "--preflight-only",
    ])

    assert code == 1
    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert [item["episode"] for item in report["accepted"]] == [
        "day/task/episode_000002",
    ]
    assert [(item["episode"], item["reasons"]) for item in report["skipped"]] == [
        ("day/task/episode_000001", ["manifest_session_aborted"]),
    ]
    assert report["failed"] == []


def test_rejects_unverified_nested_timestamp_stitching(tmp_path: Path) -> None:
    root = tmp_path / "cleaned"
    relative = "289/date/task/episode_000002"
    episode = make_cleaned_episode(
        root,
        relative,
        ranges=[[0, 1], [2, 3]],
        source_frames=5,
        rewrite_version=None,
    )
    checked = validate_episode(root, episode, video_probe=fake_probe)
    assert not checked.valid
    assert "unverified_nested_timestamp_stitching" in checked.reasons


def test_preflight_natural_discovery_language_and_atomic_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "289/date/close_fridge/episode_000010")
    make_cleaned_episode(root, "289/date/close_fridge/episode_000002")
    output = tmp_path / "lerobot"
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main([
        "convert", "--input-root", str(root), "--output-root", str(output), "--preflight-only",
    ])
    assert code == 0
    summary = json.loads((output / "conversion_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {"tasks": 1, "accepted": 2, "skipped": 0, "failed": 0}
    assert summary["action_mode"] == "both"
    assert summary["task_language"] == "english"
    report_path = output / "289/date/close_fridge/conversion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task_title"] == "close_fridge"
    assert report["source_description"] == "关闭冰箱"
    assert report["stored_task"] == "close fridge"
    assert [Path(item["episode"]).name for item in report["accepted"]] == [
        "episode_000002", "episode_000010",
    ]
    assert not list(output.rglob("*.tmp"))


def test_include_episode_selects_exact_paths_and_reports_discovery_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "date/task/episode_000001")
    make_cleaned_episode(root, "date/task/episode_000002")
    output = tmp_path / "lerobot"
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main([
        "convert", "--input-root", str(root), "--output-root", str(output), "--preflight-only",
        "--include-episode", "date/task/episode_000002",
        "--video-codec", "h264", "--video-crf", "18", "--encoder-preset", "fast",
        "--encoder-threads", "6", "--video-encoding-mode", "parallel",
    ])

    assert code == 0
    summary = json.loads((output / "conversion_summary.json").read_text(encoding="utf-8"))
    assert summary["discovered_episodes"] == 1
    assert summary["selected_episodes"] == 1
    assert summary["included_episodes"] == ["date/task/episode_000002"]
    assert summary["encoder_config"] == {
        "video_codec": "h264", "video_crf": 18, "encoder_preset": "fast",
        "encoder_threads": 6, "video_encoding_mode": "parallel", "video_workers": 3,
    }
    report = json.loads((output / "date/task/conversion_report.json").read_text(encoding="utf-8"))
    assert [item["episode_id"] for item in report["accepted"]] == [2]


@pytest.mark.parametrize(
    "selectors",
    [
        ["date/task/episode_000001", "date/task/episode_000001"],
        ["../episode_000001"],
        ["date/task/episode_999999"],
    ],
)
def test_include_episode_rejects_duplicate_unsafe_and_missing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selectors: list[str],
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "date/task/episode_000001")
    monkeypatch.setattr(
        cli, "discover_episodes", lambda _root: pytest.fail("include path performed full discovery"),
    )

    code = cli.main([
        "convert", "--input-root", str(root), "--output-root", str(tmp_path / "output"),
        "--preflight-only", *[item for value in selectors for item in ("--include-episode", value)],
    ])

    assert code == 2


def test_merged_preflight_defaults_below_input_and_accepts_tasks_with_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(
        root, "date/task_a/episode_000001", task_title="pick_the_tray",
        description="拿起托盘",
    )
    make_cleaned_episode(
        root, "date/task_b/episode_000001", task_title="close_fridge",
        description="关闭冰箱",
    )
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main(["convert-merged", "--input-root", str(root), "--preflight-only"])

    output = root / "lerobot_data"
    assert code == 0
    summary = json.loads((output / "conversion_summary.json").read_text(encoding="utf-8"))
    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert summary["command"] == "convert-merged"
    assert summary["output_root"] == str(output)
    assert summary["counts"] == {"tasks": 2, "accepted": 2, "skipped": 0, "failed": 0}
    assert [item["episode"] for item in report["accepted"]] == [
        "date/task_a/episode_000001", "date/task_b/episode_000001",
    ]
    assert report["task_titles"] == ["close_fridge", "pick_the_tray"]
    assert report["stored_tasks"] == ["close fridge", "pick the tray"]
    assert {item["stored_task"] for item in report["accepted"]} == {"close fridge", "pick the tray"}
    assert {item["episode_id"] for item in report["accepted"]} == {1}


def test_merged_preflight_chooses_largest_training_schema_globally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "group/task_a/episode_000001", task_title="task_a")
    make_cleaned_episode(root, "group/task_b/episode_000002", task_title="task_b")
    make_cleaned_episode(root, "group/schema_outlier/episode_000003", task_title="task_c")

    def probe(path: Path) -> dict:
        result = fake_probe(path)
        if "schema_outlier" in path.parts:
            result["width"] = 80
        return result

    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: probe(path))
    output = tmp_path / "merged"

    code = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
        "--preflight-only",
    ])

    assert code == 1
    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert [item["episode"] for item in report["accepted"]] == [
        "group/task_a/episode_000001", "group/task_b/episode_000002",
    ]
    assert report["skipped"][0]["episode"] == "group/schema_outlier/episode_000003"
    assert report["skipped"][0]["reasons"] == ["schema_outlier"]
    assert report["schema"]["videos"]["rgbd_head_color"]["width"] == 64


def test_merged_preflight_locks_schema_to_existing_owner_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "group/current_a/episode_000001", task_title="task_a")
    make_cleaned_episode(root, "group/current_b/episode_000002", task_title="task_b")
    make_cleaned_episode(root, "group/stored_schema/episode_000003", task_title="task_c")

    def probe(path: Path) -> dict:
        result = fake_probe(path)
        if "stored_schema" in path.parts:
            result["width"] = 80
        return result

    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: probe(path))
    output = tmp_path / "merged"
    owner = output / "whole_body_joint"
    owner.mkdir(parents=True)
    videos = {
        name: {"width": 80, "height": 48, "is_depth": False}
        for name in ("rgbd_head_color", "hand_left", "hand_right")
    }
    (owner / "conversion_state.json").write_text(json.dumps({
        "version": 3,
        "action_mode": "whole_body_joint",
        "dataset_layout": "merged",
        "conversion_config": {"action_dim": 23},
        "schema": {"fps": 30, "videos": videos},
    }), encoding="utf-8")

    code = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
        "--preflight-only", "--action-mode", "whole_body_joint",
    ])

    assert code == 1
    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert [item["episode"] for item in report["accepted"]] == [
        "group/stored_schema/episode_000003",
    ]
    assert {item["episode"] for item in report["skipped"]} == {
        "group/current_a/episode_000001", "group/current_b/episode_000002",
    }
    assert all(item["reasons"] == ["schema_outlier"] for item in report["skipped"])
    assert report["schema"]["videos"]["rgbd_head_color"]["width"] == 80


def test_merged_conversion_calls_engine_with_per_episode_tasks_and_reports_path_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(
        root, "day/task_a/episode_000001", task_title="pick_the_egg_tart",
    )
    make_cleaned_episode(
        root, "day/task_b/episode_000001", task_title="close_fridge",
    )
    output = tmp_path / "merged"
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))
    calls: dict[str, object] = {"loaded": []}

    class Config:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    class Result:
        action_mode = "both"
        failed = ()

        def __init__(self, ledger: dict) -> None:
            self.ledger = ledger

    class Engine:
        ConverterConfig = Config

        @staticmethod
        def load_source_episode(path: Path, task: str) -> dict:
            calls["loaded"].append((path, task))
            return {"path": path, "task": task}

        @staticmethod
        def convert_merged_bundle(
            sources: list[dict], bundle_root: Path, namespace: str, *,
            source_root: Path, config: Config, action_mode: str,
        ) -> Result:
            calls["convert"] = {
                "sources": sources,
                "bundle_root": bundle_root,
                "namespace": namespace,
                "source_root": source_root,
                "config": config.values,
                "action_mode": action_mode,
            }
            paths = [source["path"].relative_to(source_root).as_posix() for source in sources]
            outcomes = ("existing", "appended")
            return Result({
                "episodes": [
                    {
                        "source_relative_path": relative,
                        "modes": {
                            "whole_body_joint": {"status": "committed", "outcome": outcome},
                            "body_joint_eef": {"status": "committed", "outcome": outcome},
                        },
                    }
                    for relative, outcome in zip(paths, outcomes)
                ],
            })

    monkeypatch.setattr(cli, "_engine_import", lambda: Engine)
    code = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
        "--namespace", "merged_test", "--video-codec", "h264",
        "--encoder-threads", "4", "--video-encoding-mode", "parallel",
    ])

    assert code == 0
    assert [(path.relative_to(root).as_posix(), task) for path, task in calls["loaded"]] == [
        ("day/task_a/episode_000001", "pick the egg tart"),
        ("day/task_b/episode_000001", "close fridge"),
    ]
    conversion = calls["convert"]
    assert conversion["bundle_root"] == output
    assert conversion["source_root"] == root
    assert conversion["namespace"] == "merged_test"
    assert conversion["action_mode"] == "both"
    assert conversion["config"]["video_codec"] == "h264"
    assert conversion["config"]["encoder_threads"] == 4
    assert conversion["config"]["video_encoding_mode"] == "parallel"
    assert conversion["config"]["video_workers"] == 3

    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    by_path = {item["episode"]: item for item in report["accepted"]}
    assert all(item["source_relative_path"] == item["episode"] for item in report["accepted"])
    assert by_path["day/task_a/episode_000001"]["conversion"]["modes"]["whole_body_joint"]["outcome"] == "existing"
    assert by_path["day/task_b/episode_000001"]["conversion"]["modes"]["body_joint_eef"]["outcome"] == "appended"


def test_merged_conversion_skips_one_source_load_failure_and_converts_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "day/task/episode_000001")
    make_cleaned_episode(root, "day/task/episode_000002")
    output = tmp_path / "merged"
    monkeypatch.setattr(
        "lightworkbench.validation.probe_video",
        lambda path, decoded=True: fake_probe(path),
    )
    converted_paths: list[str] = []

    class Config:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    class Result:
        action_mode = "both"
        failed = ()

        def __init__(self, ledger: dict) -> None:
            self.ledger = ledger

    class Engine:
        ConverterConfig = Config

        @staticmethod
        def load_source_episode(path: Path, task: str) -> dict:
            if path.name == "episode_000001":
                raise ValueError("synthetic load failure")
            return {"path": path, "task": task}

        @staticmethod
        def convert_merged_bundle(
            sources: list[dict], bundle_root: Path, namespace: str, *,
            source_root: Path, config: Config, action_mode: str,
        ) -> Result:
            del bundle_root, namespace, config, action_mode
            converted_paths.extend(
                source["path"].relative_to(source_root).as_posix() for source in sources
            )
            return Result({
                "episodes": [
                    {
                        "source_relative_path": relative,
                        "modes": {
                            "whole_body_joint": {"status": "committed"},
                            "body_joint_eef": {"status": "committed"},
                        },
                    }
                    for relative in converted_paths
                ],
            })

    monkeypatch.setattr(cli, "_engine_import", lambda: Engine)

    code = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
    ])

    assert code == 1
    assert converted_paths == ["day/task/episode_000002"]
    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert [item["episode"] for item in report["accepted"]] == [
        "day/task/episode_000002",
    ]
    assert [(item["episode"], item["reasons"]) for item in report["skipped"]] == [
        ("day/task/episode_000001", ["source_load_failed:synthetic load failure"]),
    ]
    assert report["failed"] == []
    assert report["counts"] == {"tasks": 1, "accepted": 1, "skipped": 1, "failed": 0}


def test_merged_include_episode_uses_the_shared_exact_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "date/task_a/episode_000001", task_title="task_a")
    make_cleaned_episode(root, "date/task_b/episode_000002", task_title="task_b")
    output = tmp_path / "merged"
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
        "--preflight-only", "--include-episode", "date/task_b/episode_000002",
    ])

    assert code == 0
    summary = json.loads((output / "conversion_summary.json").read_text(encoding="utf-8"))
    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert summary["discovered_episodes"] == 1
    assert summary["selected_episodes"] == 1
    assert summary["included_episodes"] == ["date/task_b/episode_000002"]
    assert [item["episode"] for item in report["accepted"]] == ["date/task_b/episode_000002"]


def test_merged_include_resolves_exact_path_without_scanning_unrelated_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "selected/task/episode_000001", task_title="selected")
    make_cleaned_episode(root, "unrelated/task/episode_000002", task_title="unrelated")
    output = tmp_path / "merged"
    original_rglob = Path.rglob

    def guarded_rglob(path: Path, pattern: str):
        if path == root:
            pytest.fail("include path scanned the input root")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", guarded_rglob)
    monkeypatch.setattr(
        cli, "discover_episodes", lambda _root: pytest.fail("include path performed full discovery"),
    )
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
        "--preflight-only", "--include-episode", "selected/task/episode_000001",
    ])

    assert code == 0
    summary = json.loads((output / "conversion_summary.json").read_text(encoding="utf-8"))
    report = json.loads((output / "conversion_report.json").read_text(encoding="utf-8"))
    assert summary["discovered_episodes"] == summary["selected_episodes"] == 1
    assert [item["episode"] for item in report["accepted"]] == ["selected/task/episode_000001"]


def test_merged_include_preserves_output_directory_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    output = root / "lerobot_data"
    make_cleaned_episode(output, "task/episode_000001")
    monkeypatch.setattr(
        cli, "discover_episodes", lambda _root: pytest.fail("include path performed full discovery"),
    )

    excluded = cli.main([
        "convert-merged", "--input-root", str(root), "--output-root", str(output),
        "--preflight-only", "--include-episode", "lerobot_data/task/episode_000001",
    ])
    nested = cli.main([
        "convert-merged", "--input-root", str(root),
        "--output-root", str(output / "task/episode_000001/result"),
        "--preflight-only", "--include-episode", "lerobot_data/task/episode_000001",
    ])

    assert excluded == 2
    assert nested == 2


def test_preflight_rejects_duplicate_episode_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "task/episode_1")
    make_cleaned_episode(root, "task/episode_000001")
    output = tmp_path / "lerobot"
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main(["convert", "--input-root", str(root), "--output-root", str(output), "--preflight-only"])
    report = json.loads((output / "task/conversion_report.json").read_text(encoding="utf-8"))

    assert code == 1
    assert not report["accepted"]
    assert len(report["skipped"]) == 2
    assert {tuple(item["reasons"]) for item in report["skipped"]} == {
        ("duplicate_episode_id:1",)
    }


def test_preflight_rejects_legacy_action_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "task/episode_000001")
    output = tmp_path / "lerobot"
    task_output = output / "task"
    task_output.mkdir(parents=True)
    (task_output / "conversion_state.json").write_text(
        json.dumps({"version": 1, "conversion_config": {"action_dim": 14}, "schema": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main(["convert", "--input-root", str(root), "--output-root", str(output), "--preflight-only"])
    report = json.loads((task_output / "conversion_report.json").read_text(encoding="utf-8"))

    assert code == 1
    assert report["skipped"][0]["reasons"] == [
        "incompatible_v2_20d_action_schema_requires_new_output_root"
    ]


def test_training_task_is_fixed_to_normalized_english(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(
        root, "task/episode_000001",
        task_title="pick_the_egg_tart_from_tray", description="从托盘拿蛋挞",
    )
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    output = tmp_path / "lerobot"
    code = cli.main([
        "convert", "--input-root", str(root), "--output-root", str(output), "--preflight-only",
    ])
    report = json.loads((output / "task/conversion_report.json").read_text(encoding="utf-8"))

    assert code == 0
    assert report["stored_task"] == "pick the egg tart from tray"
    assert report["source_description"] == "从托盘拿蛋挞"
    assert report["task_language"] == "english"


def test_body_only_preflight_requires_existing_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "task/episode_000001")
    output = tmp_path / "lerobot"
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main([
        "convert", "--input-root", str(root), "--output-root", str(output),
        "--preflight-only", "--action-mode", "body_joint_eef",
    ])
    report = json.loads((output / "task/conversion_report.json").read_text(encoding="utf-8"))

    assert code == 1
    assert report["skipped"][0]["reasons"] == ["body_joint_eef_requires_video_owner_use_both"]


def test_missing_optional_dependency_is_reported_and_other_tasks_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "group/task_a/episode_000001", task_title="task_a")
    make_cleaned_episode(root, "group/task_b/episode_000002", task_title="task_b")
    output = tmp_path / "lerobot"
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))
    monkeypatch.setattr(cli, "_engine_import", lambda: (_ for _ in ()).throw(RuntimeError("optional dependencies missing")))

    code = cli.main(["convert", "--input-root", str(root), "--output-root", str(output)])
    assert code == 1
    summary = json.loads((output / "conversion_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["failed"] == 2
    assert summary["counts"]["tasks"] == 2
    for task in ("task_a", "task_b"):
        report = json.loads((output / f"group/{task}/conversion_report.json").read_text(encoding="utf-8"))
        assert report["failed"][0]["reasons"] == ["conversion_failed:optional dependencies missing"]


def test_successful_conversion_result_is_json_safe() -> None:
    from lightworkbench.lerobot_converter import ConversionResult

    result = ConversionResult(
        Path("dataset"), True, (), (0,), (), (), {"stored_task": "task"}
    )
    payload = cli._conversion_payload(result)

    assert payload["output_root"] == "dataset"
    json.dumps(payload)


def test_existing_core_schema_is_normalized() -> None:
    state = {
        "schema": {
            "fps": 30,
            "videos": {
                "rgbd_head_color": {
                    "key": "observation.images.rgbd_head_color",
                    "width": 64,
                    "height": 48,
                    "channels": 3,
                    "is_depth": False,
                }
            },
        }
    }
    expected = cli.schema_key({
        "fps": 30,
        "videos": {"rgbd_head_color": {"width": 64, "height": 48, "is_depth": False}},
    })
    assert cli._stored_schema_key(state) == expected
