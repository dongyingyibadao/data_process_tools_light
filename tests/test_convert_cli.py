from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightworkbench import cli
from lightworkbench.validation import (
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
    report_path = output / "289/date/close_fridge/conversion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task_title"] == "close_fridge"
    assert report["source_description"] == "关闭冰箱"
    assert report["stored_task"] == "close fridge"
    assert [Path(item["episode"]).name for item in report["accepted"]] == [
        "episode_000002", "episode_000010",
    ]
    assert not list(output.rglob("*.tmp"))


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
    assert report["skipped"][0]["reasons"] == ["incompatible_action_schema_requires_new_output_root"]


def test_preflight_rejects_stored_task_language_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cleaned"
    make_cleaned_episode(root, "task/episode_000001")
    task_output = tmp_path / "lerobot/task"
    task_output.mkdir(parents=True)
    (task_output / "conversion_state.json").write_text(
        json.dumps({"version": 2, "conversion_config": {"action_dim": 20}, "stored_task": "close fridge"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("lightworkbench.validation.probe_video", lambda path, decoded=True: fake_probe(path))

    code = cli.main(["convert", "--input-root", str(root), "--output-root", str(tmp_path / "lerobot"), "--preflight-only", "--task-language", "source"])
    report = json.loads((task_output / "conversion_report.json").read_text(encoding="utf-8"))

    assert code == 1
    assert report["skipped"][0]["reasons"] == ["stored_task_language_or_text_conflict"]


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
