from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lightworkbench.app import app, browse_service
from lightworkbench import config
from lightworkbench.config import RESOURCES
from lightworkbench.core import BrowseService, WorkbenchError, probe_video
from lightworkbench.operations import normalize_ranges
from lightworkbench.validation import validate_episode


client = TestClient(app)


def make_episode(root: Path, relative: str = "289/2026-08-30/task/episode_000001",
                 frames: int = 8, fps: int = 8) -> tuple[Path, dict[str, str]]:
    episode = root / relative
    episode.mkdir(parents=True)
    streams = {
        name: f"videos/{name}/episode_000001.mp4"
        for name in ("hand_left", "hand_right", "head_right", "rgbd_head_color")
    }
    for index, path_value in enumerate(streams.values()):
        target = episode / path_value
        target.parent.mkdir(parents=True)
        subprocess.run([
            shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=48x32:rate={fps}:duration={frames / fps}",
            "-vf", f"hue=h={index * 40}", "-frames:v", str(frames), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(target),
        ], check=True)
    episode_id = int(episode.name.split("_")[-1])
    task_title = "pick_part"
    pose = {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
    rows = [{
        "_type": "session_header", "fps_target": fps, "episode_id": episode_id,
        "task_title": task_title, "task_description": "抓取零件",
    }]
    for index in range(frames):
        rows.append({
            "frame_idx": index,
            "t_wall": 1000 + index / fps,
            "t_ns": 1_000_000_000_000 + int(index / fps * 1_000_000_000),
            "t_monotonic": 10 + index / fps,
            "t_intended": 10 + index / fps,
            "videos": {
                name: {"path": value, "frame_id": index, "is_repeat": False, "frames_dropped": 0}
                for name, value in streams.items()
            },
            "task": {"task_title": task_title, "episode_id": episode_id},
            "joints": {
                "position": [float(index)] * 23,
                "velocity": [0.0] * 23,
                "torque": [0.0] * 23,
            },
            "current_eef_pose": {
                name: dict(pose) for name in ("left_eef_pose", "right_eef_pose", "head_pose")
            },
            "target_eef_pose": {
                name: dict(pose) for name in ("left_eef_pose", "right_eef_pose", "head_pose")
            },
            "current_height_z": {"height_z": 0.0},
            "target_height_z": {"height_z": 0.0},
            "robot_state": {"state": 2},
            "control": {},
            "action": [float(index)],
        })
    rows.append({"_type": "session_footer", "aborted": False})
    (episode / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (episode / "task_meta.json").write_text(
        json.dumps({"task_title": task_title, "description": "抓取零件"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return episode, streams


def detail(root: Path, relative: str) -> dict:
    response = client.get(f"/api/episodes/{relative}", params={"root": str(root)})
    assert response.status_code == 200, response.text
    return response.json()


def wait_operation(operation_id: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/api/operations/{operation_id}").json()
        if payload["status"] in {"completed", "completed_csv_failed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("operation did not finish")


def submit(root: Path, output: Path, relative: str, mode: str, ranges=None, overwrite=False):
    opened = detail(root, relative)
    return client.post("/api/operations", json={
        "mode": mode, "sourceRoot": str(root), "episode": relative,
        "outputRoot": str(output), "operator": "tester", "sourceToken": opened["sourceToken"],
        "ranges": ranges or [], "overwrite": overwrite,
    })


def test_two_level_browse_cache_refresh_natural_sort_and_boundary(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "290").mkdir()
    (root / "289").mkdir()
    service = BrowseService()
    first = service.browse(str(root))
    assert first["view"] == "folders"
    assert [item["name"] for item in first["folders"]] == ["289", "290"]
    (root / "291").mkdir()
    assert [item["name"] for item in service.browse(str(root))["folders"]] == ["289", "290"]
    assert [item["name"] for item in service.browse(str(root), refresh=True)["folders"]] == ["289", "290", "291"]

    task = root / "289/date/task"
    for number in (10, 2, 1):
        (task / f"episode_{number}" / "videos").mkdir(parents=True)
    parent_listing = service.browse(str(root), "289/date")
    assert parent_listing["view"] == "folders"
    assert parent_listing["folders"] == [{"name": "task", "path": "289/date/task", "episodeCount": 3}]
    assert parent_listing["totalEpisodeCount"] == 3
    listing = service.browse(str(root), "289/date/task")
    assert listing["view"] == "episodes"
    assert listing["totalEpisodeCount"] == 3
    assert [item["name"] for item in listing["episodes"]] == ["episode_1", "episode_2", "episode_10"]
    with pytest.raises(WorkbenchError):
        service.browse(str(root), "../outside")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / "escape")
    assert "escape" not in [item["name"] for item in service.browse(str(root), refresh=True)["folders"]]


def test_detail_media_range_source_change_and_corrupt_video(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    relative = "289/2026-08-30/task/episode_000001"
    episode, streams = make_episode(root, relative)
    opened = detail(root, relative)
    assert opened["valid"] is True
    assert opened["frameCount"] == 8
    assert all(item["browserPlayable"] for item in opened["streams"])
    media = client.get(
        f"/api/episodes/{relative}/media/head_right",
        params={"root": str(root)}, headers={"Range": "bytes=0-31"},
    )
    assert media.status_code == 206
    assert len(media.content) == 32
    old_token = opened["sourceToken"]
    (episode / "task_meta.json").write_text('{"description":"变化"}\n', encoding="utf-8")
    assert detail(root, relative)["sourceToken"] != old_token
    (episode / streams["hand_right"]).write_bytes(b"broken")
    broken = detail(root, relative)
    assert broken["valid"] is False
    assert any("hand_right" in issue for issue in broken["issues"])


def test_ranges_trim_copy_overwrite_and_csv(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "cleaned"
    relative = "289/2026-08-30/task/episode_000001"
    make_episode(root, relative)
    assert normalize_ranges([[-2, 2], [1, 4], [6, 20]], 8) == [(0, 4), (6, 8)]

    created = submit(root, output, relative, "trim", [[0, 2], [6, 8]])
    assert created.status_code == 202, created.text
    first_id = created.json()["operationId"]
    finished = wait_operation(first_id)
    assert created.json()["status"] == "queued"
    assert created.json()["queuePosition"] >= 1
    assert finished["status"] == "completed", finished
    destination = output / relative
    manifest_rows = [json.loads(line) for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [row for row in manifest_rows if not isinstance(row.get("_type"), str)]
    assert [row["frame_idx"] for row in records] == list(range(4))
    assert all(row["videos"]["head_right"]["frame_id"] == row["frame_idx"] for row in records)
    for video in (destination / "videos").rglob("*.mp4"):
        assert probe_video(video, decoded=True)["frames"] == 4
    info = json.loads((destination / "CUT_INFO.json").read_text(encoding="utf-8"))
    assert info["removedRanges"] == [[0, 2], [6, 8]]

    conflict = submit(root, output, relative, "no_trim")
    assert conflict.status_code == 409
    overwritten = submit(root, output, relative, "no_trim", overwrite=True)
    assert overwritten.status_code == 202, overwritten.text
    second_id = overwritten.json()["operationId"]
    finished = wait_operation(second_id)
    assert finished["status"] == "completed", finished
    info = json.loads((destination / "CUT_INFO.json").read_text(encoding="utf-8"))
    assert info["revision"] == 2
    assert info["overwrittenOperationId"] == first_id
    assert info["outputFrames"] == 8
    with (output / "CUT_HISTORY.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["operation_id"] for row in rows] == [first_id, second_id]
    assert not (output / ".lightworkbench-backups" / second_id).exists()


def test_no_trim_normalizes_only_manifest_frame_indices_and_passes_preflight(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "cleaned"
    relative = "289/2026-08-30/task/episode_000001"
    episode, streams = make_episode(root, relative)
    manifest = episode / "manifest.jsonl"
    source_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    source_frames = [row for row in source_rows if not isinstance(row.get("_type"), str)]
    for index, row in enumerate(source_frames):
        row["frame_idx"] = 500 + index
        for entry in row["videos"].values():
            entry["frame_id"] = 1000 + index
        row["videos"]["hand_right"]["is_repeat"] = index == 2
        row["videos"]["hand_right"]["frames_dropped"] = 7 if index == 2 else 0
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows), encoding="utf-8"
    )
    source_video_bytes = {
        name: (episode / relative_path).read_bytes() for name, relative_path in streams.items()
    }

    created = submit(root, output, relative, "no_trim")
    assert created.status_code == 202, created.text
    finished = wait_operation(created.json()["operationId"])
    assert finished["status"] == "completed", finished

    destination = output / relative
    output_rows = [
        json.loads(line) for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    output_frames = [row for row in output_rows if not isinstance(row.get("_type"), str)]
    assert [row["frame_idx"] for row in output_frames] == list(range(8))
    assert all(
        entry["frame_id"] == index
        for index, row in enumerate(output_frames)
        for entry in row["videos"].values()
    )
    expected_frames = json.loads(json.dumps(source_frames))
    for index, row in enumerate(expected_frames):
        row["frame_idx"] = index
        for entry in row["videos"].values():
            entry["frame_id"] = index
    assert output_frames == expected_frames
    assert output_frames[2]["videos"]["hand_right"]["is_repeat"] is True
    assert output_frames[2]["videos"]["hand_right"]["frames_dropped"] == 7
    assert {
        name: (destination / relative_path).read_bytes() for name, relative_path in streams.items()
    } == source_video_bytes
    info = json.loads((destination / "CUT_INFO.json").read_text(encoding="utf-8"))
    assert info["manifestNormalizationVersion"] == 1
    assert info["normalizedFields"] == ["frame_idx", "videos.*.frame_id"]
    assert validate_episode(output, destination).valid


def test_converter_preflight_failure_never_publishes_to_cleaned(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "cleaned"
    relative = "289/2026-08-30/task/episode_000001"
    episode, streams = make_episode(root, relative)
    manifest = episode / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        videos = row.get("videos")
        if isinstance(videos, dict):
            videos.pop("hand_left", None)
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (episode / streams["hand_left"]).unlink()

    created = submit(root, output, relative, "no_trim")
    assert created.status_code == 202, created.text
    finished = wait_operation(created.json()["operationId"])

    assert finished["status"] == "failed"
    assert "转换预检未通过" in finished["error"]
    assert "required_video_missing:hand_left" in finished["error"]
    assert not (output / relative).exists()
    assert not list(tmp_path.glob(".cleaned.lightworkbench-staging-*"))


def test_reject_too_short_stale_source_and_overlapping_output(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "cleaned"
    relative = "289/2026-08-30/task/episode_000001"
    episode, _ = make_episode(root, relative)
    opened = detail(root, relative)
    stale_body = {
        "mode": "trim", "sourceRoot": str(root), "episode": relative, "outputRoot": str(output),
        "operator": "tester", "sourceToken": opened["sourceToken"], "ranges": [[0, 7]], "overwrite": False,
    }
    too_short = client.post("/api/operations", json=stale_body)
    assert too_short.status_code == 202
    assert wait_operation(too_short.json()["operationId"])["status"] == "failed"

    (episode / "task_meta.json").write_text('{"description":"new"}\n', encoding="utf-8")
    stale_body["ranges"] = [[0, 2]]
    stale = client.post("/api/operations", json=stale_body)
    assert stale.status_code == 202
    result = wait_operation(stale.json()["operationId"])
    assert result["status"] == "failed"
    assert "源数据已变化" in result["error"]

    current = detail(root, relative)
    overlap = client.post("/api/operations", json={**stale_body, "sourceToken": current["sourceToken"], "outputRoot": str(root / "cleaned")})
    assert overlap.status_code == 400


def test_cpu_budget_uses_all_available_cpus_without_affinity_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    affinity_changes = []
    monkeypatch.setattr(config.os, "sched_getaffinity", lambda _pid: {8, 2, 5, 3})
    monkeypatch.setattr(config.os, "sched_setaffinity", lambda pid, cpus: affinity_changes.append((pid, cpus)))
    monkeypatch.setattr(config, "_quota_cpu_count", lambda: None)

    resources = config.resource_budget()

    assert resources["available"] == 4
    assert resources["budget"] == 4
    assert resources["cpus"] == [2, 3, 5, 8]
    assert affinity_changes == []


@pytest.mark.parametrize(
    ("quota", "expected_budget"),
    [(3.75, 3), (0.25, 1), (20.0, 8)],
)
def test_cpu_budget_respects_cgroup_quota(
    monkeypatch: pytest.MonkeyPatch, quota: float, expected_budget: int,
) -> None:
    quota_reads = []
    monkeypatch.setattr(config.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(config, "_quota_cpu_count", lambda: quota_reads.append(quota) or quota)

    resources = config.resource_budget(apply=False)

    assert resources["quotaAvailable"] == quota
    assert resources["available"] == min(8, quota)
    assert resources["budget"] == expected_budget
    assert len(resources["cpus"]) == expected_budget
    assert quota_reads == [quota]


def test_static_no_polling() -> None:
    assert RESOURCES["budget"] == max(1, int(float(RESOURCES["available"])))
    assert len(RESOURCES["cpus"]) == RESOURCES["budget"]
    assert int(RESOURCES["webSlots"]) <= 2
    script = Path(__file__).parents[1] / "lightworkbench/static/app.js"
    content = script.read_text(encoding="utf-8")
    assert "setInterval" not in content
    assert content.count("new EventSource(") == 1
    assert 'new EventSource("/api/operations/events")' in content
    assert "eventSources" not in content
    assert "new AbortController()" in content
    assert "{signal: controller.signal}" in content
    assert "Math.min(26" in content
    page = client.get("/")
    assert page.status_code == 200
    assert "episodeViewport" in page.text
