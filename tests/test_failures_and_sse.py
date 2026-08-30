from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch
import pytest

from fastapi.testclient import TestClient

from lightworkbench.app import app, operation_manager
from lightworkbench.config import RESOURCES
from lightworkbench.core import BrowseService, ConflictError, EpisodeService, WorkbenchError
from lightworkbench.operations import OperationManager, QueueFullError
from test_workbench import detail, make_episode, submit, wait_operation


client = TestClient(app)


def test_129_episode_browse_is_bounded_and_fast(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    task = root / "289/2026-08-30/task"
    for index in range(129):
        (task / f"episode_{index:06d}" / "videos").mkdir(parents=True)
    started = time.perf_counter()
    result = BrowseService().browse(str(root), "289/2026-08-30")
    assert result["view"] == "folders"
    assert result["folders"][0]["episodeCount"] == 129
    result = BrowseService().browse(str(root), "289/2026-08-30/task")
    elapsed = time.perf_counter() - started
    assert result["view"] == "episodes"
    assert len(result["episodes"]) == 129
    assert elapsed < 0.5


def test_overwrite_publish_failure_restores_old_output(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "cleaned"
    relative = "289/2026-08-30/task/episode_000001"
    make_episode(root, relative)
    first = submit(root, output, relative, "no_trim")
    first_id = first.json()["operationId"]
    assert wait_operation(first_id)["status"] == "completed"
    destination = output / relative
    original_info = (destination / "CUT_INFO.json").read_bytes()

    real_replace = os.replace

    def failing_publish(source, target):
        source_path, target_path = Path(source), Path(target)
        if ".lightworkbench-staging" in source_path.parts and target_path == destination:
            raise OSError("injected publish failure")
        return real_replace(source, target)

    with patch("lightworkbench.operations.os.replace", side_effect=failing_publish):
        second = submit(root, output, relative, "trim", [[0, 2]], overwrite=True)
        assert second.status_code == 202
        failed = wait_operation(second.json()["operationId"])
    assert failed["status"] == "failed"
    assert (destination / "CUT_INFO.json").read_bytes() == original_info
    assert json.loads(original_info)["operationId"] == first_id


def test_csv_retry_and_terminal_sse(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "cleaned"
    relative = "289/2026-08-30/task/episode_000001"
    make_episode(root, relative)
    with patch.object(operation_manager, "_append_csv", side_effect=OSError("read only csv")):
        created = submit(root, output, relative, "no_trim")
        operation_id = created.json()["operationId"]
        finished = wait_operation(operation_id)
    assert finished["status"] == "completed_csv_failed"
    assert (output / relative / "CUT_INFO.json").is_file()

    with client.stream("GET", f"/api/operations/{operation_id}/events") as response:
        event_text = "".join(response.iter_text())
    assert response.status_code == 200
    assert "event: progress" in event_text
    assert "completed_csv_failed" in event_text

    retried = client.post(f"/api/operations/{operation_id}/retry-csv")
    assert retried.status_code == 200
    assert retried.json()["status"] == "completed"
    assert (output / "CUT_HISTORY.csv").is_file()


def test_operation_settings_api() -> None:
    settings = client.get("/api/operations/settings")
    assert settings.status_code == 200
    assert settings.json()["recommendedConcurrency"] == 2
    changed = client.patch("/api/operations/settings", json={"concurrency": 4})
    assert changed.status_code == 200
    assert changed.json()["concurrency"] == 4
    assert client.patch("/api/operations/settings", json={"concurrency": 5}).status_code == 422
    reset = client.patch("/api/operations/settings", json={"concurrency": 2})
    assert reset.json()["concurrency"] == 2

def test_queue_capacity_duplicate_target_settings_and_slot_budget(tmp_path: Path) -> None:
    manager = OperationManager(EpisodeService())
    manager.set_concurrency(1)
    root = tmp_path / "raw"
    relative = "task/episode_000001"
    (root / relative / "videos").mkdir(parents=True)
    release = threading.Event()
    started = threading.Event()
    allocations: list[int] = []

    def fake_run(state, _request):
        allocations.append(state.ffmpeg_slots)
        started.set()
        assert release.wait(10)
        manager._update(state, 1.0, "已完成", status="completed", result={"outputPath": state.target_key})

    manager._run = fake_run
    body = {
        "mode": "no_trim",
        "sourceRoot": str(root),
        "episode": relative,
        "outputRoot": str(tmp_path / "out-0"),
        "operator": "tester",
        "sourceToken": "token",
        "ranges": [],
        "overwrite": False,
    }

    first = manager.create(body)
    assert started.wait(2)
    assert first.status == "running"
    with pytest.raises(ConflictError):
        manager.create(body)

    last = None
    for index in range(1, manager.MAX_PENDING + 1):
        last = manager.create({**body, "outputRoot": str(tmp_path / f"out-{index}")})
    assert last is not None and last.queue_position == manager.MAX_PENDING
    listing = manager.list_operations()
    assert len(listing["running"]) == 1
    assert len(listing["queued"]) == manager.MAX_PENDING
    with pytest.raises(QueueFullError):
        manager.create({**body, "outputRoot": str(tmp_path / "overflow")})

    assert manager.settings()["recommendedConcurrency"] == 2
    for invalid in (0, 5):
        with pytest.raises(WorkbenchError):
            manager.set_concurrency(invalid)
    manager.set_concurrency(4)
    assert manager.settings()["concurrency"] == 4
    assert len(manager.list_operations()["running"]) == 1

    release.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = manager.list_operations()
        if not current["queued"] and not current["running"]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("queued operations did not drain")
    assert allocations
    assert all(1 <= value <= int(RESOURCES["ffmpegSlots"]) for value in allocations)
