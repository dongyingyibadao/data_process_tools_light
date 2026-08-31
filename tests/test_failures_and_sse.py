from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch
import pytest

from fastapi.testclient import TestClient

from lightworkbench.app import app, operation_manager, operation_snapshot_events
from lightworkbench.config import RESOURCES
from lightworkbench.core import BrowseService, ConflictError, EpisodeService, WorkbenchError
from lightworkbench.operations import OperationManager, QueueFullError
from test_workbench import detail, make_episode, submit, wait_operation


client = TestClient(app)


class DisconnectRequest:
    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def parse_snapshot(event: str) -> dict:
    assert event.startswith("event: snapshot\n")
    return json.loads(event.split("data: ", 1)[1])


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


def test_shared_operation_sse_snapshots_and_disconnect(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = OperationManager(EpisodeService())
        manager.set_concurrency(1)
        root = tmp_path / "raw"
        relative = "task/episode_000001"
        (root / relative / "videos").mkdir(parents=True)
        first_release = threading.Event()
        second_release = threading.Event()
        third_release = threading.Event()
        second_started = threading.Event()
        third_started = threading.Event()

        def fake_run(state, _request):
            if state.episode.endswith("000001"):
                assert first_release.wait(5)
                manager._update(state, 1.0, "已完成", status="completed", result={"outputPath": state.target_key})
            elif state.episode.endswith("000002"):
                second_started.set()
                assert second_release.wait(5)
                raise WorkbenchError("injected failure")
            else:
                third_started.set()
                assert third_release.wait(5)
                manager._update(state, 1.0, "已完成", status="completed", result={"outputPath": state.target_key})

        manager._run = fake_run
        body = {
            "mode": "no_trim", "sourceRoot": str(root), "episode": relative,
            "outputRoot": str(tmp_path / "out-1"), "operator": "tester",
            "sourceToken": "token", "ranges": [], "overwrite": False,
        }
        first = manager.create(body)
        deadline = time.monotonic() + 2
        while first.status != "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert first.status == "running"
        for number in (2, 3):
            episode = f"task/episode_{number:06d}"
            (root / episode / "videos").mkdir(parents=True)
            manager.create({**body, "episode": episode, "outputRoot": str(tmp_path / f"out-{number}")})

        request = DisconnectRequest()
        stream = operation_snapshot_events(manager, request)
        initial = parse_snapshot(await asyncio.wait_for(anext(stream), 1))
        assert initial["running"][0]["id"] == first.id
        assert [item["queuePosition"] for item in initial["queued"]] == [1, 2]

        manager._update(first, 0.5, "halfway", status="running")
        progress = parse_snapshot(await asyncio.wait_for(anext(stream), 1))
        assert progress["running"][0]["progress"] == 0.5
        assert progress["running"][0]["message"] == "halfway"

        first_release.set()
        assert await asyncio.to_thread(second_started.wait, 2)
        await asyncio.sleep(0.25)
        shifted = parse_snapshot(await asyncio.wait_for(anext(stream), 1))
        assert shifted["running"][0]["episode"].endswith("000002")
        assert shifted["queued"][0]["queuePosition"] == 1
        assert any(item["status"] == "completed" for item in shifted["completed"])

        second_release.set()
        assert await asyncio.to_thread(third_started.wait, 2)
        await asyncio.sleep(0.25)
        failed = parse_snapshot(await asyncio.wait_for(anext(stream), 1))
        assert failed["running"][0]["episode"].endswith("000003")
        assert any(item["status"] == "failed" for item in failed["completed"])

        third_release.set()
        await asyncio.sleep(0.25)
        terminal = parse_snapshot(await asyncio.wait_for(anext(stream), 1))
        assert not terminal["queued"] and not terminal["running"]
        assert {item["status"] for item in terminal["completed"]} >= {"completed", "failed"}

        request.disconnected = True
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), 1)

    asyncio.run(scenario())
