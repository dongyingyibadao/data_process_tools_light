from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from lightworkbench.app import app


client = TestClient(app)


def make_lerobot_dataset(root: Path) -> Path:
    dataset = root / "body_joint_eef"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    video_keys = (
        "observation.images.rgbd_head_color",
        "observation.images.hand_left",
        "observation.images.hand_right",
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test_robot",
        "total_episodes": 3,
        "total_frames": 36,
        "total_tasks": 2,
        "chunks_size": 1000,
        "fps": 10,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            key: {"dtype": "video", "shape": [32, 48, 3]} for key in video_keys
        } | {"observation.state": {"dtype": "float32", "shape": [2]}},
    }
    (dataset / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pyarrow.table({"task_index": [0, 1], "task": ["Pick cup", "Place cup"]}),
        dataset / "meta/tasks.parquet",
    )
    episode_columns: dict[str, list] = {
        "episode_index": [0, 1, 2],
        "tasks": [["Pick cup"], ["Place cup"], ["Pick cup"]],
        "length": [10, 12, 14],
        "data/chunk_index": [0, 0, 0],
        "data/file_index": [0, 0, 0],
        "dataset_from_index": [0, 10, 22],
        "dataset_to_index": [10, 22, 36],
    }
    for key in video_keys:
        episode_columns[f"videos/{key}/chunk_index"] = [0, 0, 0]
        episode_columns[f"videos/{key}/file_index"] = [0, 1, 2]
        episode_columns[f"videos/{key}/from_timestamp"] = [0.0, 0.0, 0.0]
        episode_columns[f"videos/{key}/to_timestamp"] = [1.0, 1.2, 1.4]
        video_dir = dataset / "videos" / key / "chunk-000"
        video_dir.mkdir(parents=True)
        for index in range(3):
            (video_dir / f"file-{index:03d}.mp4").write_bytes(b"fake-mp4-content")
    pq.write_table(
        pyarrow.table(episode_columns), dataset / "meta/episodes/chunk-000/file-000.parquet"
    )
    pq.write_table(
        pyarrow.table({
            "episode_index": [0] * 10 + [1] * 12 + [2] * 14,
            "frame_index": list(range(10)) + list(range(12)) + list(range(14)),
            "task_index": [0] * 10 + [1] * 12 + [0] * 14,
        }),
        dataset / "data/chunk-000/file-000.parquet",
    )
    return dataset


def test_lerobot_summary_task_grouping_and_pagination(tmp_path: Path) -> None:
    dataset = make_lerobot_dataset(tmp_path)
    summary = client.get("/api/lerobot/summary", params={"root": str(tmp_path)})
    assert summary.status_code == 200, summary.text
    payload = summary.json()
    assert payload["dataset"] == "body_joint_eef"
    assert payload["totalEpisodes"] == 3
    assert payload["videoKeys"] == [
        "observation.images.rgbd_head_color",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]
    assert [(item["task"], item["episodeCount"]) for item in payload["tasks"]] == [
        ("Pick cup", 2), ("Place cup", 1)
    ]

    filtered = client.get("/api/lerobot/episodes", params={
        "root": str(dataset), "task_index": 0, "page": 1, "page_size": 1,
    })
    assert filtered.status_code == 200, filtered.text
    listing = filtered.json()
    assert listing["total"] == 2
    assert listing["totalPages"] == 2
    assert listing["items"][0]["episodeIndex"] == 0

    searched = client.get("/api/lerobot/episodes", params={"root": str(dataset), "q": "2"})
    assert [item["episodeIndex"] for item in searched.json()["items"]] == [2]


def test_lerobot_detail_media_and_page(tmp_path: Path) -> None:
    dataset = make_lerobot_dataset(tmp_path)
    detail = client.get("/api/lerobot/episodes/1", params={"root": str(dataset)})
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert payload["frameCount"] == 12
    assert payload["data"]["rows"] == 12
    assert payload["data"]["fileRows"] == 36
    assert payload["data"]["fromIndex"] == 10
    assert payload["data"]["toIndex"] == 22
    assert len(payload["streams"]) == 3
    assert all(stream["exists"] and stream["mediaUrl"] for stream in payload["streams"])

    media = client.get(payload["streams"][0]["mediaUrl"], headers={"Range": "bytes=0-3"})
    assert media.status_code == 206
    assert media.content == b"fake"
    assert client.get("/lerobot").status_code == 200
    assert client.get(
        "/api/lerobot/episodes/1/media/not-a-stream", params={"root": str(dataset)}
    ).status_code in {400, 404}
