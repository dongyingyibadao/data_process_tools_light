from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightworkbench import shard_merge


def _make_shard(
    tmp_path: Path,
    name: str,
    identities: list[str],
    *,
    source_root: str = "/raw/source",
) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / name / "whole_body_joint"
    root.mkdir(parents=True)
    episodes = []
    rows = []
    for index, identity in enumerate(identities):
        signature = f"signature-{name}-{index}"
        task = f"task {identity}"
        episodes.append(
            {
                "source_episode_name": Path(identity).name,
                "source_episode_id": index,
                "source_signature": {"algorithm": "sha256", "digest": signature, "files": 6},
                "lerobot_episode_index": index,
                "source_frames": 3,
                "output_frames": 3,
                "source_relative_path": identity,
                "stored_task": task,
            }
        )
        relative = Path("auxiliary/videos/depth") / "chunk-000" / f"file-{index:03d}.mp4"
        video = root / relative
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{name}-{index}".encode())
        rows.append(
            {
                "episode_index": index,
                "source_episode_id": index,
                "stream": "depth",
                "relative_path": relative.as_posix(),
                "source_signature": signature,
            }
        )
    state = {
        "version": 3,
        "action_mode": "whole_body_joint",
        "dataset_layout": "merged",
        "dataset_role": "video_owner",
        "source_task": source_root,
        "source_root": source_root,
        "repo_id": f"local/{name}",
        "conversion_config": {"schema_version": 6},
        "schema": {"fps": 30, "videos": {}},
        "shared_video_owner": None,
        "episodes": episodes,
        "stored_tasks": sorted(entry["stored_task"] for entry in episodes),
        "created_at": "old",
        "updated_at": "old",
    }
    (root / "conversion_state.json").write_text(json.dumps(state), encoding="utf-8")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "auxiliary/index.parquet")
    return root


def _fake_runtime(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> list[dict]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    calls: list[dict] = []

    def aggregate(**kwargs):
        calls.append(kwargs)
        if fail:
            raise RuntimeError("official merge failed")
        root = Path(kwargs["aggr_root"])
        (root / "meta").mkdir(parents=True)
        count = sum(
            len(json.loads((Path(item) / "conversion_state.json").read_text())["episodes"])
            for item in kwargs["roots"]
        )
        (root / "meta/info.json").write_text(
            json.dumps({"total_episodes": count, "chunks_size": 1000}), encoding="utf-8"
        )

    monkeypatch.setattr(
        shard_merge,
        "_runtime",
        lambda: {"pa": pa, "pq": pq, "aggregate_datasets": aggregate},
    )
    return calls


def test_merge_preserves_episode_identity_and_reindexes_auxiliary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    first = _make_shard(tmp_path, "shard-a", ["day/task/episode_000001"])
    second = _make_shard(
        tmp_path, "shard-b", ["day/task/episode_000002", "day/other/episode_000001"]
    )
    calls = _fake_runtime(monkeypatch)
    output = tmp_path / "final" / "whole_body_joint"

    result = shard_merge.merge_whole_body_joint_shards(
        [first.parent, second], output
    )

    assert result == output.resolve()
    assert len(calls) == 1
    assert calls[0]["roots"] == [first.resolve(), second.resolve()]
    assert calls[0]["concatenate_videos"] is False
    assert calls[0]["concatenate_data"] is False
    state = json.loads((output / "conversion_state.json").read_text())
    assert [entry["lerobot_episode_index"] for entry in state["episodes"]] == [0, 1, 2]
    assert [entry["source_relative_path"] for entry in state["episodes"]] == [
        "day/task/episode_000001",
        "day/task/episode_000002",
        "day/other/episode_000001",
    ]
    assert [entry["source_signature"]["digest"] for entry in state["episodes"]] == [
        "signature-shard-a-0", "signature-shard-b-0", "signature-shard-b-1"
    ]
    assert state["stored_tasks"] == sorted(entry["stored_task"] for entry in state["episodes"])

    rows = pq.read_table(output / "auxiliary/index.parquet").to_pylist()
    assert [row["episode_index"] for row in rows] == [0, 1, 2]
    assert [row["source_signature"] for row in rows] == [
        "signature-shard-a-0", "signature-shard-b-0", "signature-shard-b-1"
    ]
    for index, row in enumerate(rows):
        assert row["relative_path"] == f"auxiliary/videos/depth/chunk-000/file-{index:03d}.mp4"
        assert (output / row["relative_path"]).is_file()


def test_merge_rejects_incompatible_state_before_official_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_shard(tmp_path, "shard-a", ["a/episode_000001"])
    second = _make_shard(
        tmp_path, "shard-b", ["b/episode_000001"], source_root="/different/source"
    )
    calls = _fake_runtime(monkeypatch)

    with pytest.raises(shard_merge.StateConflictError, match="source_root differs"):
        shard_merge.merge_whole_body_joint_shards([first, second], tmp_path / "final")

    assert calls == []
    assert not (tmp_path / "final").exists()


def test_failed_merge_does_not_replace_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = _make_shard(tmp_path, "shard-a", ["a/episode_000001"])
    output = tmp_path / "final"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    _fake_runtime(monkeypatch, fail=True)

    with pytest.raises(RuntimeError, match="official merge failed"):
        shard_merge.merge_whole_body_joint_shards([shard], output, overwrite=True)

    assert marker.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".final.staging-*"))
