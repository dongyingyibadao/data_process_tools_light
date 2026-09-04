from __future__ import annotations

import json
import os
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
    assert state["source_root"] == str(Path("/raw/source").resolve())
    assert state["source_roots"] == [str(Path("/raw/source").resolve())]
    assert {entry["source_root"] for entry in state["episodes"]} == {
        str(Path("/raw/source").resolve())
    }
    assert (output.parent / ".bundle/convert-merged.lock").is_file()

    rows = pq.read_table(output / "auxiliary/index.parquet").to_pylist()
    assert [row["episode_index"] for row in rows] == [0, 1, 2]
    assert [row["source_signature"] for row in rows] == [
        "signature-shard-a-0", "signature-shard-b-0", "signature-shard-b-1"
    ]
    for index, row in enumerate(rows):
        assert row["relative_path"] == f"auxiliary/videos/depth/chunk-000/file-{index:03d}.mp4"
        assert (output / row["relative_path"]).is_file()
    assert os.path.samefile(
        first / "auxiliary/videos/depth/chunk-000/file-000.mp4",
        output / rows[0]["relative_path"],
    )


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


def test_incremental_merge_uses_existing_owner_and_allows_same_path_from_different_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _make_shard(
        tmp_path, "final", ["day/task/episode_000001"], source_root="/raw/source-z",
    )
    new_shard = _make_shard(
        tmp_path, "shard-b", ["day/task/episode_000001"], source_root="/raw/source-a",
    )
    calls = _fake_runtime(monkeypatch)

    result = shard_merge.merge_whole_body_joint_shards(
        [new_shard], output, incremental=True,
    )

    assert result == output.resolve()
    assert calls[0]["roots"] == [output.resolve(), new_shard.resolve()]
    state = json.loads((output / "conversion_state.json").read_text(encoding="utf-8"))
    assert [entry["source_relative_path"] for entry in state["episodes"]] == [
        "day/task/episode_000001", "day/task/episode_000001",
    ]
    assert [entry["source_root"] for entry in state["episodes"]] == [
        str(Path("/raw/source-z").resolve()),
        str(Path("/raw/source-a").resolve()),
    ]
    assert state["source_root"] == str(Path("/raw/source-a").resolve())
    assert state["source_roots"] == [
        str(Path("/raw/source-a").resolve()),
        str(Path("/raw/source-z").resolve()),
    ]
    assert state["source_task"] == str(Path("/raw/source-a").resolve())
    assert state["created_at"] == "old"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conversion_config", {"schema_version": 999}, "conversion_config differs"),
        ("schema", {"fps": 60, "videos": {}}, "schema differs"),
    ],
)
def test_incremental_merge_rejects_duplicate_identity_and_incompatible_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: dict,
    message: str,
) -> None:
    output = _make_shard(
        tmp_path, "final", ["day/task/episode_000001"], source_root="/raw/source-a",
    )
    duplicate = _make_shard(
        tmp_path, "duplicate", ["day/task/episode_000001"], source_root="/raw/source-a",
    )
    calls = _fake_runtime(monkeypatch)

    with pytest.raises(shard_merge.StateConflictError, match="duplicate source identity"):
        shard_merge.merge_whole_body_joint_shards([duplicate], output, incremental=True)
    assert calls == []

    incompatible = _make_shard(
        tmp_path, "incompatible", ["day/task/episode_000002"], source_root="/raw/source-b",
    )
    state_path = incompatible / "conversion_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(shard_merge.StateConflictError, match=message):
        shard_merge.merge_whole_body_joint_shards([incompatible], output, incremental=True)
    assert calls == []


def test_incremental_failure_preserves_existing_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _make_shard(tmp_path, "final", ["a/episode_000001"])
    new_shard = _make_shard(tmp_path, "new", ["b/episode_000002"])
    original_state = (output / "conversion_state.json").read_bytes()
    _fake_runtime(monkeypatch, fail=True)

    with pytest.raises(RuntimeError, match="official merge failed"):
        shard_merge.merge_whole_body_joint_shards([new_shard], output, incremental=True)

    assert (output / "conversion_state.json").read_bytes() == original_state
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_incremental_merge_normalizes_episode_metadata_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output = _make_shard(
        tmp_path, "final", ["a/episode_000001", "a/episode_000002"],
    )
    new_shard = _make_shard(tmp_path, "new", ["b/episode_000003"])
    episodes_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "episode_index": [0, 1],
            "meta/episodes/chunk_index": [0, 0],
            "meta/episodes/file_index": [0, 1],
        }),
        episodes_path,
    )
    original_bytes = episodes_path.read_bytes()
    seen: dict[str, object] = {}

    def aggregate(**kwargs):
        aggregate_root = Path(kwargs["roots"][0])
        seen["root"] = aggregate_root
        table = pq.read_table(aggregate_root / "meta/episodes/chunk-000/file-000.parquet")
        seen["file_indices"] = table["meta/episodes/file_index"].to_pylist()
        root = Path(kwargs["aggr_root"])
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        pq.write_table(
            pa.table({
                "episode_index": [0, 1, 2],
                "meta/episodes/chunk_index": [0, 0, 0],
                "meta/episodes/file_index": [0, 1, 2],
            }),
            root / "meta/episodes/chunk-000/file-000.parquet",
        )
        (root / "meta/info.json").write_text(
            json.dumps({"total_episodes": 3, "chunks_size": 1000}), encoding="utf-8",
        )

    monkeypatch.setattr(
        shard_merge,
        "_runtime",
        lambda: {"pa": pa, "pq": pq, "aggregate_datasets": aggregate},
    )

    shard_merge.merge_whole_body_joint_shards([new_shard], output, incremental=True)

    assert seen["file_indices"] == [0, 0]
    assert not Path(seen["root"]).exists()
    assert original_bytes != (output / "meta/episodes/chunk-000/file-000.parquet").read_bytes()
    assert pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")[
        "meta/episodes/file_index"
    ].to_pylist() == [0, 0, 0]


def test_failed_merge_removes_normalized_aggregation_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output = _make_shard(tmp_path, "final", ["a/episode_000001"])
    new_shard = _make_shard(tmp_path, "new", ["b/episode_000002"])
    episodes_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "episode_index": [0],
            "meta/episodes/chunk_index": [0],
            "meta/episodes/file_index": [9],
        }),
        episodes_path,
    )
    original_bytes = episodes_path.read_bytes()
    _fake_runtime(monkeypatch, fail=True)

    with pytest.raises(RuntimeError, match="official merge failed"):
        shard_merge.merge_whole_body_joint_shards([new_shard], output, incremental=True)

    assert episodes_path.read_bytes() == original_bytes
    assert not list(output.parent.glob(".aggregate-input-*"))


def test_incremental_output_must_point_directly_to_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _make_shard(tmp_path, "final", ["a/episode_000001"])
    new_shard = _make_shard(tmp_path, "new", ["b/episode_000002"])
    calls = _fake_runtime(monkeypatch)

    with pytest.raises(ValueError, match="directly at the owner"):
        shard_merge.merge_whole_body_joint_shards(
            [new_shard], output.parent, incremental=True,
        )

    assert calls == []


def test_auxiliary_copy_fallback_when_hardlink_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = _make_shard(tmp_path, "shard-a", ["a/episode_000001"])
    output = tmp_path / "final"
    _fake_runtime(monkeypatch)

    def unavailable(_source: Path, _target: Path) -> None:
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(shard_merge.os, "link", unavailable)
    shard_merge.merge_whole_body_joint_shards([shard], output)

    source = shard / "auxiliary/videos/depth/chunk-000/file-000.mp4"
    target = output / "auxiliary/videos/depth/chunk-000/file-000.mp4"
    assert target.read_bytes() == source.read_bytes()
    assert not os.path.samefile(source, target)
