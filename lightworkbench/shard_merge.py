"""Merge completed whole-body LeRobot shards without re-encoding videos."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lerobot_converter import (
    AUXILIARY_INDEX_PATH,
    CONVERSION_STATE_FILENAME,
    MERGED_DATASET_LAYOUT,
    STATE_VERSION,
    WHOLE_BODY_JOINT,
    OptionalDependencyError,
    StateConflictError,
    atomic_write_json,
)


def _runtime() -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from lerobot.datasets.aggregate import aggregate_datasets
    except ImportError as exc:
        raise OptionalDependencyError(
            "LeRobot shard merge dependencies are unavailable; install the optional "
            "'lerobot' dependency group"
        ) from exc
    return {"pa": pa, "pq": pq, "aggregate_datasets": aggregate_datasets}


def _owner_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / CONVERSION_STATE_FILENAME).is_file():
        return path
    owner = path / WHOLE_BODY_JOINT
    if (owner / CONVERSION_STATE_FILENAME).is_file():
        return owner
    raise StateConflictError(f"{path}: no completed whole_body_joint conversion state")


def _load_state(root: Path) -> dict[str, Any]:
    try:
        state = json.loads((root / CONVERSION_STATE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateConflictError(f"{root}: invalid conversion state: {exc}") from exc
    if not isinstance(state, dict):
        raise StateConflictError(f"{root}: conversion state must be an object")
    if (
        state.get("version") != STATE_VERSION
        or state.get("action_mode") != WHOLE_BODY_JOINT
        or state.get("dataset_layout") != MERGED_DATASET_LAYOUT
        or not isinstance(state.get("episodes"), list)
        or not state.get("source_root")
    ):
        raise StateConflictError(f"{root}: not a completed merged whole_body_joint shard")
    if state.get("pending_episode") is not None:
        raise StateConflictError(f"{root}: shard has an uncommitted pending episode")
    for index, entry in enumerate(state["episodes"]):
        if not isinstance(entry, Mapping) or entry.get("lerobot_episode_index") != index:
            raise StateConflictError(f"{root}: shard episode indices are not contiguous")
        for key in ("source_relative_path", "source_signature", "stored_task"):
            if key not in entry:
                raise StateConflictError(f"{root}: episode {index} has no {key}")
        if not isinstance(entry["source_signature"], Mapping):
            raise StateConflictError(f"{root}: episode {index} has invalid source_signature")
        if not isinstance(entry["stored_task"], str) or not entry["stored_task"]:
            raise StateConflictError(f"{root}: episode {index} has invalid stored_task")
    return state


def _validate_states(roots: Sequence[Path], states: Sequence[Mapping[str, Any]]) -> None:
    first = states[0]
    for root, state in zip(roots[1:], states[1:], strict=True):
        for key in ("conversion_config", "schema", "source_root"):
            if state.get(key) != first.get(key):
                raise StateConflictError(f"{root}: shard {key} differs from the first shard")
    identities: set[str] = set()
    for root, state in zip(roots, states, strict=True):
        for entry in state["episodes"]:
            identity = str(entry["source_relative_path"])
            relative = Path(identity)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise StateConflictError(f"{root}: invalid source_relative_path {identity!r}")
            if identity in identities:
                raise StateConflictError(f"duplicate source_relative_path across shards: {identity}")
            identities.add(identity)


def _read_chunk_size(staging: Path) -> int:
    try:
        info = json.loads((staging / "meta/info.json").read_text(encoding="utf-8"))
        size = int(info["chunks_size"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise StateConflictError(f"{staging}: merged LeRobot metadata has no valid chunks_size") from exc
    if size <= 0:
        raise StateConflictError(f"{staging}: merged LeRobot chunks_size must be positive")
    return size


def _merge_auxiliary(
    runtime: Mapping[str, Any],
    roots: Sequence[Path],
    states: Sequence[Mapping[str, Any]],
    staging: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    offset = 0
    chunk_size = _read_chunk_size(staging)
    schema = None
    for root, state in zip(roots, states, strict=True):
        index_path = root / AUXILIARY_INDEX_PATH
        if not index_path.is_file():
            raise StateConflictError(f"{root}: missing auxiliary/index.parquet")
        table = runtime["pq"].read_table(index_path)
        if schema is None:
            schema = table.schema
        elif table.schema != schema:
            raise StateConflictError(f"{root}: auxiliary index schema differs from the first shard")
        local_count = len(state["episodes"])
        seen: set[tuple[int, str]] = set()
        for raw in table.to_pylist():
            row = dict(raw)
            try:
                local_index = int(row["episode_index"])
                stream = str(row["stream"])
                source_relative = Path(str(row["relative_path"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise StateConflictError(f"{index_path}: malformed auxiliary row") from exc
            if local_index < 0 or local_index >= local_count:
                raise StateConflictError(
                    f"{index_path}: auxiliary episode {local_index} is outside shard state"
                )
            key = (local_index, stream)
            if key in seen:
                raise StateConflictError(f"{index_path}: duplicate auxiliary row {key}")
            seen.add(key)
            stream_path = Path(stream)
            if len(stream_path.parts) != 1 or stream_path.name in ("", ".", ".."):
                raise StateConflictError(f"{index_path}: unsafe auxiliary stream {stream!r}")
            if source_relative.is_absolute() or ".." in source_relative.parts:
                raise StateConflictError(f"{index_path}: unsafe auxiliary path {source_relative}")
            source = root / source_relative
            if not source.is_file():
                raise StateConflictError(f"{index_path}: missing auxiliary video {source_relative}")
            global_index = offset + local_index
            suffix = source.suffix or ".mp4"
            relative = (
                Path("auxiliary/videos")
                / stream
                / f"chunk-{global_index // chunk_size:03d}"
                / f"file-{global_index % chunk_size:03d}{suffix}"
            )
            target = staging / relative
            if target.exists():
                raise StateConflictError(f"duplicate merged auxiliary target: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            row["episode_index"] = global_index
            row["relative_path"] = relative.as_posix()
            rows.append(row)
        offset += local_count

    index_path = staging / AUXILIARY_INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if schema is None:
        raise StateConflictError("no auxiliary index schema found")
    rows.sort(key=lambda row: (int(row["episode_index"]), str(row["stream"])))
    table = runtime["pa"].Table.from_pylist(rows, schema=schema)
    runtime["pq"].write_table(table, index_path)


def _merged_state(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    merged = copy.deepcopy(dict(states[0]))
    episodes: list[dict[str, Any]] = []
    for state in states:
        for raw in state["episodes"]:
            entry = copy.deepcopy(dict(raw))
            entry["lerobot_episode_index"] = len(episodes)
            episodes.append(entry)
    merged["episodes"] = episodes
    merged["stored_tasks"] = sorted({str(entry["stored_task"]) for entry in episodes})
    merged["created_at"] = now
    merged["updated_at"] = now
    merged.pop("pending_episode", None)
    return merged


def _publish(staging: Path, output_root: Path, overwrite: bool) -> None:
    if not output_root.exists():
        os.replace(staging, output_root)
        return
    if not overwrite:
        raise FileExistsError(f"output already exists: {output_root}")
    backup = output_root.with_name(f".{output_root.name}.replaced-{uuid.uuid4().hex}")
    os.replace(output_root, backup)
    try:
        os.replace(staging, output_root)
    except Exception:
        os.replace(backup, output_root)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def merge_whole_body_joint_shards(
    shard_roots: Sequence[str | Path],
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Merge ordered completed shards into one atomically published owner dataset.

    Each input may point either at a ``whole_body_joint`` directory or its bundle
    parent. The input order defines the final global episode order.
    """

    if not shard_roots:
        raise ValueError("at least one shard is required")
    roots = [_owner_root(Path(item)) for item in shard_roots]
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate shard root")
    output_root = Path(output_root).expanduser().resolve()
    if output_root in roots:
        raise ValueError("output_root must not be one of the shard roots")
    states = [_load_state(root) for root in roots]
    _validate_states(roots, states)
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_root}")

    runtime = _runtime()
    staging = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime["aggregate_datasets"](
            repo_ids=[str(state["repo_id"]) for state in states],
            aggr_repo_id=str(states[0]["repo_id"]),
            roots=roots,
            aggr_root=staging,
            concatenate_videos=False,
            concatenate_data=False,
        )
        _merge_auxiliary(runtime, roots, states, staging)
        state = _merged_state(states)
        info = json.loads((staging / "meta/info.json").read_text(encoding="utf-8"))
        if int(info.get("total_episodes", -1)) != len(state["episodes"]):
            raise StateConflictError("official merge episode count differs from conversion state")
        atomic_write_json(staging / CONVERSION_STATE_FILENAME, state)
        _publish(staging, output_root, overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_root


__all__ = ["merge_whole_body_joint_shards"]
