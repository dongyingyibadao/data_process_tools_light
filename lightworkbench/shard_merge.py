"""Merge completed whole-body LeRobot shards without re-encoding videos."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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


def _resolved_source_root(value: Any, root: Path) -> str:
    if not isinstance(value, str) or not value:
        raise StateConflictError(f"{root}: invalid source_root")
    return str(Path(value).expanduser().resolve())


def _entry_source_root(entry: Mapping[str, Any], state: Mapping[str, Any], root: Path) -> str:
    return _resolved_source_root(entry.get("source_root", state.get("source_root")), root)


def _validate_states(
    roots: Sequence[Path],
    states: Sequence[Mapping[str, Any]],
    *,
    allow_multiple_sources: bool = False,
) -> None:
    first = states[0]
    for root, state in zip(roots[1:], states[1:], strict=True):
        for key in ("conversion_config", "schema"):
            if state.get(key) != first.get(key):
                raise StateConflictError(f"{root}: shard {key} differs from the first shard")
        if not allow_multiple_sources and _resolved_source_root(
            state.get("source_root"), root
        ) != _resolved_source_root(first.get("source_root"), roots[0]):
            raise StateConflictError(f"{root}: shard source_root differs from the first shard")
    identities: set[tuple[str, str]] = set()
    for root, state in zip(roots, states, strict=True):
        for entry in state["episodes"]:
            source_relative_path = str(entry["source_relative_path"])
            relative = Path(source_relative_path)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise StateConflictError(
                    f"{root}: invalid source_relative_path {source_relative_path!r}"
                )
            identity = (_entry_source_root(entry, state, root), relative.as_posix())
            if identity in identities:
                raise StateConflictError(
                    "duplicate source identity across shards: "
                    f"source_root={identity[0]} source_relative_path={identity[1]}"
                )
            identities.add(identity)


def _bundle_root_for_owner(output_root: Path) -> Path:
    return output_root.parent


@contextmanager
def _bundle_lock(output_root: Path) -> Iterator[None]:
    lock_path = _bundle_root_for_owner(output_root) / ".bundle" / "convert-merged.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_chunk_size(staging: Path) -> int:
    try:
        info = json.loads((staging / "meta/info.json").read_text(encoding="utf-8"))
        size = int(info["chunks_size"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise StateConflictError(f"{staging}: merged LeRobot metadata has no valid chunks_size") from exc
    if size <= 0:
        raise StateConflictError(f"{staging}: merged LeRobot chunks_size must be positive")
    return size


_EPISODE_CHUNK_COLUMN = "meta/episodes/chunk_index"
_EPISODE_FILE_COLUMN = "meta/episodes/file_index"


def _episode_metadata_files(root: Path) -> list[tuple[Path, int, int]]:
    files: list[tuple[Path, int, int]] = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        chunk_token = path.parent.name.removeprefix("chunk-")
        file_token = path.stem.removeprefix("file-")
        if not chunk_token.isdigit() or not file_token.isdigit():
            raise StateConflictError(f"{path}: invalid Episode metadata path")
        files.append((path, int(chunk_token), int(file_token)))
    return files


def _normalized_episode_table(
    runtime: Mapping[str, Any], path: Path, chunk_index: int, file_index: int,
) -> tuple[Any, bool]:
    try:
        table = runtime["pq"].read_table(path)
    except Exception as exc:
        raise StateConflictError(f"{path}: cannot read Episode metadata: {exc}") from exc
    changed = False
    for column, value in (
        (_EPISODE_CHUNK_COLUMN, chunk_index),
        (_EPISODE_FILE_COLUMN, file_index),
    ):
        index = table.schema.get_field_index(column)
        if index < 0:
            raise StateConflictError(f"{path}: Episode metadata has no {column}")
        if any(item != value for item in table[column].to_pylist()):
            field = table.schema.field(index)
            values = runtime["pa"].array([value] * table.num_rows, type=field.type)
            table = table.set_column(index, field, values)
            changed = True
    return table, changed


def _normalize_episode_metadata_in_place(runtime: Mapping[str, Any], root: Path) -> bool:
    changed_any = False
    for path, chunk_index, file_index in _episode_metadata_files(root):
        table, changed = _normalized_episode_table(runtime, path, chunk_index, file_index)
        if not changed:
            continue
        temporary = path.with_name(f".{path.name}.normalizing-{uuid.uuid4().hex}")
        try:
            runtime["pq"].write_table(table, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        changed_any = True
    return changed_any


def _link_tree_except(source: Path, target: Path, excluded: set[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name not in excluded:
            (target / item.name).symlink_to(item, target_is_directory=item.is_dir())


def _normalized_aggregation_view(
    runtime: Mapping[str, Any], root: Path, scratch_parent: Path,
) -> Path | None:
    metadata_files = _episode_metadata_files(root)
    normalized: list[tuple[Path, Any]] = []
    for path, chunk_index, file_index in metadata_files:
        table, changed = _normalized_episode_table(runtime, path, chunk_index, file_index)
        if changed:
            normalized.append((path, table))
    if not normalized:
        return None

    tables = {path: table for path, table in normalized}
    view = scratch_parent / f".aggregate-input-{root.name}-{uuid.uuid4().hex}"
    try:
        _link_tree_except(root, view, {"meta"})
        _link_tree_except(root / "meta", view / "meta", {"episodes"})
        for source, _chunk_index, _file_index in metadata_files:
            target = view / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            table = tables.get(source)
            if table is None:
                target.symlink_to(source)
            else:
                runtime["pq"].write_table(table, target)
    except Exception:
        shutil.rmtree(view, ignore_errors=True)
        raise
    return view


@contextmanager
def _aggregation_roots(
    runtime: Mapping[str, Any], roots: Sequence[Path], scratch_parent: Path,
) -> Iterator[list[Path]]:
    views: list[Path] = []
    aggregation_roots: list[Path] = []
    try:
        for root in roots:
            view = _normalized_aggregation_view(runtime, root, scratch_parent)
            if view is not None:
                views.append(view)
            aggregation_roots.append(view or root)
        yield aggregation_roots
    finally:
        for view in views:
            shutil.rmtree(view, ignore_errors=True)


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
            try:
                os.link(source, target)
            except OSError:
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


def _merged_state(
    roots: Sequence[Path],
    states: Sequence[Mapping[str, Any]],
    *,
    preserve_created_at: bool = False,
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    merged = copy.deepcopy(dict(states[0]))
    episodes: list[dict[str, Any]] = []
    source_roots: list[str] = []
    for root, state in zip(roots, states, strict=True):
        for raw in state["episodes"]:
            entry = copy.deepcopy(dict(raw))
            source_root = _entry_source_root(entry, state, root)
            entry["source_root"] = source_root
            entry["lerobot_episode_index"] = len(episodes)
            episodes.append(entry)
            if source_root not in source_roots:
                source_roots.append(source_root)
    merged["episodes"] = episodes
    merged["stored_tasks"] = sorted({str(entry["stored_task"]) for entry in episodes})
    source_roots.sort()
    merged["source_roots"] = source_roots
    merged["source_root"] = source_roots[0]
    merged["source_task"] = source_roots[0]
    if not preserve_created_at:
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
    incremental: bool = False,
) -> Path:
    """Merge ordered completed shards into one atomically published owner dataset.

    Each input may point either at a ``whole_body_joint`` directory or its bundle
    parent. The input order defines the final global episode order. Incremental
    mode uses an existing output owner as the first aggregation input.
    """

    if not shard_roots:
        raise ValueError("at least one shard is required")
    output_root = Path(output_root).expanduser().resolve()
    with _bundle_lock(output_root):
        roots = [_owner_root(Path(item)) for item in shard_roots]
        if len(roots) != len(set(roots)):
            raise ValueError("duplicate shard root")
        if output_root in roots:
            raise ValueError("output_root must not be one of the shard roots")
        if incremental:
            if not output_root.exists():
                raise FileNotFoundError(f"incremental output does not exist: {output_root}")
            existing_owner = _owner_root(output_root)
            if existing_owner != output_root:
                raise ValueError("incremental output_root must point directly at the owner dataset")
            roots.insert(0, existing_owner)
        states = [_load_state(root) for root in roots]
        _validate_states(roots, states, allow_multiple_sources=incremental)
        if output_root.exists() and not incremental and not overwrite:
            raise FileExistsError(f"output already exists: {output_root}")

        runtime = _runtime()
        staging = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
        staging.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _aggregation_roots(runtime, roots, staging.parent) as aggregation_roots:
                runtime["aggregate_datasets"](
                    repo_ids=[str(state["repo_id"]) for state in states],
                    aggr_repo_id=str(states[0]["repo_id"]),
                    roots=aggregation_roots,
                    aggr_root=staging,
                    concatenate_videos=False,
                    concatenate_data=False,
                )
            _normalize_episode_metadata_in_place(runtime, staging)
            _merge_auxiliary(runtime, roots, states, staging)
            state = _merged_state(
                roots, states, preserve_created_at=incremental,
            )
            info = json.loads((staging / "meta/info.json").read_text(encoding="utf-8"))
            if int(info.get("total_episodes", -1)) != len(state["episodes"]):
                raise StateConflictError("official merge episode count differs from conversion state")
            atomic_write_json(staging / CONVERSION_STATE_FILENAME, state)
            _publish(staging, output_root, overwrite or incremental)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return output_root


__all__ = ["merge_whole_body_joint_shards"]
