from __future__ import annotations

import importlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .core import WorkbenchError


DEFAULT_LEROBOT_ROOT = Path(
    "/inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29_26_cz_merged"
)


class LerobotDependencyError(WorkbenchError):
    status_code = 503


class LerobotNotFoundError(WorkbenchError):
    status_code = 404


@dataclass(frozen=True)
class DatasetLocation:
    requested_root: Path
    dataset: Path
    available_datasets: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSnapshot:
    location: DatasetLocation
    signature: tuple[tuple[str, int, int], ...]
    info: dict[str, Any]
    tasks: tuple[dict[str, Any], ...]
    episodes: tuple[dict[str, Any], ...]
    episodes_by_index: dict[int, dict[str, Any]]
    video_keys: tuple[str, ...]


def _require_parquet() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except (ImportError, ModuleNotFoundError) as exc:
        raise LerobotDependencyError(
            "LeRobot 浏览器需要 pyarrow。请在运行服务的 Python 环境中安装 pyarrow。"
        ) from exc


def _existing_directory(value: str | Path) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkbenchError(f"目录不存在: {value}") from exc
    if not path.is_dir():
        raise WorkbenchError(f"目录不存在: {path}")
    return path


def _relative_child(base: Path, relative: str | Path, *, must_exist: bool = True) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise WorkbenchError("LeRobot 元数据包含不安全的绝对路径")
    try:
        candidate = (base / value).resolve(strict=must_exist)
        candidate.relative_to(base)
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkbenchError(f"路径超出 LeRobot dataset 或不存在: {relative}") from exc
    return candidate


def _dataset_location(root_value: str | Path) -> DatasetLocation:
    root = _existing_directory(root_value)
    if (root / "meta" / "info.json").is_file():
        return DatasetLocation(root, root, (root.name,))

    candidates: list[Path] = []
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name.casefold())
    except OSError as exc:
        raise WorkbenchError(f"无法读取 LeRobot 根目录: {exc}") from exc
    for child in children:
        if child.name.startswith(".") or not child.is_dir():
            continue
        try:
            resolved = child.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if (resolved / "meta" / "info.json").is_file():
            candidates.append(resolved)
    if not candidates:
        raise WorkbenchError(f"没有在目录下找到 LeRobot dataset: {root}")
    # The API also accepts a dataset directory directly when a container has several datasets.
    return DatasetLocation(root, candidates[0], tuple(path.name for path in candidates))


def _read_info(dataset: Path) -> dict[str, Any]:
    info_path = _relative_child(dataset, "meta/info.json")
    try:
        value = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkbenchError(f"无法读取 meta/info.json: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkbenchError("meta/info.json 顶层必须是对象")
    if not str(value.get("codebase_version", "")).startswith("v3"):
        raise WorkbenchError("当前浏览器仅支持 LeRobot v3 dataset")
    return value


def _metadata_files(dataset: Path) -> list[Path]:
    files = [
        _relative_child(dataset, "meta/info.json"),
        _relative_child(dataset, "meta/tasks.parquet"),
    ]
    episode_dir = _relative_child(dataset, "meta/episodes")
    for raw_path in sorted(episode_dir.glob("**/*.parquet")):
        try:
            path = raw_path.resolve(strict=True)
            path.relative_to(dataset)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkbenchError("episode 元数据路径超出 LeRobot dataset") from exc
        if path.is_file():
            files.append(path)
    if len(files) == 2:
        raise WorkbenchError("meta/episodes 下没有 parquet 文件")
    return files


def _signature(dataset: Path) -> tuple[tuple[str, int, int], ...]:
    values: list[tuple[str, int, int]] = []
    for path in _metadata_files(dataset):
        stat = path.stat()
        values.append((path.relative_to(dataset).as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(values)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LerobotBrowserService:
    def __init__(self, cache_size: int = 8):
        self.cache_size = cache_size
        self._cache: OrderedDict[str, DatasetSnapshot] = OrderedDict()
        self._lock = threading.RLock()

    def _load(self, root_value: str | Path) -> DatasetSnapshot:
        location = _dataset_location(root_value)
        key = str(location.dataset)
        signature = _signature(location.dataset)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached.signature == signature:
                self._cache.move_to_end(key)
                # Preserve the caller's requested root in generated URLs and responses.
                if cached.location == location:
                    return cached

            parquet = _require_parquet()
            info = _read_info(location.dataset)
            tasks_path = _relative_child(location.dataset, "meta/tasks.parquet")
            try:
                task_rows = parquet.read_table(tasks_path, columns=["task_index", "task"]).to_pylist()
            except Exception as exc:
                raise WorkbenchError(f"无法读取 meta/tasks.parquet: {exc}") from exc

            task_by_name: dict[str, int] = {}
            task_stats: dict[int, dict[str, Any]] = {}
            for row in task_rows:
                task = str(row.get("task", ""))
                task_index = _int(row.get("task_index"), -1)
                if task_index < 0 or not task:
                    continue
                task_by_name[task] = task_index
                task_stats[task_index] = {
                    "taskIndex": task_index,
                    "task": task,
                    "episodeCount": 0,
                    "frameCount": 0,
                }

            features = info.get("features") if isinstance(info.get("features"), dict) else {}
            video_keys = tuple(
                key for key, feature in features.items()
                if isinstance(feature, dict) and feature.get("dtype") == "video"
            )
            columns = [
                "episode_index",
                "tasks",
                "length",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
            ]
            for video_key in video_keys:
                prefix = f"videos/{video_key}"
                columns.extend(
                    [
                        f"{prefix}/chunk_index",
                        f"{prefix}/file_index",
                        f"{prefix}/from_timestamp",
                        f"{prefix}/to_timestamp",
                    ]
                )
            episode_paths = _metadata_files(location.dataset)[2:]
            try:
                schema_names = set(parquet.ParquetFile(episode_paths[0]).schema_arrow.names)
                selected_columns = [column for column in columns if column in schema_names]
                episode_table = parquet.read_table(episode_paths, columns=selected_columns)
                episode_rows = episode_table.to_pylist()
            except Exception as exc:
                raise WorkbenchError(f"无法读取 meta/episodes parquet: {exc}") from exc

            fps = _float(info.get("fps"), 0.0)
            episodes: list[dict[str, Any]] = []
            for row in episode_rows:
                episode_index = _int(row.get("episode_index"), -1)
                if episode_index < 0:
                    continue
                frame_count = max(0, _int(row.get("length")))
                raw_tasks = row.get("tasks") or []
                if isinstance(raw_tasks, str):
                    raw_tasks = [raw_tasks]
                episode_tasks: list[dict[str, Any]] = []
                for task_value in raw_tasks:
                    task = str(task_value)
                    task_index = task_by_name.get(task)
                    episode_tasks.append({"taskIndex": task_index, "task": task})
                    if task_index is not None:
                        task_stats[task_index]["episodeCount"] += 1
                        task_stats[task_index]["frameCount"] += frame_count
                episodes.append(
                    {
                        "episodeIndex": episode_index,
                        "tasks": episode_tasks,
                        "frameCount": frame_count,
                        "durationSeconds": frame_count / fps if fps > 0 else 0.0,
                        "fps": fps,
                        "_metadata": row,
                    }
                )
            episodes.sort(key=lambda episode: episode["episodeIndex"])
            tasks = []
            for task_index in sorted(task_stats):
                task = task_stats[task_index]
                tasks.append(
                    {
                        **task,
                        "durationSeconds": task["frameCount"] / fps if fps > 0 else 0.0,
                    }
                )
            snapshot = DatasetSnapshot(
                location=location,
                signature=signature,
                info=info,
                tasks=tuple(tasks),
                episodes=tuple(episodes),
                episodes_by_index={episode["episodeIndex"]: episode for episode in episodes},
                video_keys=video_keys,
            )
            self._cache[key] = snapshot
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return snapshot

    @staticmethod
    def _public_episode(episode: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in episode.items() if key != "_metadata"}

    def summary(self, root_value: str | Path = DEFAULT_LEROBOT_ROOT) -> dict[str, Any]:
        snapshot = self._load(root_value)
        info = snapshot.info
        features_value = info.get("features") if isinstance(info.get("features"), dict) else {}
        features = []
        for name, feature_value in features_value.items():
            feature = feature_value if isinstance(feature_value, dict) else {}
            features.append(
                {
                    "name": name,
                    "dtype": feature.get("dtype"),
                    "shape": feature.get("shape"),
                }
            )
        return {
            "root": str(snapshot.location.requested_root),
            "dataset": snapshot.location.dataset.name,
            "datasetPath": str(snapshot.location.dataset),
            "availableDatasets": list(snapshot.location.available_datasets),
            "codebaseVersion": info.get("codebase_version"),
            "robotType": info.get("robot_type"),
            "fps": _float(info.get("fps")),
            "totalEpisodes": len(snapshot.episodes),
            "totalFrames": _int(info.get("total_frames")),
            "totalTasks": len(snapshot.tasks),
            "videoKeys": list(snapshot.video_keys),
            "features": features,
            "tasks": list(snapshot.tasks),
        }

    def episodes(
        self,
        root_value: str | Path = DEFAULT_LEROBOT_ROOT,
        *,
        task_index: int | None = None,
        query: str = "",
        page: int = 1,
        page_size: int = 48,
    ) -> dict[str, Any]:
        snapshot = self._load(root_value)
        normalized_query = query.strip()
        values = snapshot.episodes
        if task_index is not None:
            values = tuple(
                episode for episode in values
                if any(task["taskIndex"] == task_index for task in episode["tasks"])
            )
        if normalized_query:
            values = tuple(
                episode for episode in values
                if normalized_query in str(episode["episodeIndex"])
            )
        total = len(values)
        total_pages = math.ceil(total / page_size) if total else 0
        offset = (page - 1) * page_size
        items = [self._public_episode(episode) for episode in values[offset : offset + page_size]]
        return {
            "root": str(snapshot.location.requested_root),
            "dataset": snapshot.location.dataset.name,
            "taskIndex": task_index,
            "query": normalized_query,
            "page": page,
            "pageSize": page_size,
            "total": total,
            "totalPages": total_pages,
            "items": items,
        }

    def detail(self, episode_index: int, root_value: str | Path = DEFAULT_LEROBOT_ROOT) -> dict[str, Any]:
        snapshot = self._load(root_value)
        episode = snapshot.episodes_by_index.get(episode_index)
        if episode is None:
            raise LerobotNotFoundError(f"Episode {episode_index} 不存在")
        metadata = episode["_metadata"]
        info = snapshot.info
        try:
            data_relative = str(info.get("data_path", "")).format(
                chunk_index=_int(metadata.get("data/chunk_index")),
                file_index=_int(metadata.get("data/file_index")),
            )
        except (KeyError, ValueError) as exc:
            raise WorkbenchError(f"data_path 模板无效: {exc}") from exc
        data_path = _relative_child(snapshot.location.dataset, data_relative)
        parquet = _require_parquet()
        try:
            parquet_file = parquet.ParquetFile(data_path)
            data_rows = parquet_file.metadata.num_rows
            data_columns = parquet_file.schema_arrow.names
        except Exception as exc:
            raise WorkbenchError(f"无法读取 episode 数据 parquet: {exc}") from exc

        streams: list[dict[str, Any]] = []
        video_template = info.get("video_path")
        if snapshot.video_keys and not isinstance(video_template, str):
            raise WorkbenchError("info.json 缺少 video_path 模板")
        encoded_root = quote(str(snapshot.location.requested_root), safe="")
        for key in snapshot.video_keys:
            prefix = f"videos/{key}"
            try:
                relative = video_template.format(
                    video_key=key,
                    chunk_index=_int(metadata.get(f"{prefix}/chunk_index")),
                    file_index=_int(metadata.get(f"{prefix}/file_index")),
                )
            except (KeyError, ValueError) as exc:
                raise WorkbenchError(f"video_path 模板无效: {exc}") from exc
            path = _relative_child(snapshot.location.dataset, relative, must_exist=False)
            exists = path.is_file()
            from_timestamp = _float(metadata.get(f"{prefix}/from_timestamp"))
            to_timestamp = _float(metadata.get(f"{prefix}/to_timestamp"))
            streams.append(
                {
                    "key": key,
                    "path": path.relative_to(snapshot.location.dataset).as_posix(),
                    "exists": exists,
                    "sizeBytes": path.stat().st_size if exists else None,
                    "fromTimestamp": from_timestamp,
                    "toTimestamp": to_timestamp,
                    "durationSeconds": max(0.0, to_timestamp - from_timestamp),
                    "mediaUrl": (
                        f"/api/lerobot/episodes/{episode_index}/media/{quote(key, safe='')}"
                        f"?root={encoded_root}"
                    ) if exists else None,
                }
            )
        return {
            "root": str(snapshot.location.requested_root),
            "dataset": snapshot.location.dataset.name,
            **self._public_episode(episode),
            "data": {
                "path": data_path.relative_to(snapshot.location.dataset).as_posix(),
                "sizeBytes": data_path.stat().st_size,
                "rows": episode["frameCount"],
                "fileRows": data_rows,
                "fromIndex": _int(metadata.get("dataset_from_index")),
                "toIndex": _int(metadata.get("dataset_to_index")),
                "columns": data_columns,
            },
            "streams": streams,
        }

    def media(self, episode_index: int, stream_key: str, root_value: str | Path = DEFAULT_LEROBOT_ROOT) -> Path:
        snapshot = self._load(root_value)
        episode = snapshot.episodes_by_index.get(episode_index)
        if episode is None:
            raise LerobotNotFoundError(f"Episode {episode_index} 不存在")
        if stream_key not in snapshot.video_keys:
            raise LerobotNotFoundError(f"视频流不存在: {stream_key}")
        metadata = episode["_metadata"]
        prefix = f"videos/{stream_key}"
        video_template = snapshot.info.get("video_path")
        if not isinstance(video_template, str):
            raise WorkbenchError("info.json 缺少 video_path 模板")
        try:
            relative = video_template.format(
                video_key=stream_key,
                chunk_index=_int(metadata.get(f"{prefix}/chunk_index")),
                file_index=_int(metadata.get(f"{prefix}/file_index")),
            )
        except (KeyError, ValueError) as exc:
            raise WorkbenchError(f"video_path 模板无效: {exc}") from exc
        path = _relative_child(snapshot.location.dataset, relative, must_exist=False)
        if not path.is_file():
            raise LerobotNotFoundError(f"视频文件不存在: {relative}")
        return path
