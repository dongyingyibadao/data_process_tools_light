from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FFPROBE, INTERNAL_NAMES, IO_WORKERS


EPISODE_RE = re.compile(r"episode_\d+$")
NATURAL_RE = re.compile(r"(\d+)")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


class WorkbenchError(RuntimeError):
    status_code = 400


class ConflictError(WorkbenchError):
    status_code = 409


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in NATURAL_RE.split(value)]


def normalize_root(value: str | Path, *, create: bool = False) -> Path:
    raw = Path(value).expanduser()
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    root = raw.resolve()
    if not root.is_dir():
        raise WorkbenchError(f"目录不存在: {root}")
    return root


def contained(root: Path, value: str | Path, *, directory: bool = True) -> Path:
    candidate = Path(value)
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkbenchError("路径超出源根目录") from exc
    if directory and not candidate.is_dir():
        raise WorkbenchError("目录不存在")
    return candidate


def resolve_episode(root: Path, relative: str) -> Path:
    episode = contained(root, relative)
    if not EPISODE_RE.fullmatch(episode.name):
        raise WorkbenchError("目标不是 episode_数字 目录")
    return episode


def resolve_video(episode: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise WorkbenchError("manifest 视频路径不能为绝对路径")
    target = (episode / value).resolve()
    try:
        target.relative_to(episode.resolve())
    except ValueError as exc:
        raise WorkbenchError("manifest 视频路径越过 Episode 边界") from exc
    return target


def hidden(name: str) -> bool:
    return name.startswith(".") or name in INTERNAL_NAMES


class BrowseService:
    def __init__(self, cache_size: int = 128):
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _children(path: Path) -> list[Path]:
        values: list[Path] = []
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if hidden(entry.name) or not entry.is_dir(follow_symlinks=False):
                        continue
                    values.append(Path(entry.path))
        except OSError as exc:
            raise WorkbenchError(f"无法读取目录: {exc}") from exc
        return sorted(values, key=lambda item: natural_key(item.name))

    def browse(self, root_value: str, path_value: str = "", refresh: bool = False) -> dict[str, Any]:
        root = normalize_root(root_value)
        selected = contained(root, path_value or ".")
        relative = selected.relative_to(root).as_posix()
        relative = "" if relative == "." else relative
        key = (str(root), relative)
        with self._lock:
            if refresh:
                self._cache.pop(key, None)
            elif key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value
                return {**value, "cached": True}

        children = [item for item in self._children(selected) if not item.is_symlink()]
        episodes = [
            item for item in children
            if EPISODE_RE.fullmatch(item.name) and (item / "videos").is_dir()
        ]

        crumbs = [{"name": root.name or str(root), "path": ""}]
        cursor = Path()
        for part in Path(relative).parts:
            cursor /= part
            crumbs.append({"name": part, "path": cursor.as_posix()})
        if episodes:
            payload: dict[str, Any] = {
                "view": "episodes",
                "root": str(root),
                "path": relative,
                "breadcrumbs": crumbs,
                "episodes": [
                    {
                        "name": item.name,
                        "episode": item.relative_to(root).as_posix(),
                        "parent": item.parent.relative_to(root).as_posix(),
                    }
                    for item in sorted(episodes, key=lambda item: natural_key(item.relative_to(root).as_posix()))
                ],
                "totalEpisodeCount": len(episodes),
            }
        else:
            folders: list[dict[str, Any]] = []
            total_episode_count = 0
            for item in children:
                direct_episodes = sum(
                    1 for child in self._children(item)
                    if not child.is_symlink()
                    and EPISODE_RE.fullmatch(child.name)
                    and (child / "videos").is_dir()
                )
                total_episode_count += direct_episodes
                folders.append({
                    "name": item.name,
                    "path": item.relative_to(root).as_posix(),
                    "episodeCount": direct_episodes,
                })
            payload = {
                "view": "folders",
                "root": str(root),
                "path": relative,
                "breadcrumbs": crumbs,
                "folders": folders,
                "totalEpisodeCount": total_episode_count,
            }
        with self._lock:
            self._cache[key] = payload
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return {**payload, "cached": False}


@dataclass(frozen=True)
class ManifestData:
    frame_count: int
    fps: float
    task: str
    stream_paths: dict[str, str]
    errors: tuple[str, ...]


def read_manifest(episode: Path) -> ManifestData:
    manifest = episode / "manifest.jsonl"
    if not manifest.is_file() or manifest.stat().st_size == 0:
        return ManifestData(0, 30.0, "", {}, ("manifest.jsonl 缺失或为空",))
    header: dict[str, Any] = {}
    count = 0
    paths: dict[str, set[str]] = {}
    missing: dict[str, int] = {}
    errors: list[str] = []
    try:
        with manifest.open("r", encoding="utf-8", errors="strict") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"manifest 第 {line_no} 行不是有效 JSON")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"manifest 第 {line_no} 行不是对象")
                    continue
                if row.get("_type") == "session_header":
                    header = row
                elif isinstance(row.get("_type"), str):
                    continue
                else:
                    videos = row.get("videos")
                    if not isinstance(videos, dict):
                        errors.append(f"第 {count} 帧缺少 videos")
                    else:
                        known = set(paths) | set(videos)
                        for name in known:
                            entry = videos.get(name)
                            path = entry.get("path") if isinstance(entry, dict) else None
                            if isinstance(path, str) and path:
                                paths.setdefault(name, set()).add(path)
                            else:
                                missing[name] = missing.get(name, 0) + 1
                    count += 1
    except (OSError, UnicodeError) as exc:
        errors.append(f"manifest 无法读取: {exc}")
    result: dict[str, str] = {}
    for name, values in paths.items():
        if len(values) != 1:
            errors.append(f"视频流 {name} 在 Episode 内引用多个文件")
        else:
            result[name] = next(iter(values))
        if missing.get(name):
            errors.append(f"视频流 {name} 有 {missing[name]} 帧缺少引用")
    raw_fps = header.get("fps_target", 30)
    try:
        fps = float(raw_fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError
    except (TypeError, ValueError):
        fps = 30.0
        errors.append("fps_target 非法")
    task = str(header.get("task_description") or "")
    if not task:
        try:
            meta = json.loads((episode / "task_meta.json").read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                task = str(meta.get("task_description") or meta.get("description") or "")
        except (OSError, json.JSONDecodeError):
            errors.append("task_meta.json 缺失或损坏")
    return ManifestData(count, fps, task, result, tuple(errors))


def probe_video(path: Path, *, decoded: bool = False) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"valid": False, "error": "视频缺失或为空", "frames": 0}
    entries = "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames" if decoded else "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_packets"
    command = [FFPROBE, "-v", "error", "-count_frames" if decoded else "-count_packets", "-select_streams", "v:0", "-show_entries", entries, "-of", "json", str(path)]
    try:
        done = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        stream = json.loads(done.stdout)["streams"][0]
        numerator, denominator = str(stream.get("avg_frame_rate") or "0/1").split("/", 1)
        fps = float(numerator) / float(denominator)
        frames = int(stream.get("nb_read_frames" if decoded else "nb_read_packets") or 0)
        if frames <= 0 or fps <= 0:
            raise ValueError("无有效帧")
        return {
            "valid": True,
            "codec": str(stream.get("codec_name") or "unknown"),
            "pixelFormat": str(stream.get("pix_fmt") or "unknown"),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "fps": fps,
            "frames": frames,
        }
    except (OSError, subprocess.SubprocessError, KeyError, IndexError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        return {"valid": False, "error": f"FFprobe 无法读取: {type(exc).__name__}", "frames": 0}


def source_token(episode: Path, stream_paths: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in ("manifest.jsonl", "task_meta.json"):
        path = episode / name
        digest.update(name.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    for name, relative in sorted(stream_paths.items()):
        digest.update(name.encode())
        digest.update(relative.encode())
        try:
            stat = resolve_video(episode, relative).stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        except (OSError, WorkbenchError):
            digest.update(b"<missing>")
    return digest.hexdigest()


class EpisodeService:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=IO_WORKERS, thread_name_prefix="detail")
        self._registry: dict[tuple[str, str], dict[str, str]] = {}
        self._lock = threading.Lock()

    def detail(self, root_value: str, relative: str) -> dict[str, Any]:
        root = normalize_root(root_value)
        episode = resolve_episode(root, relative)
        manifest_future = self._pool.submit(read_manifest, episode)
        meta = manifest_future.result()
        futures = {}
        pre_errors = list(meta.errors)
        for name, value in meta.stream_paths.items():
            try:
                path = resolve_video(episode, value)
            except WorkbenchError as exc:
                pre_errors.append(f"{name}: {exc}")
                continue
            futures[self._pool.submit(probe_video, path)] = (name, value)
        streams: list[dict[str, Any]] = []
        for future in as_completed(futures):
            name, value = futures[future]
            probe = future.result()
            valid = bool(probe.get("valid")) and probe.get("frames") == meta.frame_count and math.isclose(float(probe.get("fps") or 0), meta.fps, abs_tol=0.05)
            error = probe.get("error")
            if probe.get("valid") and probe.get("frames") != meta.frame_count:
                error = f"视频 {probe.get('frames')} 帧，manifest {meta.frame_count} 帧"
            elif probe.get("valid") and not math.isclose(float(probe.get("fps") or 0), meta.fps, abs_tol=0.05):
                error = f"视频 FPS {probe.get('fps')}，manifest FPS {meta.fps}"
            codec = probe.get("codec")
            pixel = str(probe.get("pixelFormat") or "")
            playable = valid and Path(value).suffix.casefold() == ".mp4" and codec == "h264" and pixel in {"yuv420p", "yuvj420p"} and "depth" not in name.casefold()
            streams.append({"name": name, "path": value, **probe, "valid": valid, "error": error, "browserPlayable": playable})
        streams.sort(key=lambda item: natural_key(item["name"]))
        registered = {item["name"]: item["path"] for item in streams}
        with self._lock:
            self._registry[(str(root), relative)] = registered
        token = source_token(episode, meta.stream_paths)
        valid = meta.frame_count >= 2 and not pre_errors and bool(streams) and all(item["valid"] for item in streams) and len(streams) == len(meta.stream_paths)
        return {
            "episode": relative,
            "frameCount": meta.frame_count,
            "fps": meta.fps,
            "durationSec": meta.frame_count / meta.fps,
            "task": meta.task,
            "streams": streams,
            "issues": pre_errors + [f"{item['name']}: {item['error']}" for item in streams if not item["valid"]],
            "valid": valid,
            "sourceToken": token,
        }

    def media(self, root_value: str, relative: str, stream: str) -> Path:
        root = normalize_root(root_value)
        episode = resolve_episode(root, relative)
        with self._lock:
            path_value = self._registry.get((str(root), relative), {}).get(stream)
        if path_value is None:
            detail = self.detail(str(root), relative)
            path_value = next((item["path"] for item in detail["streams"] if item["name"] == stream), None)
        if path_value is None:
            raise WorkbenchError("视频流未在该 Episode 登记")
        path = resolve_video(episode, path_value)
        if not path.is_file():
            raise WorkbenchError("视频文件不存在")
        return path
