from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import FFMPEG, RESOURCES
from .core import (
    ConflictError,
    EpisodeService,
    WorkbenchError,
    normalize_root,
    read_manifest,
    resolve_episode,
    resolve_video,
    source_token,
)
from .validation import validate_episode


CSV_NAME = "CUT_HISTORY.csv"
CSV_FIELDS = [
    "operation_id", "revision", "overwritten_operation_id", "source_root", "episode",
    "operator", "mode", "removed_ranges", "source_frames", "output_frames",
    "source_token", "output_fingerprint", "output_path", "completed_at_utc",
]
FINGERPRINT_VERSION = "sha256-relative-path-size-mtime-ns-v1"
FINGERPRINT_EXCLUDES = ("CUT_INFO.json",)
TIMESTAMP_REWRITE_VERSION = 2
MANIFEST_NORMALIZATION_VERSION = 1


class QueueFullError(WorkbenchError):
    status_code = 429


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def paths_overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def normalize_ranges(values: list[list[int]], frame_count: int) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    for value in values:
        if not isinstance(value, list) or len(value) != 2:
            raise WorkbenchError("删除区间必须是 [start, end)")
        start, end = value
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise WorkbenchError("删除区间帧号必须是整数")
        start, end = max(0, start), min(frame_count, end)
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()
    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def kept_spans(removed: list[tuple[int, int]], frame_count: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in removed:
        if cursor < start:
            spans.append((cursor, start))
        cursor = end
    if cursor < frame_count:
        spans.append((cursor, frame_count))
    return spans




def trim_video(source: Path, destination: Path, spans: list[tuple[int, int]], fps: float,
               stream_name: str, pixel_format: str, threads: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{index}]" for index, (start, end) in enumerate(spans)]
    if len(spans) == 1:
        filters.append(f"[v0]setpts=N/{fps}/TB[outv]")
    else:
        labels = "".join(f"[v{index}]" for index in range(len(spans)))
        filters.append(f"{labels}concat=n={len(spans)}:v=1:a=0,setpts=N/{fps}/TB[outv]")
    nice = shutil.which("nice")
    command = ([nice, "-n", "10"] if nice else []) + [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-filter_threads", "1",
        "-filter_complex_threads", "1", "-i", str(source), "-filter_complex", ";".join(filters),
        "-map", "[outv]", "-an", "-r", f"{fps:.8f}",
    ]
    if "depth" in stream_name.casefold() or pixel_format.startswith("gray") or destination.suffix.casefold() == ".mkv":
        command += ["-c:v", "ffv1"]
    else:
        command += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
        if destination.suffix.casefold() == ".mp4":
            command += ["-movflags", "+faststart"]
    command += ["-threads", str(max(1, threads)), str(destination)]
    try:
        subprocess.run(command, check=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkbenchError(f"FFmpeg 剪切失败: {stream_name}") from exc


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _kept_span_shifts(source: Path, keep: set[int], fps: float) -> tuple[dict[int, float], float | None]:
    wall_times: list[float | None] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("_type"), str):
                continue
            wall = row.get("t_wall")
            wall_times.append(float(wall) if _finite_number(wall) else None)

    kept_indices = sorted(index for index in keep if 0 <= index < len(wall_times))
    if not kept_indices:
        return {}, None
    spans: list[tuple[int, int]] = []
    start = previous = kept_indices[0]
    for index in kept_indices[1:]:
        if index != previous + 1:
            spans.append((start, previous + 1))
            start = index
        previous = index
    spans.append((start, previous + 1))

    def elapsed(start_index: int, end_index: int) -> float:
        start_wall, end_wall = wall_times[start_index], wall_times[end_index]
        if start_wall is not None and end_wall is not None:
            duration = end_wall - start_wall
            if math.isfinite(duration) and duration >= 0:
                return duration
        return (end_index - start_index) / fps

    shifts: dict[int, float] = {}
    cumulative_shift = elapsed(0, spans[0][0])
    for span_index, (start, end) in enumerate(spans):
        if span_index:
            previous_end = spans[span_index - 1][1]
            cumulative_shift += elapsed(previous_end, start)
        for frame_index in range(start, end):
            shifts[frame_index] = cumulative_shift

    first_wall = wall_times[spans[0][0]]
    first_output_wall = first_wall - shifts[spans[0][0]] if first_wall is not None else None
    return shifts, first_output_wall


def _shift_record_times(row: dict[str, Any], shift_seconds: float,
                        first_output_wall: float | None) -> dict[str, Any]:
    def transform(value: Any, key: str | None = None, parent: str | None = None) -> Any:
        if isinstance(value, dict):
            return {child_key: transform(child, child_key, key) for child_key, child in value.items()}
        if isinstance(value, list):
            return [transform(child, key, parent) for child in value]
        if not _finite_number(value) or key is None or key.endswith("_age_ms"):
            return value
        if key == "t_ns":
            return int(value) - int(round(shift_seconds * 1_000_000_000))
        if key == "ts" and parent == "control":
            return float(value) - shift_seconds * 1000.0
        if key in {"t_wall", "t_monotonic", "t_intended", "session_start_t", "timestamp", "stamp"} or key.endswith("_t"):
            if float(value) == 0.0 and (key.endswith("_t") or key in {"timestamp", "stamp"}):
                return value
            return float(value) - shift_seconds
        return value

    transformed = transform(row)
    if first_output_wall is not None:
        transformed["session_start_t"] = first_output_wall
    monotonic = transformed.get("t_monotonic")
    intended = transformed.get("t_intended")
    if _finite_number(monotonic) and _finite_number(intended):
        transformed["t_jitter_ms"] = (float(monotonic) - float(intended)) * 1000.0
    output_wall = transformed.get("t_wall")
    topics = transformed.get("topics_t")
    if _finite_number(output_wall) and isinstance(topics, dict):
        for key in list(topics):
            if not key.endswith("_age_ms"):
                continue
            timestamp = topics.get(f"{key[:-7]}_t")
            if _finite_number(timestamp):
                topics[key] = (float(output_wall) - float(timestamp)) * 1000.0
    return transformed


def rewrite_manifest(source: Path, destination: Path, keep: set[int], fps: float,
                     audit: dict[str, Any]) -> int:
    source_index = 0
    output_index = 0
    shifts, first_output_wall = _kept_span_shifts(source, keep, fps)
    first_shift = shifts[min(shifts)] if shifts else 0.0
    last_shift = shifts[max(shifts)] if shifts else first_shift
    with source.open("r", encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("_type"), str):
                row = _shift_record_times(
                    row,
                    first_shift if row.get("_type") == "session_header" else last_shift,
                    first_output_wall,
                )
                if row.get("_type") == "session_header":
                    if first_output_wall is not None:
                        row["t_wall"] = first_output_wall
                    row["lightworkbench"] = audit
                json.dump(row, dst, ensure_ascii=False, separators=(",", ":"))
                dst.write("\n")
                continue
            if source_index in keep:
                row = _shift_record_times(row, shifts.get(source_index, 0.0), first_output_wall)
                row["frame_idx"] = output_index
                videos = row.get("videos")
                if isinstance(videos, dict):
                    for entry in videos.values():
                        if isinstance(entry, dict):
                            entry["frame_id"] = output_index
                            entry["is_repeat"] = False
                            entry["frames_dropped"] = 0
                json.dump(row, dst, ensure_ascii=False, separators=(",", ":"))
                dst.write("\n")
                output_index += 1
            source_index += 1
    return output_index


def normalize_manifest_frame_ids(path: Path) -> int:
    """Normalize Episode-local frame indices without changing other manifest values."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    frame_index = 0
    try:
        with path.open("r", encoding="utf-8", errors="strict") as source, os.fdopen(
            descriptor, "w", encoding="utf-8"
        ) as destination:
            descriptor = -1
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise WorkbenchError("manifest 行不是对象")
                if not isinstance(row.get("_type"), str):
                    row["frame_idx"] = frame_index
                    videos = row.get("videos")
                    if isinstance(videos, dict):
                        for entry in videos.values():
                            if isinstance(entry, dict):
                                entry["frame_id"] = frame_index
                    frame_index += 1
                json.dump(row, destination, ensure_ascii=False, separators=(",", ":"))
                destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkbenchError("no_trim manifest 帧编号标准化失败") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return frame_index


def output_fingerprint(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*")
        if candidate.is_file() and candidate.relative_to(path).as_posix() not in FINGERPRINT_EXCLUDES
    ):
        stat = item.stat()
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()

@dataclass
class OperationState:
    id: str
    episode: str
    mode: str
    request: dict[str, Any] = field(repr=False)
    submitted_queue_position: int
    target_key: str = field(repr=False)
    submitted_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    ffmpeg_slots: int = 0
    queue_position: int | None = None
    status: str = "queued"
    progress: float = 0.0
    message: str = "等待执行"
    result: dict[str, Any] | None = None
    error: str | None = None
    csv_row: dict[str, Any] | None = None
    version: int = 0
    condition: threading.Condition = field(default_factory=threading.Condition)

    def event(self) -> dict[str, Any]:
        value = {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "queuePosition": self.queue_position,
            "episode": self.episode,
            "mode": self.mode,
            "submittedAt": self.submitted_at,
            "startedAt": self.started_at,
            "ffmpegSlots": self.ffmpeg_slots,
        }
        if self.result is not None:
            value["result"] = self.result
        if self.error:
            value["error"] = self.error
        return value


class OperationManager:
    MAX_PENDING = 256
    TERMINAL_HISTORY = 100
    DEFAULT_CONCURRENCY = 2
    MAX_CONCURRENCY = 4

    def __init__(self, episodes: EpisodeService):
        self.episodes = episodes
        self._states: dict[str, OperationState] = {}
        self._queue: deque[str] = deque()
        self._running: set[str] = set()
        self._active_targets: set[str] = set()
        self._terminal: deque[str] = deque()
        self._concurrency = self.DEFAULT_CONCURRENCY
        self._condition = threading.Condition(threading.RLock())
        self._snapshot_version = 0
        self._csv_lock = threading.Lock()
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher, daemon=True, name="operation-dispatcher"
        )
        self._dispatcher_thread.start()

    def get(self, operation_id: str) -> OperationState:
        with self._condition:
            state = self._states.get(operation_id)
        if state is None:
            raise WorkbenchError("操作不存在")
        return state

    def settings(self) -> dict[str, int]:
        with self._condition:
            concurrency = self._concurrency
        return {
            "concurrency": concurrency,
            "minConcurrency": 1,
            "maxConcurrency": self.MAX_CONCURRENCY,
            "recommendedConcurrency": self.DEFAULT_CONCURRENCY,
            "ffmpegSlots": int(RESOURCES["ffmpegSlots"]),
        }

    def set_concurrency(self, value: int) -> dict[str, int]:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= self.MAX_CONCURRENCY:
            raise WorkbenchError("并发任务数必须在 1–4 之间")
        with self._condition:
            self._concurrency = value
            self._condition.notify_all()
        return self.settings()

    def snapshot(self) -> tuple[int, dict[str, list[dict[str, Any]]]]:
        with self._condition:
            queued = [self._states[item].event() for item in self._queue]
            running = sorted(
                (self._states[item].event() for item in self._running),
                key=lambda item: item["submittedAt"],
            )
            completed = [
                self._states[item].event()
                for item in reversed(self._terminal)
                if item in self._states
            ]
            return self._snapshot_version, {
                "queued": queued,
                "running": running,
                "completed": completed,
            }

    def list_operations(self) -> dict[str, list[dict[str, Any]]]:
        return self.snapshot()[1]

    def _update(self, state: OperationState, progress: float, message: str, **changes: Any) -> None:
        with self._condition:
            with state.condition:
                state.progress = progress
                state.message = message
                for key, value in changes.items():
                    setattr(state, key, value)
                state.version += 1
                state.condition.notify_all()
            self._snapshot_version += 1
            self._condition.notify_all()

    def _refresh_queue_positions_locked(self) -> None:
        for position, operation_id in enumerate(self._queue, 1):
            state = self._states[operation_id]
            if state.queue_position != position:
                self._update(
                    state,
                    state.progress,
                    f"队列中，第 {position} 位",
                    queue_position=position,
                )

    def _slot_quota_locked(self) -> int:
        return max(1, int(RESOURCES["ffmpegSlots"]) // self._concurrency)

    def _can_start_locked(self) -> bool:
        if not self._queue or len(self._running) >= self._concurrency:
            return False
        quota = self._slot_quota_locked()
        used = sum(self._states[item].ffmpeg_slots for item in self._running)
        return used + quota <= int(RESOURCES["ffmpegSlots"])

    def _dispatcher(self) -> None:
        while True:
            with self._condition:
                while not self._can_start_locked():
                    self._condition.wait()
                operation_id = self._queue.popleft()
                state = self._states[operation_id]
                state.ffmpeg_slots = self._slot_quota_locked()
                state.started_at = utc_now()
                state.queue_position = None
                self._running.add(operation_id)
                self._update(
                    state,
                    0.0,
                    "开始执行",
                    status="running",
                    started_at=state.started_at,
                    queue_position=None,
                    ffmpeg_slots=state.ffmpeg_slots,
                )
                self._refresh_queue_positions_locked()
            thread = threading.Thread(
                target=self._guarded_run,
                args=(state,),
                daemon=True,
                name=f"operation-{state.id[:8]}",
            )
            thread.start()

    def create(self, request: dict[str, Any]) -> OperationState:
        mode = request.get("mode")
        if mode not in {"trim", "no_trim"}:
            raise WorkbenchError("mode 必须是 trim 或 no_trim")
        if not str(request.get("operator") or "").strip():
            raise WorkbenchError("操作人员不能为空")
        if not str(request.get("sourceToken") or "").strip():
            raise WorkbenchError("sourceToken 不能为空")

        root = normalize_root(str(request.get("sourceRoot") or ""))
        relative = str(request.get("episode") or "")
        resolve_episode(root, relative)
        output_value = str(request.get("outputRoot") or "").strip()
        if not output_value:
            raise WorkbenchError("cleaned 输出根目录不能为空")
        output_root = Path(output_value).expanduser().resolve()
        if paths_overlap(root, output_root):
            raise WorkbenchError("cleaned 输出根目录不得与源目录重叠")
        destination = (output_root / relative).resolve()
        try:
            destination.relative_to(output_root)
        except ValueError as exc:
            raise WorkbenchError("输出 Episode 路径越界") from exc
        if destination.exists() and not bool(request.get("overwrite")):
            raise ConflictError(f"目标已存在，需要确认覆盖: {destination}")

        target_key = str(destination)
        with self._condition:
            if target_key in self._active_targets:
                raise ConflictError("相同输出 Episode 已在队列中或正在执行")
            if len(self._queue) >= self.MAX_PENDING:
                raise QueueFullError("任务队列已满（最多缓存 256 个待执行任务）")
            state = OperationState(
                id=uuid.uuid4().hex,
                episode=relative,
                mode=str(mode),
                request=dict(request),
                target_key=target_key,
                submitted_queue_position=len(self._queue) + 1,
                queue_position=len(self._queue) + 1,
            )
            state.message = f"队列中，第 {state.queue_position} 位"
            self._states[state.id] = state
            self._queue.append(state.id)
            self._active_targets.add(target_key)
            self._snapshot_version += 1
            self._condition.notify_all()
        return state

    def _guarded_run(self, state: OperationState) -> None:
        try:
            self._run(state, state.request)
        except Exception as exc:
            self._update(state, state.progress, "操作失败", status="failed", error=str(exc))
        finally:
            with self._condition:
                self._running.discard(state.id)
                self._active_targets.discard(state.target_key)
                if len(self._terminal) >= self.TERMINAL_HISTORY:
                    expired = self._terminal.popleft()
                    self._states.pop(expired, None)
                self._terminal.append(state.id)
                self._refresh_queue_positions_locked()
                self._snapshot_version += 1
                self._condition.notify_all()

    def _run(self, state: OperationState, request: dict[str, Any]) -> None:
        mode = request.get("mode")
        if mode not in {"trim", "no_trim"}:
            raise WorkbenchError("mode 必须是 trim 或 no_trim")
        operator = str(request.get("operator") or "").strip()
        if not operator:
            raise WorkbenchError("操作人员不能为空")
        root = normalize_root(str(request.get("sourceRoot") or ""))
        output_root = normalize_root(str(request.get("outputRoot") or ""), create=True)
        if paths_overlap(root, output_root):
            raise WorkbenchError("cleaned 输出根目录不得与源目录重叠")
        relative = str(request.get("episode") or "")
        episode = resolve_episode(root, relative)
        if any(item.is_symlink() for item in episode.rglob("*")):
            raise WorkbenchError("源 Episode 内含符号链接，拒绝复制或剪切")
        self._update(state, 0.02, "正在严格校验源数据", status="running")
        detail = self.episodes.detail(str(root), relative)
        if not detail["valid"]:
            raise WorkbenchError("源 Episode 校验失败: " + "; ".join(detail["issues"][:4]))
        expected = str(request.get("sourceToken") or "")
        if not expected or detail["sourceToken"] != expected:
            raise ConflictError("源数据已变化，请重新打开 Episode")
        removed = normalize_ranges(request.get("ranges") or [], detail["frameCount"]) if mode == "trim" else []
        spans = kept_spans(removed, detail["frameCount"])
        output_frames = sum(end - start for start, end in spans)
        if output_frames < 2:
            raise WorkbenchError("剪切后必须至少保留两帧")

        destination = output_root / relative
        overwrite = bool(request.get("overwrite"))
        if destination.exists() and not overwrite:
            raise ConflictError(f"目标已存在，需要确认覆盖: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_root = output_root.parent / f".{output_root.name}.lightworkbench-staging-{state.id}"
        staging = staging_root / relative
        staging.mkdir(parents=True, exist_ok=False)
        overwritten_id = ""
        revision = 1
        if destination.exists():
            try:
                old_info = json.loads((destination / "CUT_INFO.json").read_text(encoding="utf-8"))
                overwritten_id = str(old_info.get("operationId") or "")
                revision = int(old_info.get("revision") or 0) + 1
            except (OSError, ValueError, json.JSONDecodeError):
                revision = 1

        audit = {
            "operationId": state.id, "revision": revision, "overwrittenOperationId": overwritten_id or None,
            "mode": mode, "sourceRoot": str(root), "episode": relative, "operator": operator,
            "removedRanges": [list(item) for item in removed], "sourceFrames": detail["frameCount"],
            "outputFrames": output_frames, "sourceToken": expected, "completedAtUtc": None,
            "fingerprintVersion": FINGERPRINT_VERSION,
            "fingerprintExcludes": list(FINGERPRINT_EXCLUDES),
            "timestampRewriteVersion": TIMESTAMP_REWRITE_VERSION,
        }
        try:
            if mode == "no_trim":
                self._update(state, 0.18, "正在复制 Episode")
                shutil.rmtree(staging)
                shutil.copytree(episode, staging, symlinks=False)
                written = normalize_manifest_frame_ids(staging / "manifest.jsonl")
                if written != output_frames:
                    raise WorkbenchError("no_trim manifest 输出帧数不一致")
                audit["manifestNormalizationVersion"] = MANIFEST_NORMALIZATION_VERSION
                audit["normalizedFields"] = ["frame_idx", "videos.*.frame_id"]
            else:
                self._update(state, 0.12, "正在并行剪切全部视频")
                streams = detail["streams"]
                workers = min(len(streams), state.ffmpeg_slots) or 1
                threads_each = max(1, state.ffmpeg_slots // workers)
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ffmpeg") as pool:
                    futures = {}
                    for stream in streams:
                        src = resolve_video(episode, stream["path"])
                        dst = staging / stream["path"]
                        futures[pool.submit(trim_video, src, dst, spans, detail["fps"], stream["name"], stream.get("pixelFormat", ""), threads_each)] = stream["name"]
                    completed = 0
                    for future in as_completed(futures):
                        future.result()
                        completed += 1
                        self._update(state, 0.12 + 0.5 * completed / len(streams), f"已剪切 {futures[future]}")
                for item in episode.iterdir():
                    if item.name in {"videos", "manifest.jsonl", "CUT_INFO.json"}:
                        continue
                    target = staging / item.name
                    shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
                keep = {index for start, end in spans for index in range(start, end)}
                written = rewrite_manifest(episode / "manifest.jsonl", staging / "manifest.jsonl", keep, detail["fps"], audit)
                if written != output_frames:
                    raise WorkbenchError("manifest 输出帧数不一致")

            self._update(state, 0.68, "正在确认源数据状态")
            current = source_token(episode, read_manifest(episode).stream_paths)
            if current != expected:
                raise ConflictError("处理期间源数据发生变化，未发布输出")

            completed_at = utc_now()
            audit["completedAtUtc"] = completed_at
            audit["outputFingerprint"] = output_fingerprint(staging)
            (staging / "CUT_INFO.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._update(state, 0.84, "正在执行转换预检")
            preflight = validate_episode(staging_root, staging, require_source=False)
            if not preflight.valid:
                reasons = "; ".join(preflight.reasons[:8])
                raise WorkbenchError(f"转换预检未通过: {reasons}")
            self._update(state, 0.88, "正在原子发布")
            backup: Path | None = None
            backup_root: Path | None = None
            if destination.exists():
                backup_root = output_root / ".lightworkbench-backups" / state.id
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except Exception:
                if backup is not None and backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            if backup_root is not None:
                shutil.rmtree(backup_root, ignore_errors=True)

            row = {
                "operation_id": state.id, "revision": revision, "overwritten_operation_id": overwritten_id,
                "source_root": str(root), "episode": relative, "operator": operator, "mode": mode,
                "removed_ranges": json.dumps([list(item) for item in removed], separators=(",", ":")),
                "source_frames": detail["frameCount"], "output_frames": output_frames,
                "source_token": expected, "output_fingerprint": audit["outputFingerprint"],
                "output_path": str(destination), "completed_at_utc": completed_at,
            }
            result = {"outputPath": str(destination), "outputFrames": output_frames, "revision": revision, "csvRecorded": True}
            state.csv_row = row
            try:
                self._append_csv(output_root, row)
                self._update(state, 1.0, "已完成", status="completed", result=result)
            except OSError as exc:
                result["csvRecorded"] = False
                result["csvError"] = str(exc)
                self._update(state, 1.0, "输出已发布，表格记录失败", status="completed_csv_failed", result=result)
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)

    def _append_csv(self, output_root: Path, row: dict[str, Any]) -> None:
        csv_path = output_root / CSV_NAME
        with self._csv_lock:
            exists = csv_path.is_file() and csv_path.stat().st_size > 0
            with csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())

    def retry_csv(self, operation_id: str) -> OperationState:
        state = self.get(operation_id)
        if state.status != "completed_csv_failed" or not state.csv_row or not state.result:
            raise WorkbenchError("该操作没有可重试的表格记录")
        output_root = Path(state.csv_row["output_path"])
        for _ in Path(state.csv_row["episode"]).parts:
            output_root = output_root.parent
        self._append_csv(output_root, state.csv_row)
        result = {**state.result, "csvRecorded": True}
        result.pop("csvError", None)
        self._update(state, 1.0, "已补写表格记录", status="completed", result=result)
        return state
