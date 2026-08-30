"""Incremental conversion of validated cleaned episodes to LeRobot v3.

The module deliberately has no heavy imports at import time.  NumPy, PyAV,
OpenCV and LeRobot are loaded only when source videos are probed or a dataset
is written.  This keeps the workbench and its preflight CLI usable without the
optional conversion environment installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REQUIRED_VIDEO_STREAMS = ("rgbd_head_color", "hand_left", "hand_right")
CONVERSION_STATE_FILENAME = "conversion_state.json"
STATE_VERSION = 2
CONVERSION_SCHEMA_VERSION = 5
ACTION_SCHEMA = "eef_delta_gripper_commands_v1"

JOINT_NAMES = [
    "Joint_Ankle", "Joint_Knee", "Joint_Waist_Pitch", "Joint_Waist_Yaw",
    "Joint_Left_Shoulder_Inner", "Joint_Left_Shoulder_Outer",
    "Joint_Left_UpperArm", "Joint_Left_Elbow", "Joint_Left_Forearm",
    "Joint_Left_Wrist_Upper", "Joint_Left_Wrist_Lower",
    "Joint_Right_Shoulder_Inner", "Joint_Right_Shoulder_Outer",
    "Joint_Right_UpperArm", "Joint_Right_Elbow", "Joint_Right_Forearm",
    "Joint_Right_Wrist_Upper", "Joint_Right_Wrist_Lower",
    "Joint_Left_Gripper", "Joint_Right_Gripper", "Joint_Neck_Yaw",
    "Joint_Neck_Pitch", "Joint_Neck_Roll",
]

EEF_NAMES = [
    f"{part}.{axis}"
    for part in (
        "left.position", "left.quaternion", "right.position",
        "right.quaternion", "head.position", "head.quaternion",
    )
    for axis in (("x", "y", "z") if part.endswith("position") else ("x", "y", "z", "w"))
]
ARM_EEF_NAMES = EEF_NAMES[:14]
OBSERVATION_STATE_NAMES = (
    [f"joint_position.{name}" for name in JOINT_NAMES]
    + [f"current_eef_pose.{name}" for name in ARM_EEF_NAMES]
    + ["current_height_z"]
)
ACTION_NAMES = [
    f"{side}.{name}"
    for side in ("left", "right")
    for name in (
        "delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz",
        "gripper_speed", "gripper_force", "gripper_speed_valid",
        "gripper_force_valid",
    )
]


class OptionalDependencyError(RuntimeError):
    """Raised when conversion-only packages are not installed."""


class StateConflictError(ValueError):
    """Raised when appending would mix incompatible or revised sources."""


@dataclass(frozen=True, slots=True)
class VideoSource:
    stream: str
    path: Path
    width: int
    height: int
    fps: float
    frames: int
    is_depth: bool


@dataclass(frozen=True, slots=True)
class SourceEpisode:
    path: Path
    source_episode_id: int
    task: str
    fps: int
    header: dict[str, Any]
    records: list[dict[str, Any]]
    videos: dict[str, VideoSource]


@dataclass(frozen=True, slots=True)
class ConverterConfig:
    video_codec: str = "libsvtav1"
    video_crf: int = 23
    encoder_preset: str | int = 8
    encoder_threads: int = 1
    encoder_queue_maxsize: int = 30
    video_encoding_mode: str = "sequential"

    def __post_init__(self) -> None:
        if self.video_codec not in {"h264", "libsvtav1"}:
            raise ValueError(f"unsupported video codec: {self.video_codec}")
        if not 0 <= self.video_crf <= 63:
            raise ValueError("video_crf must be between 0 and 63")
        if self.encoder_threads <= 0 or self.encoder_queue_maxsize <= 0:
            raise ValueError("encoder thread and queue settings must be positive")
        if self.video_encoding_mode not in {"sequential", "streaming"}:
            raise ValueError("video_encoding_mode must be sequential or streaming")

    def state_value(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "schema_version": CONVERSION_SCHEMA_VERSION,
                "action_schema": ACTION_SCHEMA,
                "action_dim": len(ACTION_NAMES),
                "action_names": ACTION_NAMES,
                "required_video_streams": list(REQUIRED_VIDEO_STREAMS),
                "video_stream_policy": "all_active_manifest_streams",
            }
        )
        return value


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output_root: Path
    created: bool
    existing_episode_indices: tuple[int, ...]
    appended_episode_indices: tuple[int, ...]
    recovered_episode_indices: tuple[int, ...]
    failed: tuple[dict[str, str], ...]
    state: dict[str, Any]


VideoProbe = Callable[[Path, str], VideoSource]


def video_key(stream: str) -> str:
    return f"observation.images.{stream}"


def is_depth_stream(stream: str) -> bool:
    return stream.endswith("_depth") or stream == "rgbd_head_depth"


def ordered_video_streams(streams: Mapping[str, Any] | Iterable[str]) -> list[str]:
    names = set(streams)
    return [*REQUIRED_VIDEO_STREAMS, *sorted(names - set(REQUIRED_VIDEO_STREAMS))]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number (got NaN or infinity)")
    return result


def _finite_vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label}: expected {size} finite numeric values")
    return [_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _normalize_quaternion(value: Sequence[float], label: str) -> list[float]:
    quaternion = _finite_vector(value, 4, label)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm < 1e-12:
        raise ValueError(f"{label}: zero-length quaternion")
    return [item / norm for item in quaternion]


def _quat_conjugate(q: Sequence[float]) -> list[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def _quat_multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_rotate(q: Sequence[float], vector: Sequence[float]) -> list[float]:
    qx, qy, qz, qw = _normalize_quaternion(q, "rotation quaternion")
    vx, vy, vz = _finite_vector(vector, 3, "rotation vector input")
    cx, cy, cz = qy * vz - qz * vy, qz * vx - qx * vz, qx * vy - qy * vx
    dx, dy, dz = qy * cz - qz * cy, qz * cx - qx * cz, qx * cy - qy * cx
    return [vx + 2 * (qw * cx + dx), vy + 2 * (qw * cy + dy), vz + 2 * (qw * cz + dz)]


def _quat_to_rotvec(value: Sequence[float]) -> list[float]:
    q = _normalize_quaternion(value, "relative quaternion")
    if q[3] < 0:
        q = [-item for item in q]
    sin_half = math.sqrt(sum(item * item for item in q[:3]))
    if sin_half < 1e-12:
        return [2 * item for item in q[:3]]
    scale = 2 * math.atan2(sin_half, q[3]) / sin_half
    return [item * scale for item in q[:3]]


def relative_pose_delta(current_pose: Sequence[float], next_pose: Sequence[float]) -> list[float]:
    current = _finite_vector(current_pose, 7, "current EEF pose")
    following = _finite_vector(next_pose, 7, "next EEF pose")
    current_q = _normalize_quaternion(current[3:], "current EEF quaternion")
    next_q = _normalize_quaternion(following[3:], "next EEF quaternion")
    local_translation = _quat_rotate(
        _quat_conjugate(current_q),
        [following[index] - current[index] for index in range(3)],
    )
    local_rotation = _quat_to_rotvec(_quat_multiply(_quat_conjugate(current_q), next_q))
    return [*local_translation, *local_rotation]


def _arm_poses(record: Mapping[str, Any], label: str) -> list[list[float]]:
    root = record.get("current_eef_pose")
    if not isinstance(root, Mapping):
        raise ValueError(f"{label}: missing current_eef_pose")
    result: list[list[float]] = []
    for side in ("left", "right"):
        pose = root.get(f"{side}_eef_pose")
        if not isinstance(pose, Mapping):
            raise ValueError(f"{label}: missing current_eef_pose.{side}_eef_pose")
        result.append(
            [
                *_finite_vector(pose.get("position"), 3, f"{label}:{side}.position"),
                *_normalize_quaternion(pose.get("rotation"), f"{label}:{side}.rotation"),
            ]
        )
    return result


def _command(record: Mapping[str, Any], key: str, label: str) -> tuple[float, float]:
    control = record.get("control")
    if not isinstance(control, Mapping):
        return 0.0, 0.0
    flat_key = f"commands.{key}"
    if flat_key in control:
        return _finite_number(control[flat_key], f"{label}:control.{flat_key}"), 1.0
    commands = control.get("commands")
    if isinstance(commands, Mapping) and key in commands:
        return _finite_number(commands[key], f"{label}:control.commands.{key}"), 1.0
    return 0.0, 0.0


def build_action(
    current_record: Mapping[str, Any], next_record: Mapping[str, Any], label: str = "frame"
) -> list[float]:
    """Build the fixed 20-D action from frame t and the EEF pose at t+1."""

    current_poses = _arm_poses(current_record, f"{label}:current")
    next_poses = _arm_poses(next_record, f"{label}:next")
    action: list[float] = []
    for side, current_pose, next_pose in zip(("LEFT", "RIGHT"), current_poses, next_poses):
        speed, speed_valid = _command(current_record, f"SET_{side}_GRIPPER_SPEED", label)
        force, force_valid = _command(current_record, f"SET_{side}_FORCE", label)
        action.extend(relative_pose_delta(current_pose, next_pose))
        action.extend((speed, force, speed_valid, force_valid))
    assert len(action) == len(ACTION_NAMES)
    return action


def _flatten_eef(record: Mapping[str, Any], key: str, label: str) -> list[float]:
    root = record.get(key)
    if not isinstance(root, Mapping):
        raise ValueError(f"{label}: missing {key}")
    values: list[float] = []
    for pose_name in ("left_eef_pose", "right_eef_pose", "head_pose"):
        pose = root.get(pose_name)
        if not isinstance(pose, Mapping):
            raise ValueError(f"{label}: missing {key}.{pose_name}")
        values.extend(_finite_vector(pose.get("position"), 3, f"{label}:{key}.{pose_name}.position"))
        values.extend(_normalize_quaternion(pose.get("rotation"), f"{label}:{key}.{pose_name}.rotation"))
    return values


def load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if len(rows) < 2 or rows[0].get("_type") != "session_header":
        raise ValueError(f"{path}: missing session header or records")
    for row in rows[1:]:
        if row.get("_type") == "session_footer" and row.get("aborted") is True:
            raise ValueError(f"{path}: aborted session")
    records = [row for row in rows[1:] if not isinstance(row.get("_type"), str)]
    if not records:
        raise ValueError(f"{path}: no data records")
    return rows[0], records


def _default_video_probe(path: Path, stream: str) -> VideoSource:
    try:
        import cv2
    except ImportError as exc:
        raise OptionalDependencyError(
            "Video validation requires the optional LeRobot conversion dependencies (opencv-python)."
        ) from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise ValueError(f"invalid video metadata for {path}")
    return VideoSource(stream, path, width, height, fps, frames, is_depth_stream(stream))


def _resolve_video_path(episode: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"absolute video path is not allowed: {relative}")
    result = (episode / relative).resolve()
    try:
        result.relative_to(episode.resolve())
    except ValueError as exc:
        raise ValueError(f"video path escapes episode directory: {relative}") from exc
    return result


def load_source_episode(
    path: Path,
    task: str | None = None,
    *,
    video_probe: VideoProbe | None = None,
) -> SourceEpisode:
    """Parse one cleaned episode.  CUT_INFO validation intentionally belongs to the CLI layer."""

    path = path.expanduser().resolve()
    manifest = path / "manifest.jsonl"
    metadata_path = path / "task_meta.json"
    if not manifest.is_file() or not metadata_path.is_file():
        raise ValueError(f"{path}: manifest.jsonl or task_meta.json is missing")
    header, records = load_jsonl(manifest)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{metadata_path}: invalid task metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: task metadata must be an object")

    raw_fps = header.get("fps_target", records[0].get("fps_target"))
    fps_value = _finite_number(raw_fps, f"{manifest}:fps_target")
    if fps_value <= 0 or not math.isclose(fps_value, round(fps_value)):
        raise ValueError(f"{manifest}: LeRobot requires a positive integer fps")
    fps = int(round(fps_value))
    nested_task = records[0].get("task")
    inferred_task = (
        header.get("task_description")
        or metadata.get("task_description")
        or metadata.get("description")
        or (nested_task.get("task_title") if isinstance(nested_task, Mapping) else None)
    )
    stored_task = task if task is not None else inferred_task
    if not isinstance(stored_task, str) or not stored_task.strip():
        raise ValueError(f"{path}: no stored task text")

    active_streams: set[str] = set()
    for index, record in enumerate(records):
        label = f"{manifest}:frame {index}"
        if isinstance(record.get("frame_idx"), bool) or record.get("frame_idx") != index:
            raise ValueError(f"{label}: non-contiguous frame_idx={record.get('frame_idx')!r}")
        joints = record.get("joints")
        if not isinstance(joints, Mapping):
            raise ValueError(f"{label}: missing joints")
        for field in ("position", "velocity", "torque"):
            _finite_vector(joints.get(field), len(JOINT_NAMES), f"{label}:joints.{field}")
        _flatten_eef(record, "current_eef_pose", label)
        _flatten_eef(record, "target_eef_pose", label)
        for field in ("current_height_z", "target_height_z"):
            value = record.get(field)
            if not isinstance(value, Mapping):
                raise ValueError(f"{label}: missing {field}")
            _finite_number(value.get("height_z"), f"{label}:{field}.height_z")
        robot_state = record.get("robot_state")
        if (
            not isinstance(robot_state, Mapping)
            or isinstance(robot_state.get("state"), bool)
            or not isinstance(robot_state.get("state"), int)
        ):
            raise ValueError(f"{label}: invalid robot_state.state")
        if isinstance(record.get("t_ns"), bool) or not isinstance(record.get("t_ns"), int):
            raise ValueError(f"{label}: invalid t_ns")
        videos = record.get("videos")
        if not isinstance(videos, Mapping):
            raise ValueError(f"{label}: missing videos")
        active_streams.update(
            stream
            for stream, entry in videos.items()
            if isinstance(stream, str)
            and isinstance(entry, Mapping)
            and isinstance(entry.get("path"), str)
            and bool(entry["path"])
        )
        # Commands are optional, but every command that exists must be usable.
        for side in ("LEFT", "RIGHT"):
            _command(record, f"SET_{side}_GRIPPER_SPEED", label)
            _command(record, f"SET_{side}_FORCE", label)

    missing = set(REQUIRED_VIDEO_STREAMS) - active_streams
    if missing:
        raise ValueError(f"{manifest}: missing required video streams {sorted(missing)}")
    probe = video_probe or _default_video_probe
    video_sources: dict[str, VideoSource] = {}
    for stream in ordered_video_streams(active_streams):
        paths: set[str] = set()
        for index, record in enumerate(records):
            entry = record["videos"].get(stream)
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not entry["path"]:
                raise ValueError(f"{manifest}:frame {index}: missing videos.{stream}.path")
            frame_id = entry.get("frame_id")
            if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                raise ValueError(f"{manifest}:frame {index}: invalid videos.{stream}.frame_id")
            paths.add(entry["path"])
        if len(paths) != 1:
            raise ValueError(f"{manifest}: videos.{stream}.path changes within episode")
        video_path = _resolve_video_path(path, paths.pop())
        if not video_path.is_file():
            raise ValueError(f"missing video: {video_path}")
        inspected = probe(video_path, stream)
        video = VideoSource(
            stream, video_path, inspected.width, inspected.height, inspected.fps,
            inspected.frames, is_depth_stream(stream),
        )
        if video.frames != len(records):
            raise ValueError(f"{video.path}: {video.frames} frames but manifest has {len(records)}")
        if not math.isclose(video.fps, fps, rel_tol=0, abs_tol=0.05):
            raise ValueError(f"{video.path}: fps {video.fps} differs from manifest fps {fps}")
        video_sources[stream] = video

    source_id = header.get("episode_id")
    if isinstance(source_id, bool) or not isinstance(source_id, int):
        match = re.fullmatch(r"episode_(\d+)", path.name)
        if not match:
            raise ValueError(f"{path}: cannot determine source episode id")
        source_id = int(match.group(1))
    return SourceEpisode(path, source_id, stored_task.strip(), fps, header, records, video_sources)


def schema_for_episode(episode: SourceEpisode) -> dict[str, Any]:
    return {
        "fps": episode.fps,
        "videos": {
            stream: {
                "key": video_key(stream),
                "width": episode.videos[stream].width,
                "height": episode.videos[stream].height,
                "channels": 1 if episode.videos[stream].is_depth else 3,
                "is_depth": episode.videos[stream].is_depth,
            }
            for stream in ordered_video_streams(episode.videos)
        },
    }


def select_compatible_episodes(
    episodes: Sequence[SourceEpisode],
) -> tuple[list[SourceEpisode], list[tuple[SourceEpisode, str]]]:
    """Choose the largest schema group, resolving ties by the first natural input episode."""

    if not episodes:
        return [], []
    keys = [json.dumps(schema_for_episode(item), sort_keys=True) for item in episodes]
    counts = Counter(keys)
    best_count = max(counts.values())
    best = next(key for key in keys if counts[key] == best_count)
    accepted = [item for item, key in zip(episodes, keys) if key == best]
    skipped = [
        (item, "schema_outlier") for item, key in zip(episodes, keys) if key != best
    ]
    return accepted, skipped


def converted_episode_length(episode: SourceEpisode) -> int:
    if len(episode.records) < 2:
        raise ValueError(f"{episode.path}: action requires at least two source frames")
    return len(episode.records) - 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_episode_signature(episode: SourceEpisode) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    for filename in ("manifest.jsonl", "task_meta.json", "CUT_INFO.json"):
        path = episode.path / filename
        if path.is_file():
            inventory.append({"path": filename, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    for stream in ordered_video_streams(episode.videos):
        path = episode.videos[stream].path
        stat = path.stat()
        inventory.append(
            {
                "path": path.relative_to(episode.path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {
        "algorithm": "sha256(control-content+video-path-size-mtime-v3)",
        "digest": hashlib.sha256(encoded).hexdigest(),
        "files": len(inventory),
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _state_entry(episode: SourceEpisode, index: int) -> dict[str, Any]:
    return {
        "source_episode_name": episode.path.name,
        "source_episode_id": episode.source_episode_id,
        "source_signature": source_episode_signature(episode),
        "lerobot_episode_index": index,
        "source_frames": len(episode.records),
        "output_frames": converted_episode_length(episode),
    }


def load_conversion_state(root: Path) -> dict[str, Any] | None:
    path = root / CONVERSION_STATE_FILENAME
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateConflictError(f"{path}: invalid conversion state: {exc}") from exc
    if not isinstance(state, dict):
        raise StateConflictError(f"{path}: conversion state must be an object")
    if state.get("version") != STATE_VERSION:
        raise StateConflictError(
            f"{path}: incompatible legacy conversion state; the 20-D action schema requires a new output root"
        )
    config = state.get("conversion_config")
    if not isinstance(config, dict) or config.get("action_dim") != len(ACTION_NAMES):
        raise StateConflictError(
            f"{path}: existing output is not the required 20-D action schema; use a new output root"
        )
    if (
        config.get("action_names") != ACTION_NAMES
        or not isinstance(state.get("stored_task"), str) or not state["stored_task"]
        or not isinstance(state.get("episodes"), list)
    ):
        raise StateConflictError(f"{path}: incompatible or malformed conversion state")
    return state


def _write_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(root / CONVERSION_STATE_FILENAME, state)


def new_conversion_state(
    source_task: Path, repo_id: str, config: ConverterConfig, schema: Mapping[str, Any],
    stored_task: str,
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "version": STATE_VERSION,
        "source_task": str(source_task.resolve()),
        "repo_id": repo_id,
        "conversion_config": config.state_value(),
        "schema": dict(schema),
        "stored_task": stored_task,
        "episodes": [],
        "created_at": now,
        "updated_at": now,
    }


def validate_incremental_state(
    state: Mapping[str, Any],
    episodes: Sequence[SourceEpisode],
    *,
    source_task: Path,
    repo_id: str,
    config: ConverterConfig,
) -> list[SourceEpisode]:
    if state.get("repo_id") != repo_id:
        raise StateConflictError("conversion state repo_id differs from requested value")
    if state.get("conversion_config") != config.state_value():
        raise StateConflictError("conversion settings conflict with the existing dataset")
    if Path(str(state.get("source_task", ""))).resolve() != source_task.resolve():
        raise StateConflictError("conversion state belongs to a different source task directory")
    if not episodes:
        if state.get("episodes"):
            raise StateConflictError("all previously converted source episodes are missing")
        return []
    stored_tasks = {episode.task for episode in episodes}
    if len(stored_tasks) != 1 or state.get("stored_task") not in stored_tasks:
        raise StateConflictError("stored task language or text conflicts with the existing dataset")
    if state.get("schema") != schema_for_episode(episodes[0]):
        raise StateConflictError("source video schema conflicts with the existing dataset")

    by_id: dict[int, SourceEpisode] = {}
    for episode in episodes:
        if episode.source_episode_id in by_id:
            raise StateConflictError(f"duplicate source episode id {episode.source_episode_id}")
        by_id[episode.source_episode_id] = episode
    existing_ids: set[int] = set()
    for expected_index, entry in enumerate(state.get("episodes", [])):
        if not isinstance(entry, Mapping) or entry.get("lerobot_episode_index") != expected_index:
            raise StateConflictError("conversion state episode indices are not contiguous")
        source_id = entry.get("source_episode_id")
        if source_id in existing_ids:
            raise StateConflictError(f"conversion state has duplicate source episode id {source_id}")
        existing_ids.add(source_id)
        episode = by_id.get(source_id)
        if episode is None:
            raise StateConflictError(f"previously converted source episode {source_id} is missing")
        if entry.get("source_episode_name") != episode.path.name:
            raise StateConflictError(
                f"source episode {source_id} changed name from {entry.get('source_episode_name')} to {episode.path.name}"
            )
        current = source_episode_signature(episode)
        if entry.get("source_signature", {}).get("digest") != current["digest"]:
            raise StateConflictError(f"{episode.path}: source changed after it was converted")
    return [episode for episode in episodes if episode.source_episode_id not in existing_ids]


def axes_feature(size: int, names: Sequence[str], dtype: str = "float32") -> dict[str, Any]:
    return {"dtype": dtype, "shape": (size,), "names": {"axes": list(names)}}


def build_features(videos: Mapping[str, VideoSource]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for stream in ordered_video_streams(videos):
        video = videos[stream]
        feature: dict[str, Any] = {
            "dtype": "video",
            "shape": (video.height, video.width, 1 if video.is_depth else 3),
            "names": ["height", "width", "channels"],
        }
        if video.is_depth:
            feature["info"] = {"is_depth_map": True, "depth_unit": "mm"}
        features[video_key(stream)] = feature
    features.update(
        {
            "observation.state": axes_feature(len(OBSERVATION_STATE_NAMES), OBSERVATION_STATE_NAMES),
            "observation.joint_velocity": axes_feature(len(JOINT_NAMES), JOINT_NAMES),
            "observation.joint_torque": axes_feature(len(JOINT_NAMES), JOINT_NAMES),
            "observation.robot_state": axes_feature(1, ["state"], "int64"),
            "observation.current_eef_pose": axes_feature(len(EEF_NAMES), EEF_NAMES),
            "observation.target_eef_pose": axes_feature(len(EEF_NAMES), EEF_NAMES),
            "observation.current_height_z": axes_feature(1, ["height_z"]),
            "observation.target_height_z": axes_feature(1, ["height_z"]),
            "source.frame_index": axes_feature(1, ["source_frame_index"], "int64"),
            "source.timestamp_ns": axes_feature(1, ["source_timestamp_ns"], "int64"),
            "source.episode_id": axes_feature(1, ["source_episode_id"], "int64"),
            "action": axes_feature(len(ACTION_NAMES), ACTION_NAMES),
        }
    )
    return features


def _heavy_runtime() -> dict[str, Any]:
    try:
        import av
        import cv2
        import numpy as np
        from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise OptionalDependencyError(
            "LeRobot conversion dependencies are unavailable. Install the optional 'lerobot' dependency group."
        ) from exc
    return {
        "av": av, "cv2": cv2, "np": np, "LeRobotDataset": LeRobotDataset,
        "RGBEncoderConfig": RGBEncoderConfig, "DepthEncoderConfig": DepthEncoderConfig,
    }


class _VideoReader:
    def __init__(self, source: VideoSource, runtime: Mapping[str, Any]):
        self.source = source
        self.runtime = runtime
        self.capture = None
        self.container = None
        self.frames = None
        if source.is_depth:
            self.container = runtime["av"].open(str(source.path))
            self.frames = iter(self.container.decode(video=0))
        else:
            self.capture = runtime["cv2"].VideoCapture(str(source.path))
            if not self.capture.isOpened():
                self.capture.release()
                raise RuntimeError(f"failed to open video: {source.path}")

    def read(self) -> Any | None:
        if self.source.is_depth:
            try:
                return next(self.frames).to_ndarray(format="gray16le")[:, :, None]
            except StopIteration:
                return None
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            return None
        return self.runtime["cv2"].cvtColor(bgr, self.runtime["cv2"].COLOR_BGR2RGB)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.container is not None:
            self.container.close()


def _make_frame(episode: SourceEpisode, frame_index: int, images: Mapping[str, Any], np: Any) -> dict[str, Any]:
    record = episode.records[frame_index]
    joints = record["joints"]
    current_pose = _flatten_eef(record, "current_eef_pose", f"frame {frame_index}")
    target_pose = _flatten_eef(record, "target_eef_pose", f"frame {frame_index}")
    current_height = _finite_number(record["current_height_z"]["height_z"], "current_height_z")
    target_height = _finite_number(record["target_height_z"]["height_z"], "target_height_z")
    state = [*_finite_vector(joints["position"], len(JOINT_NAMES), "joints.position"), *current_pose[:14], current_height]
    return {
        **images,
        "observation.state": np.asarray(state, dtype=np.float32),
        "observation.joint_velocity": np.asarray(joints["velocity"], dtype=np.float32),
        "observation.joint_torque": np.asarray(joints["torque"], dtype=np.float32),
        "observation.robot_state": np.asarray([record["robot_state"]["state"]], dtype=np.int64),
        "observation.current_eef_pose": np.asarray(current_pose, dtype=np.float32),
        "observation.target_eef_pose": np.asarray(target_pose, dtype=np.float32),
        "observation.current_height_z": np.asarray([current_height], dtype=np.float32),
        "observation.target_height_z": np.asarray([target_height], dtype=np.float32),
        "source.frame_index": np.asarray([record["frame_idx"]], dtype=np.int64),
        "source.timestamp_ns": np.asarray([record["t_ns"]], dtype=np.int64),
        "source.episode_id": np.asarray([episode.source_episode_id], dtype=np.int64),
        "action": np.asarray(build_action(record, episode.records[frame_index + 1], f"frame {frame_index}"), dtype=np.float32),
        "task": episode.task,
    }


def _add_frames(dataset: Any, episode: SourceEpisode, runtime: Mapping[str, Any]) -> None:
    readers: dict[str, _VideoReader] = {}
    try:
        readers = {stream: _VideoReader(video, runtime) for stream, video in episode.videos.items()}
        for index in range(converted_episode_length(episode)):
            images: dict[str, Any] = {}
            for stream in ordered_video_streams(readers):
                image = readers[stream].read()
                if image is None:
                    raise RuntimeError(f"{episode.path}: {stream} ended before frame {index}")
                video = episode.videos[stream]
                expected = (video.height, video.width, 1 if video.is_depth else 3)
                if image.shape != expected:
                    raise RuntimeError(f"{episode.path}: {stream} frame {index} shape {image.shape}, expected {expected}")
                images[video_key(stream)] = image
            dataset.add_frame(_make_frame(episode, index, images, runtime["np"]))
        # Consume the terminal source frame and prove there are no extras.
        for stream, reader in readers.items():
            if reader.read() is None:
                raise RuntimeError(f"{episode.path}: {stream} ended before terminal frame")
            if reader.read() is not None:
                raise RuntimeError(f"{episode.path}: {stream} has frames beyond manifest")
    finally:
        for reader in readers.values():
            reader.close()


def _encoders(runtime: Mapping[str, Any], config: ConverterConfig, fps: int) -> tuple[Any, Any]:
    preset: str | int = config.encoder_preset
    if isinstance(preset, str) and preset.isdigit():
        preset = int(preset)
    rgb = runtime["RGBEncoderConfig"](
        vcodec=config.video_codec, pix_fmt="yuv420p", g=fps,
        crf=config.video_crf, preset=preset,
    )
    depth = runtime["DepthEncoderConfig"](
        g=fps,
        extra_options={"x265-params": f"lossless=1:bframes=0:pools={config.encoder_threads}:frame-threads=1"},
    )
    return rgb, depth


def _open_dataset(
    runtime: Mapping[str, Any], root: Path, repo_id: str, episode: SourceEpisode,
    config: ConverterConfig, *, create: bool,
) -> Any:
    rgb, depth = _encoders(runtime, config, episode.fps)
    common = {
        "repo_id": repo_id, "root": root, "video_backend": "pyav",
        "rgb_encoder": rgb, "depth_encoder": depth,
        "streaming_encoding": config.video_encoding_mode == "streaming",
        "encoder_queue_maxsize": config.encoder_queue_maxsize,
        "encoder_threads": config.encoder_threads,
    }
    if create:
        return runtime["LeRobotDataset"].create(
            **common, robot_type="autolife_s1_robot_v2_2", fps=episode.fps,
            features=build_features(episode.videos), use_videos=True,
        )
    return runtime["LeRobotDataset"].resume(**common)


def _dataset_episode_count(runtime: Mapping[str, Any], root: Path, repo_id: str) -> int:
    dataset = runtime["LeRobotDataset"](
        repo_id=repo_id, root=root, video_backend="pyav", return_uint8=True,
    )
    try:
        return int(dataset.meta.total_episodes)
    finally:
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            finalize()


def _verify_episode(runtime: Mapping[str, Any], root: Path, repo_id: str, index: int, episode: SourceEpisode) -> None:
    np = runtime["np"]
    dataset = runtime["LeRobotDataset"](
        repo_id=repo_id, root=root, video_backend="pyav", return_uint8=True,
    )
    try:
        if int(dataset.meta.total_episodes) <= index:
            raise RuntimeError(f"LeRobot episode {index} was not persisted")
        metadata = dataset.meta.episodes[index]
        start, end = int(metadata["dataset_from_index"]), int(metadata["dataset_to_index"])
        if end - start != converted_episode_length(episode):
            raise RuntimeError(f"LeRobot episode {index} frame count differs from source")
        ids = np.asarray(dataset.hf_dataset["source.episode_id"][start:end]).reshape(-1)
        if ids.size != end - start or not np.all(ids == episode.source_episode_id):
            raise RuntimeError(f"LeRobot episode {index} source id readback failed")
        actions = np.asarray(dataset.hf_dataset["action"][start:end])
        expected_actions = np.asarray(
            [build_action(episode.records[position], episode.records[position + 1]) for position in range(end - start)],
            dtype=np.float32,
        )
        if (
            actions.shape != expected_actions.shape
            or not np.isfinite(actions).all()
            or not np.allclose(actions, expected_actions, rtol=1e-6, atol=1e-6)
        ):
            raise RuntimeError(f"LeRobot episode {index} action readback failed")
        expected_rows = [
            _make_frame(episode, position, {}, np) for position in range(end - start)
        ]
        for key in (name for name in expected_rows[0] if name != "task"):
            wanted = np.stack([row[key] for row in expected_rows])
            actual = np.asarray(dataset.hf_dataset[key][start:end])
            if actual.shape != wanted.shape:
                if wanted.shape[-1:] == (1,) and actual.shape == wanted.shape[:-1]:
                    wanted = wanted.reshape(actual.shape)
                else:
                    raise RuntimeError(f"LeRobot episode {index} {key} readback shape failed")
            if np.issubdtype(wanted.dtype, np.floating):
                if not np.isfinite(actual).all() or not np.allclose(actual, wanted, rtol=1e-6, atol=1e-6):
                    raise RuntimeError(f"LeRobot episode {index} {key} numeric readback failed")
            elif not np.array_equal(actual, wanted):
                raise RuntimeError(f"LeRobot episode {index} {key} numeric readback failed")
        # Decode first and last output frames for every video feature.
        for position in sorted({start, end - 1}):
            frame = dataset[position]
            for stream in episode.videos:
                image = frame[video_key(stream)]
                if tuple(image.shape) != (
                    1 if episode.videos[stream].is_depth else 3,
                    episode.videos[stream].height,
                    episode.videos[stream].width,
                ):
                    raise RuntimeError(f"LeRobot episode {index} video {stream} readback failed")
        for stream in episode.videos:
            output_path = root / dataset.meta.get_video_file_path(index, video_key(stream))
            container = runtime["av"].open(str(output_path))
            try:
                decoded_frames = sum(1 for _ in container.decode(video=0))
            finally:
                container.close()
            if decoded_frames != end - start:
                raise RuntimeError(
                    f"LeRobot episode {index} video {stream} has {decoded_frames} frames, expected {end - start}"
                )
    finally:
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            finalize()


def _append_one(
    runtime: Mapping[str, Any], root: Path, repo_id: str, episode: SourceEpisode,
    config: ConverterConfig, *, create: bool,
) -> int:
    dataset = _open_dataset(runtime, root, repo_id, episode, config, create=create)
    index = int(dataset.meta.total_episodes)
    signature_before = source_episode_signature(episode)
    try:
        try:
            _add_frames(dataset, episode, runtime)
            if source_episode_signature(episode)["digest"] != signature_before["digest"]:
                raise RuntimeError("source episode changed while conversion was running")
        except Exception:
            if getattr(dataset, "has_pending_frames", lambda: False)():
                dataset.clear_episode_buffer()
            raise
        dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    _verify_episode(runtime, root, repo_id, index, episode)
    return index


def _recover_uncommitted_state(
    runtime: Mapping[str, Any], root: Path, repo_id: str, state: dict[str, Any],
    episodes: Sequence[SourceEpisode],
) -> list[int]:
    count = _dataset_episode_count(runtime, root, repo_id)
    known = len(state["episodes"])
    if count < known or count > known + 1:
        raise StateConflictError(
            f"LeRobot metadata/state mismatch: dataset={count}, state={known}"
        )
    if count == known:
        return []
    by_id = {episode.source_episode_id: episode for episode in episodes}
    dataset = runtime["LeRobotDataset"](
        repo_id=repo_id, root=root, video_backend="pyav", return_uint8=True,
    )
    try:
        metadata = dataset.meta.episodes[known]
        start = int(metadata["dataset_from_index"])
        values = runtime["np"].asarray(dataset.hf_dataset["source.episode_id"][start:start + 1]).reshape(-1)
        source_id = int(values[0])
    finally:
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            finalize()
    episode = by_id.get(source_id)
    if episode is None:
        raise StateConflictError(f"cannot recover saved episode for missing source id {source_id}")
    _verify_episode(runtime, root, repo_id, known, episode)
    state["episodes"].append(_state_entry(episode, known))
    _write_state(root, state)
    return [known]


def convert_task(
    episodes: Sequence[SourceEpisode],
    output_root: Path,
    repo_id: str,
    *,
    source_task: Path,
    config: ConverterConfig | None = None,
) -> ConversionResult:
    """Create or incrementally append one task-level LeRobot v3 dataset."""

    config = config or ConverterConfig()
    output_root = output_root.expanduser().resolve()
    source_task = source_task.expanduser().resolve()
    episodes = list(episodes)
    if not episodes:
        raise ValueError("at least one compatible source episode is required")
    ids = [item.source_episode_id for item in episodes]
    if len(ids) != len(set(ids)):
        duplicate = next(item for item, count in Counter(ids).items() if count > 1)
        raise StateConflictError(f"duplicate source episode id {duplicate}")
    expected_schema = schema_for_episode(episodes[0])
    for episode in episodes[1:]:
        if schema_for_episode(episode) != expected_schema:
            raise ValueError(f"{episode.path}: source episode is a schema outlier")
    runtime = _heavy_runtime()

    report_only = False
    preserved_report: bytes | None = None
    if output_root.exists() and output_root.is_dir():
        children = list(output_root.iterdir())
        report_only = bool(children) and all(
            item.name == "conversion_report.json" and item.is_file()
            for item in children
        )
        if report_only:
            preserved_report = (output_root / "conversion_report.json").read_bytes()

    if output_root.exists() and not report_only:
        if not output_root.is_dir():
            raise StateConflictError(f"output exists and is not a directory: {output_root}")
        state = load_conversion_state(output_root)
        if state is None:
            raise StateConflictError(
                f"{output_root}: existing output has no 20-D conversion state; use a new output root"
            )
        pending = validate_incremental_state(
            state, episodes, source_task=source_task, repo_id=repo_id, config=config,
        )
        recovered = _recover_uncommitted_state(runtime, output_root, repo_id, state, episodes)
        # Recovery changes which sources still need appending.
        pending = validate_incremental_state(
            state, episodes, source_task=source_task, repo_id=repo_id, config=config,
        )
        existing = tuple(entry["lerobot_episode_index"] for entry in state["episodes"])
        appended: list[int] = []
        failed: list[dict[str, str]] = []
        for episode in pending:
            try:
                index = _append_one(runtime, output_root, repo_id, episode, config, create=False)
                if index != len(state["episodes"]):
                    raise StateConflictError("LeRobot assigned a non-contiguous episode index")
                state["episodes"].append(_state_entry(episode, index))
                _write_state(output_root, state)
                appended.append(index)
            except Exception as exc:
                failed.append({"episode": episode.path.name, "reason": f"{type(exc).__name__}: {exc}"})
                break
        return ConversionResult(
            output_root, False, existing, tuple(appended), tuple(recovered), tuple(failed), state,
        )

    staging = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
    displaced = output_root.with_name(f".{output_root.name}.report-{uuid.uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    state = new_conversion_state(source_task, repo_id, config, expected_schema, episodes[0].task)
    appended: list[int] = []
    failed: list[dict[str, str]] = []
    try:
        for episode in episodes:
            try:
                index = _append_one(runtime, staging, repo_id, episode, config, create=not appended)
                if index != len(state["episodes"]):
                    raise StateConflictError("LeRobot assigned a non-contiguous episode index")
                state["episodes"].append(_state_entry(episode, index))
                _write_state(staging, state)
                appended.append(index)
            except Exception as exc:
                failed.append({"episode": episode.path.name, "reason": f"{type(exc).__name__}: {exc}"})
                break
        if not appended:
            reason = failed[0]["reason"] if failed else "no episode was converted"
            raise RuntimeError(reason)
        if preserved_report is not None:
            (staging / "conversion_report.json").write_bytes(preserved_report)
            os.replace(output_root, displaced)
        try:
            os.replace(staging, output_root)
        except Exception:
            if displaced.exists() and not output_root.exists():
                os.replace(displaced, output_root)
            raise
        if displaced.exists():
            shutil.rmtree(displaced)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ConversionResult(output_root, True, (), tuple(appended), (), tuple(failed), state)


__all__ = [
    "ACTION_NAMES", "ACTION_SCHEMA", "CONVERSION_SCHEMA_VERSION",
    "CONVERSION_STATE_FILENAME", "ConverterConfig", "ConversionResult",
    "OptionalDependencyError", "REQUIRED_VIDEO_STREAMS", "SourceEpisode",
    "StateConflictError", "VideoSource", "atomic_write_json", "build_action",
    "build_features", "convert_task", "converted_episode_length",
    "load_conversion_state", "load_source_episode", "new_conversion_state",
    "ordered_video_streams", "relative_pose_delta", "schema_for_episode",
    "select_compatible_episodes", "source_episode_signature",
    "validate_incremental_state", "video_key",
]
