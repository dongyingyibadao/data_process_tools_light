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
TRAINING_VIDEO_STREAMS = REQUIRED_VIDEO_STREAMS
ACTION_MODES = ("body_joint_eef", "whole_body_joint")
BODY_JOINT_EEF = "body_joint_eef"
WHOLE_BODY_JOINT = "whole_body_joint"
TASK_DATASET_LAYOUT = "task"
MERGED_DATASET_LAYOUT = "merged"
DATASET_LAYOUTS = (TASK_DATASET_LAYOUT, MERGED_DATASET_LAYOUT)
CONVERSION_STATE_FILENAME = "conversion_state.json"
STATE_VERSION = 3
CONVERSION_SCHEMA_VERSION = 6
AUXILIARY_INDEX_PATH = Path("auxiliary/index.parquet")

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
BODY_JOINT_EEF_ACTION_NAMES = [
    *[f"lower_body.{name}" for name in ("Ankle", "Knee", "Waist_Pitch", "Waist_Yaw")],
    *[f"left.{name}" for name in (
        "delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper",
    )],
    *[f"right.{name}" for name in (
        "delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper",
    )],
    *[f"neck.{name}" for name in ("Yaw", "Pitch", "Roll")],
]
WHOLE_BODY_JOINT_ACTION_NAMES = [f"joint_position.{name}" for name in JOINT_NAMES]
ACTION_NAMES_BY_MODE = {
    BODY_JOINT_EEF: BODY_JOINT_EEF_ACTION_NAMES,
    WHOLE_BODY_JOINT: WHOLE_BODY_JOINT_ACTION_NAMES,
}
ACTION_SCHEMAS = {
    BODY_JOINT_EEF: "body_joint_local_eef_delta_v1",
    WHOLE_BODY_JOINT: "whole_body_absolute_joint_v1",
}


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
        if self.video_encoding_mode not in {"sequential", "parallel", "streaming"}:
            raise ValueError("video_encoding_mode must be sequential, parallel, or streaming")

    def state_value(
        self, action_mode: str = WHOLE_BODY_JOINT, shared_video_owner: str | None = None,
    ) -> dict[str, Any]:
        names = action_names(action_mode)
        value = asdict(self)
        value.update(
            {
                "schema_version": CONVERSION_SCHEMA_VERSION,
                "action_mode": action_mode,
                "action_schema": ACTION_SCHEMAS[action_mode],
                "action_dim": len(names),
                "action_names": names,
                "observation_state_dim": len(OBSERVATION_STATE_NAMES),
                "observation_state_names": OBSERVATION_STATE_NAMES,
                "required_video_streams": list(REQUIRED_VIDEO_STREAMS),
                "training_video_streams": list(TRAINING_VIDEO_STREAMS),
                "video_stream_policy": "training_rgb_whitelist_with_auxiliary_index",
                "task_policy": "normalized_english_task_title",
                "terminal_action_policy": "hold_with_zero_eef_delta",
                "shared_video_owner": shared_video_owner,
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


@dataclass(frozen=True, slots=True)
class BundleConversionResult:
    output_root: Path
    action_mode: str
    mode_results: dict[str, ConversionResult]
    ledger: dict[str, Any]
    failed: tuple[dict[str, str], ...]


VideoProbe = Callable[[Path, str], VideoSource]


def action_names(action_mode: str) -> list[str]:
    try:
        return list(ACTION_NAMES_BY_MODE[action_mode])
    except KeyError as exc:
        raise ValueError(f"unsupported action mode: {action_mode}") from exc


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


def build_action(
    current_record: Mapping[str, Any],
    next_record: Mapping[str, Any] | None = None,
    label: str = "frame",
    action_mode: str = BODY_JOINT_EEF,
) -> list[float]:
    """Build one action; a missing next record denotes the terminal hold frame."""

    terminal = next_record is None
    following = current_record if terminal else next_record
    next_joints = following.get("joints")
    if not isinstance(next_joints, Mapping):
        raise ValueError(f"{label}:next: missing joints")
    next_qpos = _finite_vector(next_joints.get("position"), len(JOINT_NAMES), f"{label}:next:joints.position")
    if action_mode == WHOLE_BODY_JOINT:
        return next_qpos
    if action_mode != BODY_JOINT_EEF:
        raise ValueError(f"unsupported action mode: {action_mode}")
    current_poses = _arm_poses(current_record, f"{label}:current")
    next_poses = _arm_poses(following, f"{label}:next")
    left_delta = [0.0] * 6 if terminal else relative_pose_delta(current_poses[0], next_poses[0])
    right_delta = [0.0] * 6 if terminal else relative_pose_delta(current_poses[1], next_poses[1])
    action = [
        *next_qpos[:4],
        *left_delta,
        next_qpos[18],
        *right_delta,
        next_qpos[19],
        *next_qpos[20:23],
    ]
    assert len(action) == len(BODY_JOINT_EEF_ACTION_NAMES)
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
    inferred_title = (
        header.get("task_title")
        or metadata.get("task_title")
        or (nested_task.get("task_title") if isinstance(nested_task, Mapping) else None)
    )
    inferred_task = (
        " ".join(inferred_title.replace("_", " ").split())
        if isinstance(inferred_title, str)
        else None
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
            for stream in TRAINING_VIDEO_STREAMS
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
    return len(episode.records)


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


def _source_relative_path(episode: SourceEpisode, source_root: Path) -> str:
    try:
        relative = episode.path.resolve().relative_to(source_root.resolve())
    except ValueError as exc:
        raise StateConflictError(
            f"{episode.path}: source episode is outside merged source root {source_root}"
        ) from exc
    if not relative.parts:
        raise StateConflictError(f"{episode.path}: merged source episode path is empty")
    return relative.as_posix()


def _episode_identity(
    episode: SourceEpisode, dataset_layout: str, source_root: Path,
) -> int | str:
    if dataset_layout == TASK_DATASET_LAYOUT:
        return episode.source_episode_id
    if dataset_layout == MERGED_DATASET_LAYOUT:
        return _source_relative_path(episode, source_root)
    raise ValueError(f"unsupported dataset layout: {dataset_layout}")


def _entry_identity(entry: Mapping[str, Any], dataset_layout: str) -> int | str:
    if dataset_layout == TASK_DATASET_LAYOUT:
        return int(entry["source_episode_id"])
    relative = entry.get("source_relative_path")
    if not isinstance(relative, str) or not relative:
        raise StateConflictError("merged conversion state episode has no source_relative_path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise StateConflictError(f"invalid merged source_relative_path: {relative}")
    return path.as_posix()


def _state_entry(
    episode: SourceEpisode,
    index: int,
    *,
    dataset_layout: str = TASK_DATASET_LAYOUT,
    source_root: Path | None = None,
) -> dict[str, Any]:
    entry = {
        "source_episode_name": episode.path.name,
        "source_episode_id": episode.source_episode_id,
        "source_signature": source_episode_signature(episode),
        "lerobot_episode_index": index,
        "source_frames": len(episode.records),
        "output_frames": converted_episode_length(episode),
    }
    if dataset_layout == MERGED_DATASET_LAYOUT:
        if source_root is None:
            raise ValueError("merged state entries require source_root")
        entry.update(
            {
                "source_relative_path": _source_relative_path(episode, source_root),
                "stored_task": episode.task,
            }
        )
    elif dataset_layout != TASK_DATASET_LAYOUT:
        raise ValueError(f"unsupported dataset layout: {dataset_layout}")
    return entry


def _append_state_episode(
    state: dict[str, Any],
    episode: SourceEpisode,
    index: int,
    *,
    dataset_layout: str,
    source_root: Path,
) -> None:
    state["episodes"].append(
        _state_entry(
            episode,
            index,
            dataset_layout=dataset_layout,
            source_root=source_root,
        )
    )
    state.pop("pending_episode", None)
    if dataset_layout == MERGED_DATASET_LAYOUT:
        state["stored_tasks"] = sorted(
            {str(entry["stored_task"]) for entry in state["episodes"]}
        )


def load_conversion_state(root: Path, action_mode: str | None = None) -> dict[str, Any] | None:
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
            f"{path}: incompatible legacy conversion state; v2/20-D output requires a new output root"
        )
    config = state.get("conversion_config")
    stored_mode = state.get("action_mode")
    dataset_layout = state.get("dataset_layout", TASK_DATASET_LAYOUT)
    if stored_mode not in ACTION_MODES or not isinstance(config, dict):
        raise StateConflictError(
            f"{path}: existing output has no supported dual-action schema; use a new output root"
        )
    names = action_names(stored_mode)
    if (
        config.get("action_mode") != stored_mode
        or config.get("action_schema") != ACTION_SCHEMAS[stored_mode]
        or config.get("action_dim") != len(names)
        or config.get("action_names") != names
        or dataset_layout not in DATASET_LAYOUTS
        or not isinstance(state.get("episodes"), list)
    ):
        raise StateConflictError(f"{path}: incompatible or malformed conversion state")
    if dataset_layout == TASK_DATASET_LAYOUT:
        if not isinstance(state.get("stored_task"), str) or not state["stored_task"]:
            raise StateConflictError(f"{path}: incompatible or malformed conversion state")
    else:
        if not isinstance(state.get("source_root"), str) or not state["source_root"]:
            raise StateConflictError(f"{path}: merged conversion state has no source_root")
        for entry in state["episodes"]:
            if not isinstance(entry, Mapping):
                raise StateConflictError(f"{path}: incompatible or malformed conversion state")
            _entry_identity(entry, dataset_layout)
            if not isinstance(entry.get("stored_task"), str) or not entry["stored_task"]:
                raise StateConflictError(f"{path}: merged episode has no stored task")
    if action_mode is not None and stored_mode != action_mode:
        raise StateConflictError(f"{path}: action mode {stored_mode} differs from requested {action_mode}")
    return state


def _write_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(root / CONVERSION_STATE_FILENAME, state)


def new_conversion_state(
    source_task: Path, repo_id: str, config: ConverterConfig, schema: Mapping[str, Any],
    stored_task: str | None, action_mode: str = WHOLE_BODY_JOINT,
    shared_video_owner: str | None = None,
    dataset_layout: str = TASK_DATASET_LAYOUT,
) -> dict[str, Any]:
    if dataset_layout not in DATASET_LAYOUTS:
        raise ValueError(f"unsupported dataset layout: {dataset_layout}")
    if dataset_layout == TASK_DATASET_LAYOUT and (not isinstance(stored_task, str) or not stored_task):
        raise ValueError("task conversion state requires stored_task")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state = {
        "version": STATE_VERSION,
        "action_mode": action_mode,
        "dataset_layout": dataset_layout,
        "dataset_role": "video_owner" if action_mode == WHOLE_BODY_JOINT else "video_consumer",
        "source_task": str(source_task.resolve()),
        "repo_id": repo_id,
        "conversion_config": config.state_value(action_mode, shared_video_owner),
        "schema": dict(schema),
        "shared_video_owner": shared_video_owner,
        "episodes": [],
        "created_at": now,
        "updated_at": now,
    }
    if dataset_layout == TASK_DATASET_LAYOUT:
        state["stored_task"] = stored_task
    else:
        state["source_root"] = str(source_task.resolve())
        state["stored_tasks"] = []
    return state


def validate_incremental_state(
    state: Mapping[str, Any],
    episodes: Sequence[SourceEpisode],
    *,
    source_task: Path,
    repo_id: str,
    config: ConverterConfig,
    action_mode: str = WHOLE_BODY_JOINT,
    shared_video_owner: str | None = None,
    dataset_layout: str = TASK_DATASET_LAYOUT,
) -> list[SourceEpisode]:
    if state.get("repo_id") != repo_id:
        raise StateConflictError("conversion state repo_id differs from requested value")
    if state.get("action_mode") != action_mode:
        raise StateConflictError("conversion state action mode differs from requested value")
    if state.get("dataset_layout", TASK_DATASET_LAYOUT) != dataset_layout:
        raise StateConflictError("conversion state dataset layout differs from requested value")
    if state.get("conversion_config") != config.state_value(action_mode, shared_video_owner):
        raise StateConflictError("conversion settings conflict with the existing dataset")
    source_key = "source_root" if dataset_layout == MERGED_DATASET_LAYOUT else "source_task"
    if Path(str(state.get(source_key, ""))).resolve() != source_task.resolve():
        raise StateConflictError("conversion state belongs to a different source directory")
    if not episodes:
        if state.get("episodes"):
            raise StateConflictError("all previously converted source episodes are missing")
        return []
    if dataset_layout == TASK_DATASET_LAYOUT:
        stored_tasks = {episode.task for episode in episodes}
        if len(stored_tasks) != 1 or state.get("stored_task") not in stored_tasks:
            raise StateConflictError("stored task language or text conflicts with the existing dataset")
    elif any(not isinstance(episode.task, str) or not episode.task.strip() for episode in episodes):
        raise StateConflictError("merged source episode has no stored task")
    if state.get("schema") != schema_for_episode(episodes[0]):
        raise StateConflictError("source video schema conflicts with the existing dataset")

    by_identity: dict[int | str, SourceEpisode] = {}
    for episode in episodes:
        identity = _episode_identity(episode, dataset_layout, source_task)
        if identity in by_identity:
            label = "source episode id" if dataset_layout == TASK_DATASET_LAYOUT else "source relative path"
            raise StateConflictError(f"duplicate {label} {identity}")
        by_identity[identity] = episode
    existing_identities: set[int | str] = set()
    for expected_index, entry in enumerate(state.get("episodes", [])):
        if not isinstance(entry, Mapping) or entry.get("lerobot_episode_index") != expected_index:
            raise StateConflictError("conversion state episode indices are not contiguous")
        identity = _entry_identity(entry, dataset_layout)
        if identity in existing_identities:
            raise StateConflictError(f"conversion state has duplicate source identity {identity}")
        existing_identities.add(identity)
        episode = by_identity.get(identity)
        if episode is None:
            raise StateConflictError(f"previously converted source episode {identity} is missing")
        if entry.get("source_episode_name") != episode.path.name:
            raise StateConflictError(
                f"source episode {identity} changed name from {entry.get('source_episode_name')} to {episode.path.name}"
            )
        if entry.get("source_episode_id") != episode.source_episode_id:
            raise StateConflictError(f"source episode {identity} changed numeric id")
        if dataset_layout == MERGED_DATASET_LAYOUT and entry.get("stored_task") != episode.task:
            raise StateConflictError(f"source episode {identity} changed stored task")
        current = source_episode_signature(episode)
        if entry.get("source_signature", {}).get("digest") != current["digest"]:
            raise StateConflictError(f"{episode.path}: source changed after it was converted")
    pending = [
        episode for episode in episodes
        if _episode_identity(episode, dataset_layout, source_task) not in existing_identities
    ]
    if pending and state.get("promoted_auxiliary_streams"):
        raise StateConflictError(
            "cannot append after auxiliary streams were promoted; create a new output bundle"
        )
    return pending


def axes_feature(size: int, names: Sequence[str], dtype: str = "float32") -> dict[str, Any]:
    return {"dtype": dtype, "shape": (size,), "names": {"axes": list(names)}}


def build_features(
    videos: Mapping[str, VideoSource], action_mode: str = WHOLE_BODY_JOINT,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for stream in TRAINING_VIDEO_STREAMS:
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
            "action": axes_feature(len(action_names(action_mode)), action_names(action_mode)),
        }
    )
    return features


def _heavy_runtime() -> dict[str, Any]:
    try:
        import av
        import cv2
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets.arrow_dataset import update_metadata_with_features
        from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.feature_utils import get_hf_features_from_features
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise OptionalDependencyError(
            "LeRobot conversion dependencies are unavailable. Install the optional 'lerobot' dependency group."
        ) from exc
    return {
        "av": av, "cv2": cv2, "np": np, "pa": pa, "pq": pq,
        "LeRobotDataset": LeRobotDataset,
        "aggregate_stats": aggregate_stats,
        "get_hf_features_from_features": get_hf_features_from_features,
        "update_metadata_with_features": update_metadata_with_features,
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


def _make_frame(
    episode: SourceEpisode,
    frame_index: int,
    images: Mapping[str, Any],
    np: Any,
    action_mode: str = WHOLE_BODY_JOINT,
) -> dict[str, Any]:
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
        "action": np.asarray(
            build_action(
                record,
                episode.records[frame_index + 1] if frame_index + 1 < len(episode.records) else None,
                f"frame {frame_index}",
                action_mode,
            ),
            dtype=np.float32,
        ),
        "task": episode.task,
    }


def _add_frames(
    dataset: Any,
    episode: SourceEpisode,
    runtime: Mapping[str, Any],
    action_mode: str = WHOLE_BODY_JOINT,
) -> None:
    readers: dict[str, _VideoReader] = {}
    try:
        readers = {stream: _VideoReader(episode.videos[stream], runtime) for stream in TRAINING_VIDEO_STREAMS}
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
            dataset.add_frame(_make_frame(episode, index, images, runtime["np"], action_mode))
        # All manifest frames were consumed; prove there are no extras.
        for stream, reader in readers.items():
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
    config: ConverterConfig, *, create: bool, action_mode: str = WHOLE_BODY_JOINT,
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
            features=build_features(episode.videos, action_mode), use_videos=True,
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


def _verify_episode(
    runtime: Mapping[str, Any], root: Path, repo_id: str, index: int,
    episode: SourceEpisode, action_mode: str = WHOLE_BODY_JOINT,
) -> None:
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
            [
                build_action(
                    episode.records[position],
                    episode.records[position + 1] if position + 1 < len(episode.records) else None,
                    action_mode=action_mode,
                )
                for position in range(end - start)
            ],
            dtype=np.float32,
        )
        if (
            actions.shape != expected_actions.shape
            or not np.isfinite(actions).all()
            or not np.allclose(actions, expected_actions, rtol=1e-6, atol=1e-6)
        ):
            raise RuntimeError(f"LeRobot episode {index} action readback failed")
        expected_rows = [
            _make_frame(episode, position, {}, np, action_mode) for position in range(end - start)
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
            for stream in TRAINING_VIDEO_STREAMS:
                image = frame[video_key(stream)]
                if tuple(image.shape) != (
                    1 if episode.videos[stream].is_depth else 3,
                    episode.videos[stream].height,
                    episode.videos[stream].width,
                ):
                    raise RuntimeError(f"LeRobot episode {index} video {stream} readback failed")
        for stream in TRAINING_VIDEO_STREAMS:
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
    config: ConverterConfig, *, create: bool, action_mode: str = WHOLE_BODY_JOINT,
) -> int:
    dataset = _open_dataset(
        runtime, root, repo_id, episode, config, create=create, action_mode=action_mode,
    )
    index = int(dataset.meta.total_episodes)
    signature_before = source_episode_signature(episode)
    try:
        try:
            _add_frames(dataset, episode, runtime, action_mode)
            if source_episode_signature(episode)["digest"] != signature_before["digest"]:
                raise RuntimeError("source episode changed while conversion was running")
        except Exception:
            if getattr(dataset, "has_pending_frames", lambda: False)():
                dataset.clear_episode_buffer()
            raise
        dataset.save_episode(parallel_encoding=config.video_encoding_mode == "parallel")
    finally:
        dataset.finalize()
    _verify_episode(runtime, root, repo_id, index, episode, action_mode)
    return index


def _auxiliary_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("episode_index", pa.int64()),
            ("source_episode_id", pa.int64()),
            ("stream", pa.string()),
            ("feature_key", pa.string()),
            ("relative_path", pa.string()),
            ("codec", pa.string()),
            ("pixel_format", pa.string()),
            ("width", pa.int64()),
            ("height", pa.int64()),
            ("fps", pa.float64()),
            ("frame_count", pa.int64()),
            ("is_depth", pa.bool_()),
            ("from_timestamp", pa.float64()),
            ("to_timestamp", pa.float64()),
            ("feature_info_json", pa.string()),
            ("stats_json", pa.string()),
            ("source_signature", pa.string()),
        ]
    )


def _read_auxiliary_rows(runtime: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    path = root / AUXILIARY_INDEX_PATH
    if not path.is_file():
        return []
    return runtime["pq"].read_table(path).to_pylist()


def _write_auxiliary_rows(
    runtime: Mapping[str, Any], root: Path, rows: Sequence[Mapping[str, Any]],
) -> None:
    path = root / AUXILIARY_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    ordered = sorted(rows, key=lambda row: (int(row["episode_index"]), str(row["stream"])))
    table = runtime["pa"].Table.from_pylist([dict(row) for row in ordered], schema=_auxiliary_schema(runtime["pa"]))
    try:
        runtime["pq"].write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _video_features_for_streams(
    videos: Mapping[str, VideoSource], streams: Iterable[str],
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for stream in streams:
        video = videos[stream]
        feature: dict[str, Any] = {
            "dtype": "video",
            "shape": (video.height, video.width, 1 if video.is_depth else 3),
            "names": ["height", "width", "channels"],
        }
        if video.is_depth:
            feature["info"] = {"is_depth_map": True, "depth_unit": "mm"}
        features[video_key(stream)] = feature
    return features


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _encode_auxiliary_episode(
    runtime: Mapping[str, Any], root: Path, index: int, episode: SourceEpisode,
    config: ConverterConfig,
) -> list[dict[str, Any]]:
    streams = sorted(set(episode.videos) - set(TRAINING_VIDEO_STREAMS))
    if not streams:
        return []
    temporary_root = root / "auxiliary" / f".encoding-{index:06d}-{uuid.uuid4().hex}"
    rgb, depth = _encoders(runtime, config, episode.fps)
    dataset = runtime["LeRobotDataset"].create(
        repo_id=f"auxiliary/episode-{index:06d}",
        root=temporary_root,
        robot_type="autolife_s1_robot_v2_2",
        fps=episode.fps,
        features=_video_features_for_streams(episode.videos, streams),
        use_videos=True,
        video_backend="pyav",
        rgb_encoder=rgb,
        depth_encoder=depth,
        encoder_threads=config.encoder_threads,
    )
    readers: dict[str, _VideoReader] = {}
    try:
        readers = {stream: _VideoReader(episode.videos[stream], runtime) for stream in streams}
        for frame_index in range(len(episode.records)):
            frame: dict[str, Any] = {"task": episode.task}
            for stream in streams:
                image = readers[stream].read()
                if image is None:
                    raise RuntimeError(f"{episode.path}: auxiliary {stream} ended before frame {frame_index}")
                video = episode.videos[stream]
                expected = (video.height, video.width, 1 if video.is_depth else 3)
                if image.shape != expected:
                    raise RuntimeError(
                        f"{episode.path}: auxiliary {stream} frame {frame_index} shape {image.shape}, expected {expected}"
                    )
                frame[video_key(stream)] = image
            dataset.add_frame(frame)
        for stream, reader in readers.items():
            if reader.read() is not None:
                raise RuntimeError(f"{episode.path}: auxiliary {stream} has frames beyond manifest")
        dataset.save_episode(parallel_encoding=config.video_encoding_mode == "parallel")
    finally:
        for reader in readers.values():
            reader.close()
        dataset.finalize()

    try:
        info = json.loads((temporary_root / "meta/info.json").read_text(encoding="utf-8"))
        stats = json.loads((temporary_root / "meta/stats.json").read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        signature = source_episode_signature(episode)["digest"]
        for stream in streams:
            key = video_key(stream)
            matches = list((temporary_root / "videos" / key).glob("*/*.mp4"))
            if len(matches) != 1:
                raise RuntimeError(f"auxiliary encoder produced {len(matches)} files for {stream}")
            relative = Path("auxiliary/videos") / stream / f"chunk-{index // 1000:03d}" / f"file-{index % 1000:03d}.mp4"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(matches[0], destination)
            feature = info["features"][key]
            feature_info = feature.get("info") or {}
            container = runtime["av"].open(str(destination))
            try:
                decoded = sum(1 for _ in container.decode(video=0))
            finally:
                container.close()
            if decoded != len(episode.records):
                raise RuntimeError(
                    f"auxiliary video {stream} has {decoded} frames, expected {len(episode.records)}"
                )
            rows.append(
                {
                    "episode_index": index,
                    "source_episode_id": episode.source_episode_id,
                    "stream": stream,
                    "feature_key": key,
                    "relative_path": relative.as_posix(),
                    "codec": str(feature_info.get("video.codec") or "unknown"),
                    "pixel_format": str(feature_info.get("video.pix_fmt") or "unknown"),
                    "width": episode.videos[stream].width,
                    "height": episode.videos[stream].height,
                    "fps": episode.videos[stream].fps,
                    "frame_count": len(episode.records),
                    "is_depth": episode.videos[stream].is_depth,
                    "from_timestamp": 0.0,
                    "to_timestamp": len(episode.records) / episode.fps,
                    "feature_info_json": json.dumps(_jsonable(feature_info), sort_keys=True),
                    "stats_json": json.dumps(_jsonable(stats[key]), sort_keys=True),
                    "source_signature": signature,
                }
            )
        return rows
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _ensure_auxiliary_episode(
    runtime: Mapping[str, Any], root: Path, index: int, episode: SourceEpisode,
    config: ConverterConfig,
) -> None:
    rows = _read_auxiliary_rows(runtime, root)
    by_key = {(int(row["episode_index"]), str(row["stream"])): row for row in rows}
    expected_streams = set(episode.videos) - set(TRAINING_VIDEO_STREAMS)
    present = {stream for ep_index, stream in by_key if ep_index == index}
    if present == expected_streams:
        if not (root / AUXILIARY_INDEX_PATH).is_file():
            _write_auxiliary_rows(runtime, root, rows)
        for stream in present:
            row = by_key[(index, stream)]
            if not (root / str(row["relative_path"])).is_file():
                raise StateConflictError(f"missing indexed auxiliary video for episode {index} stream {stream}")
        return
    if present:
        raise StateConflictError(f"partial auxiliary index for episode {index}")
    rows.extend(_encode_auxiliary_episode(runtime, root, index, episode, config))
    _write_auxiliary_rows(runtime, root, rows)


def _recover_uncommitted_state(
    runtime: Mapping[str, Any], root: Path, repo_id: str, state: dict[str, Any],
    episodes: Sequence[SourceEpisode], config: ConverterConfig,
    action_mode: str = WHOLE_BODY_JOINT,
    dataset_layout: str = TASK_DATASET_LAYOUT,
    source_root: Path | None = None,
) -> list[int]:
    source_root = (source_root or Path(str(state.get("source_task", "")))).resolve()
    count = _dataset_episode_count(runtime, root, repo_id)
    known = len(state["episodes"])
    if count < known or count > known + 1:
        raise StateConflictError(
            f"LeRobot metadata/state mismatch: dataset={count}, state={known}"
        )
    if count == known:
        if dataset_layout == MERGED_DATASET_LAYOUT and state.pop("pending_episode", None) is not None:
            _write_state(root, state)
        return []
    if dataset_layout == MERGED_DATASET_LAYOUT:
        by_identity = {
            _episode_identity(episode, dataset_layout, source_root): episode for episode in episodes
        }
        pending_entry = state.get("pending_episode")
        if not isinstance(pending_entry, Mapping):
            raise StateConflictError("merged recovery has no pending source path identity")
        if pending_entry.get("lerobot_episode_index") != known:
            raise StateConflictError("merged recovery pending episode index is not contiguous")
        identity = _entry_identity(pending_entry, dataset_layout)
        episode = by_identity.get(identity)
        if episode is None:
            raise StateConflictError(f"cannot recover saved merged episode {identity}")
        if pending_entry.get("source_signature", {}).get("digest") != source_episode_signature(episode)["digest"]:
            raise StateConflictError(f"cannot recover changed merged episode {identity}")
    else:
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
    _verify_episode(runtime, root, repo_id, known, episode, action_mode)
    if action_mode == WHOLE_BODY_JOINT:
        _ensure_auxiliary_episode(runtime, root, known, episode, config)
    _append_state_episode(
        state,
        episode,
        known,
        dataset_layout=dataset_layout,
        source_root=source_root,
    )
    _write_state(root, state)
    return [known]


def _mark_pending_episode(
    root: Path,
    state: dict[str, Any],
    episode: SourceEpisode,
    index: int,
    *,
    dataset_layout: str,
    source_root: Path,
) -> None:
    if dataset_layout != MERGED_DATASET_LAYOUT:
        return
    state["pending_episode"] = _state_entry(
        episode,
        index,
        dataset_layout=dataset_layout,
        source_root=source_root,
    )
    _write_state(root, state)


def convert_task(
    episodes: Sequence[SourceEpisode],
    output_root: Path,
    repo_id: str,
    *,
    source_task: Path,
    config: ConverterConfig | None = None,
    action_mode: str = WHOLE_BODY_JOINT,
    dataset_layout: str = TASK_DATASET_LAYOUT,
) -> ConversionResult:
    """Create or incrementally append one LeRobot v3 video-owner dataset."""

    if action_mode != WHOLE_BODY_JOINT:
        raise ValueError("direct LeRobot writing is reserved for the whole_body_joint video owner")
    config = config or ConverterConfig()
    if dataset_layout not in DATASET_LAYOUTS:
        raise ValueError(f"unsupported dataset layout: {dataset_layout}")
    output_root = output_root.expanduser().resolve()
    source_task = source_task.expanduser().resolve()
    episodes = list(episodes)
    if not episodes:
        raise ValueError("at least one compatible source episode is required")
    identities = [_episode_identity(item, dataset_layout, source_task) for item in episodes]
    if len(identities) != len(set(identities)):
        duplicate = next(item for item, count in Counter(identities).items() if count > 1)
        label = "source episode id" if dataset_layout == TASK_DATASET_LAYOUT else "source relative path"
        raise StateConflictError(f"duplicate {label} {duplicate}")
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
        state = load_conversion_state(output_root, action_mode)
        if state is None:
            raise StateConflictError(
                f"{output_root}: existing output has no dual-action conversion state; use a new output root"
            )
        pending = validate_incremental_state(
            state, episodes, source_task=source_task, repo_id=repo_id, config=config,
            action_mode=action_mode, dataset_layout=dataset_layout,
        )
        recovered = _recover_uncommitted_state(
            runtime, output_root, repo_id, state, episodes, config, action_mode,
            dataset_layout, source_task,
        )
        # Recovery changes which sources still need appending.
        pending = validate_incremental_state(
            state, episodes, source_task=source_task, repo_id=repo_id, config=config,
            action_mode=action_mode, dataset_layout=dataset_layout,
        )
        existing = tuple(entry["lerobot_episode_index"] for entry in state["episodes"])
        appended: list[int] = []
        failed: list[dict[str, str]] = []
        for episode in pending:
            try:
                _mark_pending_episode(
                    output_root,
                    state,
                    episode,
                    len(state["episodes"]),
                    dataset_layout=dataset_layout,
                    source_root=source_task,
                )
                index = _append_one(
                    runtime, output_root, repo_id, episode, config, create=False,
                    action_mode=action_mode,
                )
                if index != len(state["episodes"]):
                    raise StateConflictError("LeRobot assigned a non-contiguous episode index")
                _ensure_auxiliary_episode(runtime, output_root, index, episode, config)
                _append_state_episode(
                    state,
                    episode,
                    index,
                    dataset_layout=dataset_layout,
                    source_root=source_task,
                )
                _write_state(output_root, state)
                appended.append(index)
            except Exception as exc:
                failed.append({
                    "episode": (
                        _source_relative_path(episode, source_task)
                        if dataset_layout == MERGED_DATASET_LAYOUT
                        else episode.path.name
                    ),
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                break
        return ConversionResult(
            output_root, False, existing, tuple(appended), tuple(recovered), tuple(failed), state,
        )

    staging = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
    displaced = output_root.with_name(f".{output_root.name}.report-{uuid.uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    state = new_conversion_state(
        source_task,
        repo_id,
        config,
        expected_schema,
        episodes[0].task if dataset_layout == TASK_DATASET_LAYOUT else None,
        action_mode,
        dataset_layout=dataset_layout,
    )
    appended: list[int] = []
    failed: list[dict[str, str]] = []
    try:
        for episode in episodes:
            try:
                if appended:
                    _mark_pending_episode(
                        staging,
                        state,
                        episode,
                        len(state["episodes"]),
                        dataset_layout=dataset_layout,
                        source_root=source_task,
                    )
                index = _append_one(
                    runtime, staging, repo_id, episode, config, create=not appended,
                    action_mode=action_mode,
                )
                if index != len(state["episodes"]):
                    raise StateConflictError("LeRobot assigned a non-contiguous episode index")
                _ensure_auxiliary_episode(runtime, staging, index, episode, config)
                _append_state_episode(
                    state,
                    episode,
                    index,
                    dataset_layout=dataset_layout,
                    source_root=source_task,
                )
                _write_state(staging, state)
                appended.append(index)
            except Exception as exc:
                failed.append({
                    "episode": (
                        _source_relative_path(episode, source_task)
                        if dataset_layout == MERGED_DATASET_LAYOUT
                        else episode.path.name
                    ),
                    "reason": f"{type(exc).__name__}: {exc}",
                })
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


def _replace_action_column(
    runtime: Mapping[str, Any], table: Any, episodes_by_index: Mapping[int, SourceEpisode],
    hf_features: Any,
) -> tuple[Any, Any, list[int]]:
    pa = runtime["pa"]
    np = runtime["np"]
    episode_indices = table["episode_index"].to_pylist()
    frame_indices = table["source.frame_index"].to_pylist()
    actions: list[list[float]] = []
    resolved_episode_indices: list[int] = []
    for raw_episode_index, raw_frame_index in zip(episode_indices, frame_indices):
        episode_index = int(
            raw_episode_index[0] if isinstance(raw_episode_index, list) else raw_episode_index
        )
        frame_index = int(raw_frame_index[0] if isinstance(raw_frame_index, list) else raw_frame_index)
        episode = episodes_by_index.get(episode_index)
        if episode is None or not 0 <= frame_index < len(episode.records):
            raise StateConflictError(
                f"cannot derive hybrid action for episode {episode_index} frame {frame_index}"
            )
        resolved_episode_indices.append(episode_index)
        actions.append(
            build_action(
                episode.records[frame_index],
                episode.records[frame_index + 1] if frame_index + 1 < len(episode.records) else None,
                action_mode=BODY_JOINT_EEF,
            )
        )
    array = pa.array(actions, type=pa.list_(pa.float32(), len(BODY_JOINT_EEF_ACTION_NAMES)))
    column_index = table.schema.get_field_index("action")
    if column_index < 0:
        raise StateConflictError("owner data parquet has no action column")
    rewritten = table.set_column(column_index, "action", array)
    rewritten = runtime["update_metadata_with_features"](rewritten, hf_features)
    return (
        rewritten,
        np.asarray(actions, dtype=np.float32),
        resolved_episode_indices,
    )


def _action_stats(np: Any, actions: Any) -> dict[str, Any]:
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise StateConflictError("cannot compute action statistics for an empty dataset")
    result: dict[str, Any] = {
        "min": actions.min(axis=0),
        "max": actions.max(axis=0),
        "mean": actions.mean(axis=0),
        "std": actions.std(axis=0),
        "count": np.asarray([actions.shape[0]], dtype=np.int64),
    }
    for name, quantile in (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)):
        result[name] = np.quantile(actions, quantile, axis=0)
    return _jsonable(result)


def _set_table_column(pa: Any, table: Any, name: str, values: Sequence[Any]) -> Any:
    array = pa.array(list(values))
    index = table.schema.get_field_index(name)
    if index >= 0:
        return table.set_column(index, name, array)
    return table.append_column(name, array)


def _rewrite_hybrid_episode_action_stats(
    runtime: Mapping[str, Any], root: Path, actions_by_episode: Mapping[int, list[Any]],
) -> None:
    stats_by_episode = {
        episode_index: _action_stats(runtime["np"], runtime["np"].stack(actions))
        for episode_index, actions in actions_by_episode.items()
    }
    for parquet in sorted((root / "meta/episodes").glob("*/*.parquet")):
        table = runtime["pq"].read_table(parquet)
        source_column = table["episode_index"].to_pylist()
        for stat_name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            values = [stats_by_episode[int(ep)][stat_name] for ep in source_column]
            table = _set_table_column(runtime["pa"], table, f"stats/action/{stat_name}", values)
        temporary = parquet.with_name(f".{parquet.name}.tmp-{uuid.uuid4().hex}")
        runtime["pq"].write_table(table, temporary)
        os.replace(temporary, parquet)


def _relative_directory_link(link: Path, target: Path, *, final_link: Path | None = None) -> None:
    reference = final_link or link
    relative = os.path.relpath(target, start=reference.parent)
    if os.path.isabs(relative):
        raise StateConflictError(f"shared link target must be relative: {target}")
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or os.readlink(link) != relative:
            raise StateConflictError(f"shared link conflicts with existing path: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(relative, link, target_is_directory=True)


def _hybrid_matches_owner(
    hybrid_root: Path, owner_root: Path, owner_state: Mapping[str, Any],
    config: ConverterConfig, shared_owner: str,
) -> dict[str, Any] | None:
    if not hybrid_root.exists():
        return None
    state = load_conversion_state(hybrid_root, BODY_JOINT_EEF)
    if state is None:
        raise StateConflictError(f"{hybrid_root}: existing hybrid output has no conversion state")
    if state.get("conversion_config") != config.state_value(BODY_JOINT_EEF, shared_owner):
        raise StateConflictError("hybrid conversion settings conflict with the requested bundle")
    owner_layout = owner_state.get("dataset_layout", TASK_DATASET_LAYOUT)
    if state.get("dataset_layout", TASK_DATASET_LAYOUT) != owner_layout:
        raise StateConflictError("hybrid and owner dataset layouts differ")
    if owner_layout == TASK_DATASET_LAYOUT:
        if state.get("stored_task") != owner_state.get("stored_task"):
            raise StateConflictError("hybrid and owner stored tasks differ")
    elif state.get("source_root") != owner_state.get("source_root"):
        raise StateConflictError("hybrid and owner merged source roots differ")
    owner_entries = owner_state.get("episodes")
    hybrid_entries = state.get("episodes")
    if not isinstance(owner_entries, list) or not isinstance(hybrid_entries, list):
        raise StateConflictError("owner or hybrid conversion state has invalid episodes")
    if len(hybrid_entries) > len(owner_entries):
        raise StateConflictError("hybrid dataset is ahead of its video owner")
    for hybrid, owner in zip(hybrid_entries, owner_entries):
        keys = [
            "source_episode_id", "source_episode_name", "lerobot_episode_index",
            "source_frames", "output_frames",
        ]
        if owner_layout == MERGED_DATASET_LAYOUT:
            keys.extend(("source_relative_path", "stored_task"))
        for key in keys:
            if hybrid.get(key) != owner.get(key):
                raise StateConflictError(f"hybrid and owner episode metadata differ at {key}")
        if hybrid.get("source_signature", {}).get("digest") != owner.get("source_signature", {}).get("digest"):
            raise StateConflictError("hybrid and owner source signatures differ")
    if len(hybrid_entries) == len(owner_entries):
        for name in ("videos", "auxiliary"):
            link = hybrid_root / name
            if not link.is_symlink() or link.resolve(strict=True) != (owner_root / name).resolve(strict=True):
                raise StateConflictError(f"hybrid shared {name} link is missing or incorrect")
        return state
    return None


def _link_or_copy_file(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def derive_hybrid_dataset(
    episodes: Sequence[SourceEpisode],
    owner_root: Path,
    hybrid_root: Path,
    repo_id: str,
    *,
    source_task: Path,
    config: ConverterConfig | None = None,
) -> ConversionResult:
    config = config or ConverterConfig()
    owner_root = owner_root.expanduser().resolve()
    hybrid_root = hybrid_root.expanduser().resolve()
    owner_state = load_conversion_state(owner_root, WHOLE_BODY_JOINT)
    if owner_state is None:
        raise StateConflictError("body_joint_eef requires an existing whole_body_joint video owner")
    dataset_layout = owner_state.get("dataset_layout", TASK_DATASET_LAYOUT)
    source_root = source_task.expanduser().resolve()
    state_source_key = "source_root" if dataset_layout == MERGED_DATASET_LAYOUT else "source_task"
    if Path(str(owner_state.get(state_source_key, ""))).resolve() != source_root:
        raise StateConflictError("hybrid source directory differs from its video owner")
    shared_owner = os.path.relpath(owner_root, start=hybrid_root)
    if os.path.isabs(shared_owner):
        raise StateConflictError("shared video owner must be represented by a relative path")
    prior_state = load_conversion_state(hybrid_root, BODY_JOINT_EEF) if hybrid_root.exists() else None
    existing_state = _hybrid_matches_owner(hybrid_root, owner_root, owner_state, config, shared_owner)
    owner_entries = owner_state["episodes"]
    existing_count = len(prior_state["episodes"]) if prior_state is not None else 0
    if existing_state is not None:
        indices = tuple(int(entry["lerobot_episode_index"]) for entry in existing_state["episodes"])
        return ConversionResult(hybrid_root, False, indices, (), (), (), existing_state)

    by_identity: dict[int | str, SourceEpisode] = {}
    for episode in episodes:
        identity = _episode_identity(episode, dataset_layout, source_root)
        if identity in by_identity:
            raise StateConflictError(f"duplicate source identity {identity}")
        by_identity[identity] = episode
    committed: list[SourceEpisode] = []
    episodes_by_index: dict[int, SourceEpisode] = {}
    for entry in owner_entries:
        identity = _entry_identity(entry, dataset_layout)
        episode = by_identity.get(identity)
        if episode is None:
            raise StateConflictError(f"owner episode {identity} is missing from source inputs")
        if entry.get("source_episode_id") != episode.source_episode_id:
            raise StateConflictError(f"owner episode {identity} changed numeric id")
        if dataset_layout == MERGED_DATASET_LAYOUT and entry.get("stored_task") != episode.task:
            raise StateConflictError(f"owner episode {identity} changed stored task")
        if source_episode_signature(episode)["digest"] != entry["source_signature"]["digest"]:
            raise StateConflictError(f"{episode.path}: source changed after owner conversion")
        committed.append(episode)
        episodes_by_index[int(entry["lerobot_episode_index"])] = episode
    if not committed:
        raise StateConflictError("video owner has no committed episodes to derive")

    runtime = _heavy_runtime()
    staging = hybrid_root.with_name(f".{hybrid_root.name}.staging-{uuid.uuid4().hex}")
    displaced = hybrid_root.with_name(f".{hybrid_root.name}.previous-{uuid.uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    all_actions: list[Any] = []
    actions_by_episode: dict[int, list[Any]] = {}
    try:
        shutil.copytree(
            owner_root,
            staging,
            ignore=shutil.ignore_patterns(
                "data", "videos", "auxiliary", CONVERSION_STATE_FILENAME, "conversion_report.json",
            ),
        )
        if prior_state is not None:
            prior_data = hybrid_root / "data"
            if not prior_data.is_dir():
                raise StateConflictError("existing hybrid dataset has no data directory")
            shutil.copytree(prior_data, staging / "data", copy_function=_link_or_copy_file)
        info_path = staging / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["features"]["action"] = axes_feature(
            len(BODY_JOINT_EEF_ACTION_NAMES), BODY_JOINT_EEF_ACTION_NAMES,
        )
        hf_features = runtime["get_hf_features_from_features"](info["features"])
        owner_parquets = sorted((owner_root / "data").glob("*/*.parquet"))
        if not owner_parquets:
            raise StateConflictError("owner dataset contains no data parquet files")
        owner_relatives = {path.relative_to(owner_root / "data") for path in owner_parquets}
        existing_relatives = (
            {path.relative_to(staging / "data") for path in (staging / "data").glob("*/*.parquet")}
            if (staging / "data").is_dir()
            else set()
        )
        unexpected = existing_relatives - owner_relatives
        if unexpected:
            raise StateConflictError(f"existing hybrid has stale data parquet {min(unexpected).as_posix()}")
        for owner_parquet in owner_parquets:
            relative = owner_parquet.relative_to(owner_root / "data")
            destination = staging / "data" / relative
            owner_table = runtime["pq"].read_table(owner_parquet)
            episode_indices = [int(value) for value in owner_table["episode_index"].to_pylist()]
            historical = bool(episode_indices) and all(index < existing_count for index in episode_indices)
            if historical:
                if not destination.is_file():
                    raise StateConflictError(f"existing hybrid is missing historical data parquet {relative}")
                table = runtime["pq"].read_table(destination)
                actions = runtime["np"].asarray(table["action"].to_pylist(), dtype=runtime["np"].float32)
                resolved_episode_indices = [int(value) for value in table["episode_index"].to_pylist()]
            else:
                rewritten, actions, resolved_episode_indices = _replace_action_column(
                    runtime, owner_table, episodes_by_index, hf_features,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
                runtime["pq"].write_table(rewritten, temporary)
                os.replace(temporary, destination)
            all_actions.append(actions)
            for episode_index, action in zip(resolved_episode_indices, actions):
                actions_by_episode.setdefault(episode_index, []).append(action)
        actions = runtime["np"].concatenate(all_actions, axis=0)
        atomic_write_json(info_path, info)
        stats_path = staging / "meta/stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["action"] = _action_stats(runtime["np"], actions)
        atomic_write_json(stats_path, stats)
        _rewrite_hybrid_episode_action_stats(
            runtime,
            staging,
            actions_by_episode,
        )
        state = new_conversion_state(
            source_task,
            repo_id,
            config,
            owner_state["schema"],
            owner_state.get("stored_task"),
            BODY_JOINT_EEF,
            shared_owner,
            dataset_layout,
        )
        state["episodes"] = [dict(entry) for entry in owner_entries]
        if dataset_layout == MERGED_DATASET_LAYOUT:
            state["stored_tasks"] = sorted(
                {str(entry["stored_task"]) for entry in state["episodes"]}
            )
        _write_state(staging, state)
        _relative_directory_link(
            staging / "videos", owner_root / "videos", final_link=hybrid_root / "videos",
        )
        _relative_directory_link(
            staging / "auxiliary", owner_root / "auxiliary", final_link=hybrid_root / "auxiliary",
        )
        if hybrid_root.exists():
            os.replace(hybrid_root, displaced)
        try:
            os.replace(staging, hybrid_root)
        except Exception:
            if displaced.exists() and not hybrid_root.exists():
                os.replace(displaced, hybrid_root)
            raise
        if displaced.exists():
            shutil.rmtree(displaced)
        for name in ("videos", "auxiliary"):
            if (hybrid_root / name).resolve(strict=True) != (owner_root / name).resolve(strict=True):
                raise StateConflictError(f"published hybrid {name} link does not resolve to owner")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    appended = tuple(range(existing_count, len(owner_entries)))
    existing = tuple(range(existing_count))
    return ConversionResult(hybrid_root, existing_count == 0, existing, appended, (), (), state)


def _stats_from_json(np: Any, value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise StateConflictError("auxiliary stats must be a JSON object")
    return {str(key): np.asarray(item) for key, item in parsed.items()}


def _patch_promoted_episode_metadata(
    runtime: Mapping[str, Any], dataset_root: Path, rows_by_stream: Mapping[str, list[Mapping[str, Any]]],
    chunks_size: int,
) -> None:
    lookup = {
        (int(row["episode_index"]), stream): row
        for stream, rows in rows_by_stream.items()
        for row in rows
    }
    for parquet in sorted((dataset_root / "meta/episodes").glob("*/*.parquet")):
        table = runtime["pq"].read_table(parquet)
        episode_indices = [int(value) for value in table["episode_index"].to_pylist()]
        for stream in rows_by_stream:
            key = video_key(stream)
            selected = [lookup[(episode_index, stream)] for episode_index in episode_indices]
            locator_values = {
                "chunk_index": [episode_index // chunks_size for episode_index in episode_indices],
                "file_index": [episode_index % chunks_size for episode_index in episode_indices],
                "from_timestamp": [float(row["from_timestamp"]) for row in selected],
                "to_timestamp": [float(row["to_timestamp"]) for row in selected],
            }
            for field, values in locator_values.items():
                table = _set_table_column(
                    runtime["pa"], table, f"videos/{key}/{field}", values,
                )
            episode_stats = [_stats_from_json(runtime["np"], str(row["stats_json"])) for row in selected]
            for stat_name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                values = [_jsonable(stats[stat_name]) for stats in episode_stats]
                table = _set_table_column(
                    runtime["pa"], table, f"stats/{key}/{stat_name}", values,
                )
        temporary = parquet.with_name(f".{parquet.name}.tmp-{uuid.uuid4().hex}")
        runtime["pq"].write_table(table, temporary)
        os.replace(temporary, parquet)


def _patch_promoted_dataset(
    runtime: Mapping[str, Any],
    dataset_root: Path,
    rows_by_stream: Mapping[str, list[Mapping[str, Any]]],
    *,
    create_links: bool,
    owner_root: Path,
) -> None:
    info_path = dataset_root / "meta/info.json"
    stats_path = dataset_root / "meta/stats.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size") or 1000)
    for stream, rows in rows_by_stream.items():
        first = rows[0]
        key = video_key(stream)
        feature_info = json.loads(str(first["feature_info_json"]))
        feature = {
            "dtype": "video",
            "shape": [
                int(first["height"]), int(first["width"]), 1 if bool(first["is_depth"]) else 3,
            ],
            "names": ["height", "width", "channels"],
            "info": feature_info,
        }
        current = info["features"].get(key)
        if current is not None and current != feature:
            raise StateConflictError(f"existing promoted feature conflicts with auxiliary stream {stream}")
        if create_links:
            for row in rows:
                episode_index = int(row["episode_index"])
                official = owner_root / "videos" / key / f"chunk-{episode_index // chunks_size:03d}" / f"file-{episode_index % chunks_size:03d}.mp4"
                indexed = Path(str(row["relative_path"]))
                if indexed.is_absolute() or ".." in indexed.parts:
                    raise StateConflictError(f"unsafe auxiliary index path: {indexed}")
                target = (owner_root / indexed).resolve(strict=True)
                try:
                    target.relative_to(owner_root.resolve())
                except ValueError as exc:
                    raise StateConflictError(f"auxiliary index path escapes owner: {indexed}") from exc
                expected_prefix = Path("auxiliary/videos") / stream
                try:
                    indexed.relative_to(expected_prefix)
                except ValueError as exc:
                    raise StateConflictError(f"auxiliary index path is outside stream directory: {indexed}") from exc
                if not target.is_file():
                    raise StateConflictError(f"indexed auxiliary video is missing: {target}")
                official.parent.mkdir(parents=True, exist_ok=True)
                relative_target = os.path.relpath(target, start=official.parent)
                if official.exists() or official.is_symlink():
                    if not official.is_symlink() or official.resolve(strict=True) != target.resolve(strict=True):
                        raise StateConflictError(f"official promoted video path conflicts: {official}")
                else:
                    os.symlink(relative_target, official)
        aggregate_input = [
            {key: _stats_from_json(runtime["np"], str(row["stats_json"]))}
            for row in rows
        ]
        stats[key] = _jsonable(runtime["aggregate_stats"](aggregate_input)[key])
        info["features"][key] = feature

    _patch_promoted_episode_metadata(runtime, dataset_root, rows_by_stream, chunks_size)
    atomic_write_json(stats_path, stats)
    # info.json is the exposure point: readers cannot see the feature before all dependencies exist.
    atomic_write_json(info_path, info)
    state = load_conversion_state(dataset_root)
    if state is not None:
        promoted = set(state.get("promoted_auxiliary_streams") or [])
        promoted.update(rows_by_stream)
        state["promoted_auxiliary_streams"] = sorted(promoted)
        _write_state(dataset_root, state)


def _default_hybrid_consumer(owner_root: Path) -> Path | None:
    parts = owner_root.parts
    indices = [index for index, part in enumerate(parts) if part == WHOLE_BODY_JOINT]
    if not indices:
        return None
    split = indices[-1]
    bundle_root = Path(*parts[:split])
    relative = Path(*parts[split + 1 :])
    candidate = bundle_root / BODY_JOINT_EEF / relative
    return candidate if candidate.is_dir() else None


def promote_aux_videos(
    owner_root: Path,
    streams: Sequence[str],
    *,
    consumer_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    owner_root = owner_root.expanduser().resolve()
    requested = list(dict.fromkeys(str(stream) for stream in streams))
    if not requested:
        raise ValueError("at least one auxiliary stream is required")
    state = load_conversion_state(owner_root, WHOLE_BODY_JOINT)
    if state is None:
        raise StateConflictError("promote-aux-videos requires a whole_body_joint owner")
    runtime = _heavy_runtime()
    rows = _read_auxiliary_rows(runtime, owner_root)
    episode_indices = {int(entry["lerobot_episode_index"]) for entry in state["episodes"]}
    state_by_index = {
        int(entry["lerobot_episode_index"]): entry for entry in state["episodes"]
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for stream in requested:
        selected = sorted(
            (row for row in rows if row["stream"] == stream),
            key=lambda row: int(row["episode_index"]),
        )
        if {int(row["episode_index"]) for row in selected} != episode_indices:
            raise StateConflictError(f"auxiliary stream {stream} does not cover every committed episode")
        for row in selected:
            entry = state_by_index[int(row["episode_index"])]
            if (
                int(row["source_episode_id"]) != int(entry["source_episode_id"])
                or int(row["frame_count"]) != int(entry["output_frames"])
                or str(row["source_signature"]) != str(entry["source_signature"]["digest"])
            ):
                raise StateConflictError(f"auxiliary stream {stream} index disagrees with conversion state")
        shapes = {
            (int(row["width"]), int(row["height"]), float(row["fps"]), bool(row["is_depth"]))
            for row in selected
        }
        feature_infos = {str(row["feature_info_json"]) for row in selected}
        if len(shapes) != 1 or len(feature_infos) != 1:
            raise StateConflictError(f"auxiliary stream {stream} has inconsistent schema or encoding")
        grouped[stream] = selected

    consumers = [path.expanduser().resolve() for path in (consumer_roots or [])]
    if consumer_roots is None:
        default_consumer = _default_hybrid_consumer(owner_root)
        consumers = [default_consumer] if default_consumer is not None else []
    # Validate every consumer before exposing the feature in either dataset.
    for consumer in consumers:
        consumer_state = load_conversion_state(consumer, BODY_JOINT_EEF)
        if consumer_state is None or len(consumer_state["episodes"]) != len(state["episodes"]):
            raise StateConflictError(f"hybrid consumer is missing or misaligned: {consumer}")
        owner_layout = state.get("dataset_layout", TASK_DATASET_LAYOUT)
        if consumer_state.get("dataset_layout", TASK_DATASET_LAYOUT) != owner_layout:
            raise StateConflictError(f"hybrid consumer dataset layout differs from owner: {consumer}")
        if owner_layout == TASK_DATASET_LAYOUT:
            if consumer_state.get("stored_task") != state.get("stored_task"):
                raise StateConflictError(f"hybrid consumer task differs from owner: {consumer}")
            entry_keys: tuple[str, ...] = ()
        else:
            if consumer_state.get("source_root") != state.get("source_root"):
                raise StateConflictError(f"hybrid consumer source root differs from owner: {consumer}")
            entry_keys = ("source_relative_path", "stored_task")
        for owner_entry, consumer_entry in zip(state["episodes"], consumer_state["episodes"]):
            for key in (
                "source_episode_id", "source_episode_name", "lerobot_episode_index",
                "source_frames", "output_frames", *entry_keys,
            ):
                if consumer_entry.get(key) != owner_entry.get(key):
                    raise StateConflictError(
                        f"hybrid consumer episode metadata differs from owner at {key}: {consumer}"
                    )
            if consumer_entry.get("source_signature", {}).get("digest") != owner_entry.get(
                "source_signature", {}
            ).get("digest"):
                raise StateConflictError(f"hybrid consumer source signature differs from owner: {consumer}")
        if not (consumer / "videos").is_symlink() or (consumer / "videos").resolve(strict=True) != (
            owner_root / "videos"
        ).resolve(strict=True):
            raise StateConflictError(f"hybrid consumer does not share owner videos: {consumer}")
    _patch_promoted_dataset(runtime, owner_root, grouped, create_links=True, owner_root=owner_root)
    for consumer in consumers:
        _patch_promoted_dataset(runtime, consumer, grouped, create_links=False, owner_root=owner_root)
    return {
        "owner_root": str(owner_root),
        "consumer_roots": [str(path) for path in consumers],
        "promoted_streams": requested,
        "episodes": len(episode_indices),
    }


def _bundle_ledger_path(bundle_root: Path, relative_task: Path | None) -> Path:
    base = bundle_root / ".bundle"
    return base / "bundle_state.json" if relative_task is None else base / relative_task / "bundle_state.json"


def _bundle_ledger(
    bundle_root: Path,
    relative_task: Path | None,
    episodes: Sequence[SourceEpisode],
    mode_results: Mapping[str, ConversionResult],
    errors: Sequence[Mapping[str, str]],
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    dataset_layout = MERGED_DATASET_LAYOUT if relative_task is None else TASK_DATASET_LAYOUT
    if dataset_layout == MERGED_DATASET_LAYOUT and source_root is None:
        raise ValueError("merged bundle ledger requires source_root")
    identity_root = source_root or bundle_root
    previous_path = _bundle_ledger_path(bundle_root, relative_task)
    previous: dict[tuple[int | str, str], Mapping[str, Any]] = {}
    created_at: str | None = None
    if previous_path.is_file():
        try:
            old = json.loads(previous_path.read_text(encoding="utf-8"))
            created_at = old.get("created_at") if isinstance(old.get("created_at"), str) else None
            for entry in old.get("episodes", []):
                for mode, status in entry.get("modes", {}).items():
                    previous[(_entry_identity(entry, dataset_layout), mode)] = status
        except (OSError, ValueError, json.JSONDecodeError):
            previous = {}
            created_at = None
    committed: dict[str, dict[int | str, tuple[int, str]]] = {}
    for mode, result in mode_results.items():
        indices_to_outcome = {
            **{int(index): "existing" for index in result.existing_episode_indices},
            **{int(index): "recovered" for index in result.recovered_episode_indices},
            **{int(index): "appended" for index in result.appended_episode_indices},
        }
        committed[mode] = {
            _entry_identity(entry, dataset_layout): (
                int(entry["lerobot_episode_index"]),
                indices_to_outcome.get(int(entry["lerobot_episode_index"]), "existing"),
            )
            for entry in result.state.get("episodes", [])
        }
    errors_by_mode = {str(item.get("mode")): str(item.get("reason")) for item in errors}
    entries: list[dict[str, Any]] = []
    for episode in episodes:
        identity = _episode_identity(episode, dataset_layout, identity_root)
        modes: dict[str, Any] = {}
        for mode in ACTION_MODES:
            old_status = previous.get((identity, mode), {})
            committed_result = committed.get(mode, {}).get(identity)
            preserved_commit = (
                mode not in mode_results and old_status.get("status") == "committed"
            )
            is_committed = committed_result is not None or preserved_commit
            if committed_result is not None:
                outcome = committed_result[1]
            elif preserved_commit:
                outcome = "existing"
            else:
                outcome = "pending"
            modes[mode] = {
                "status": "committed" if is_committed else "pending",
                "attempts": int(old_status.get("attempts", 0)) + (1 if mode in mode_results or mode in errors_by_mode else 0),
                "last_error": None if is_committed else errors_by_mode.get(mode, old_status.get("last_error")),
                "outcome": outcome,
            }
        entry = {
            "source_episode_name": episode.path.name,
            "source_episode_id": episode.source_episode_id,
            "source_signature": source_episode_signature(episode),
            "modes": modes,
        }
        if dataset_layout == MERGED_DATASET_LAYOUT:
            entry.update({"source_relative_path": str(identity), "stored_task": episode.task})
        entries.append(entry)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ledger = {
        "version": 1,
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "dataset_layout": dataset_layout,
        "video_owner": WHOLE_BODY_JOINT,
        "created_at": created_at or now,
        "updated_at": now,
        "episodes": entries,
    }
    if dataset_layout == TASK_DATASET_LAYOUT:
        assert relative_task is not None
        ledger["relative_task"] = relative_task.as_posix()
    else:
        assert source_root is not None
        ledger["source_root"] = str(source_root.resolve())
        ledger["stored_tasks"] = sorted({episode.task for episode in episodes})
    return ledger


def convert_task_bundle(
    episodes: Sequence[SourceEpisode],
    bundle_root: Path,
    relative_task: str | Path,
    namespace: str,
    *,
    source_task: Path,
    action_mode: str = "both",
    config: ConverterConfig | None = None,
) -> BundleConversionResult:
    if action_mode not in {"both", *ACTION_MODES}:
        raise ValueError(f"unsupported action mode: {action_mode}")
    config = config or ConverterConfig()
    bundle_root = bundle_root.expanduser().resolve()
    relative = Path(relative_task)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative task path must remain inside the bundle")
    owner_root = bundle_root / WHOLE_BODY_JOINT / relative
    hybrid_root = bundle_root / BODY_JOINT_EEF / relative
    task_slug = "__".join(relative.parts)
    results: dict[str, ConversionResult] = {}
    errors: list[dict[str, str]] = []

    if action_mode in {"both", WHOLE_BODY_JOINT}:
        try:
            results[WHOLE_BODY_JOINT] = convert_task(
                episodes,
                owner_root,
                f"{namespace}/{WHOLE_BODY_JOINT}__{task_slug}",
                source_task=source_task,
                config=config,
                action_mode=WHOLE_BODY_JOINT,
            )
            errors.extend(
                {"mode": WHOLE_BODY_JOINT, **item} for item in results[WHOLE_BODY_JOINT].failed
            )
        except Exception as exc:
            errors.append({"mode": WHOLE_BODY_JOINT, "episode": "", "reason": f"{type(exc).__name__}: {exc}"})
    elif action_mode == BODY_JOINT_EEF:
        state = load_conversion_state(owner_root, WHOLE_BODY_JOINT)
        if state is None:
            raise StateConflictError("body_joint_eef requires the existing whole_body_joint owner; use --action-mode both")
        pending = validate_incremental_state(
            state,
            episodes,
            source_task=source_task,
            repo_id=str(state["repo_id"]),
            config=config,
            action_mode=WHOLE_BODY_JOINT,
        )
        if pending:
            raise StateConflictError("whole_body_joint owner is missing requested episodes; use --action-mode both")
        results[WHOLE_BODY_JOINT] = ConversionResult(
            owner_root, False,
            tuple(int(entry["lerobot_episode_index"]) for entry in state["episodes"]),
            (), (), (), state,
        )

    if action_mode in {"both", BODY_JOINT_EEF} and WHOLE_BODY_JOINT in results:
        try:
            results[BODY_JOINT_EEF] = derive_hybrid_dataset(
                episodes,
                owner_root,
                hybrid_root,
                f"{namespace}/{BODY_JOINT_EEF}__{task_slug}",
                source_task=source_task,
                config=config,
            )
        except Exception as exc:
            errors.append({"mode": BODY_JOINT_EEF, "episode": "", "reason": f"{type(exc).__name__}: {exc}"})

    ledger = _bundle_ledger(bundle_root, relative, episodes, results, errors)
    atomic_write_json(_bundle_ledger_path(bundle_root, relative), ledger)
    return BundleConversionResult(bundle_root, action_mode, results, ledger, tuple(errors))


def convert_merged_bundle(
    episodes: Sequence[SourceEpisode],
    bundle_root: Path,
    namespace: str,
    *,
    source_root: Path,
    action_mode: str = "both",
    config: ConverterConfig | None = None,
) -> BundleConversionResult:
    """Create or incrementally append one cross-task dual-action bundle."""

    if action_mode not in {"both", *ACTION_MODES}:
        raise ValueError(f"unsupported action mode: {action_mode}")
    config = config or ConverterConfig()
    bundle_root = bundle_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    owner_root = bundle_root / WHOLE_BODY_JOINT
    hybrid_root = bundle_root / BODY_JOINT_EEF
    results: dict[str, ConversionResult] = {}
    errors: list[dict[str, str]] = []

    if action_mode in {"both", WHOLE_BODY_JOINT}:
        try:
            results[WHOLE_BODY_JOINT] = convert_task(
                episodes,
                owner_root,
                f"{namespace}/{WHOLE_BODY_JOINT}",
                source_task=source_root,
                config=config,
                action_mode=WHOLE_BODY_JOINT,
                dataset_layout=MERGED_DATASET_LAYOUT,
            )
            errors.extend(
                {"mode": WHOLE_BODY_JOINT, **item}
                for item in results[WHOLE_BODY_JOINT].failed
            )
        except Exception as exc:
            errors.append({
                "mode": WHOLE_BODY_JOINT,
                "episode": "",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    elif action_mode == BODY_JOINT_EEF:
        state = load_conversion_state(owner_root, WHOLE_BODY_JOINT)
        if state is None:
            raise StateConflictError(
                "body_joint_eef requires the existing whole_body_joint owner; use --action-mode both"
            )
        pending = validate_incremental_state(
            state,
            episodes,
            source_task=source_root,
            repo_id=str(state["repo_id"]),
            config=config,
            action_mode=WHOLE_BODY_JOINT,
            dataset_layout=MERGED_DATASET_LAYOUT,
        )
        if pending:
            raise StateConflictError(
                "whole_body_joint owner is missing requested episodes; use --action-mode both"
            )
        results[WHOLE_BODY_JOINT] = ConversionResult(
            owner_root,
            False,
            tuple(int(entry["lerobot_episode_index"]) for entry in state["episodes"]),
            (),
            (),
            (),
            state,
        )

    if action_mode in {"both", BODY_JOINT_EEF} and WHOLE_BODY_JOINT in results:
        try:
            results[BODY_JOINT_EEF] = derive_hybrid_dataset(
                episodes,
                owner_root,
                hybrid_root,
                f"{namespace}/{BODY_JOINT_EEF}",
                source_task=source_root,
                config=config,
            )
        except Exception as exc:
            errors.append({
                "mode": BODY_JOINT_EEF,
                "episode": "",
                "reason": f"{type(exc).__name__}: {exc}",
            })

    ledger = _bundle_ledger(
        bundle_root,
        None,
        episodes,
        results,
        errors,
        source_root=source_root,
    )
    atomic_write_json(_bundle_ledger_path(bundle_root, None), ledger)
    return BundleConversionResult(bundle_root, action_mode, results, ledger, tuple(errors))


__all__ = [
    "ACTION_MODES", "ACTION_NAMES_BY_MODE", "ACTION_SCHEMAS", "BODY_JOINT_EEF",
    "BODY_JOINT_EEF_ACTION_NAMES", "BundleConversionResult", "CONVERSION_SCHEMA_VERSION",
    "CONVERSION_STATE_FILENAME", "ConverterConfig", "ConversionResult",
    "OptionalDependencyError", "REQUIRED_VIDEO_STREAMS", "SourceEpisode",
    "StateConflictError", "VideoSource", "atomic_write_json", "build_action",
    "WHOLE_BODY_JOINT", "WHOLE_BODY_JOINT_ACTION_NAMES", "action_names",
    "build_features", "convert_merged_bundle", "convert_task", "convert_task_bundle",
    "converted_episode_length", "derive_hybrid_dataset",
    "load_conversion_state", "load_source_episode", "new_conversion_state",
    "MERGED_DATASET_LAYOUT", "ordered_video_streams", "relative_pose_delta", "schema_for_episode",
    "select_compatible_episodes", "source_episode_signature",
    "TASK_DATASET_LAYOUT", "validate_incremental_state", "video_key",
    "promote_aux_videos",
]
