from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .core import WorkbenchError, probe_video, resolve_video


EPISODE_RE = re.compile(r"episode_(\d+)$")
REQUIRED_COLOR_STREAMS = ("rgbd_head_color", "hand_left", "hand_right")
CONTROL_COMMANDS = (
    "commands.SET_LEFT_GRIPPER_SPEED",
    "commands.SET_LEFT_FORCE",
    "commands.SET_RIGHT_GRIPPER_SPEED",
    "commands.SET_RIGHT_FORCE",
)
FINGERPRINT_VERSION = "sha256-relative-path-size-mtime-ns-v1"
TIMESTAMP_REWRITE_VERSION = 2
VideoProbe = Callable[[Path], dict[str, Any]]


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


@dataclass(frozen=True)
class VideoValidation:
    name: str
    path: str
    width: int
    height: int
    fps: float
    frames: int
    codec: str
    pixel_format: str
    is_depth: bool

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self.frames,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "is_depth": self.is_depth,
        }


@dataclass
class EpisodeValidation:
    path: Path
    relative_path: str
    task_relative_path: str
    episode_id: int | None = None
    valid: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    task_title: str = ""
    source_description: str = ""
    frame_count: int = 0
    fps: float = 0.0
    header: dict[str, Any] = field(default_factory=dict, repr=False)
    records: list[dict[str, Any]] = field(default_factory=list, repr=False)
    videos: dict[str, VideoValidation] = field(default_factory=dict)
    control_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in CONTROL_COMMANDS})
    output_fingerprint: str = ""
    source_token: str = ""

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "videos": {
                name: {
                    "width": video.width,
                    "height": video.height,
                    "is_depth": video.is_depth,
                }
                for name, video in sorted(self.videos.items(), key=lambda item: natural_key(item[0]))
            },
        }

    @property
    def training_schema(self) -> dict[str, Any]:
        schema = self.schema
        schema["videos"] = {
            name: schema["videos"][name]
            for name in REQUIRED_COLOR_STREAMS
            if name in schema["videos"]
        }
        return schema

    @property
    def signature(self) -> str:
        return self.output_fingerprint

    @property
    def control_coverage(self) -> dict[str, Any]:
        return {
            name: {
                "frames": count,
                "total_frames": self.frame_count,
                "ratio": count / self.frame_count if self.frame_count else 0.0,
            }
            for name, count in self.control_counts.items()
        }

    def report(self) -> dict[str, Any]:
        return {
            "episode": self.relative_path,
            "episode_id": self.episode_id,
            "status": "accepted" if self.valid else "skipped",
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "frames": self.frame_count,
            "fps": self.fps,
            "task_title": self.task_title,
            "source_description": self.source_description,
            "source_signature": self.signature,
            "videos": [item.report() for item in self.videos.values()],
            "control_coverage": self.control_coverage,
        }


def discover_episodes(input_root: Path) -> list[Path]:
    root = input_root.resolve()
    found: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or not candidate.is_dir() or not EPISODE_RE.fullmatch(candidate.name):
            continue
        try:
            candidate.resolve().relative_to(root)
        except ValueError:
            continue
        found.append(candidate)
    return sorted(found, key=lambda item: natural_key(item.relative_to(root).as_posix()))


def output_fingerprint(episode: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item for item in episode.rglob("*")
        if item.is_file() and item.name != "CUT_INFO.json"
    )
    for item in files:
        stat = item.stat()
        digest.update(item.relative_to(episode).as_posix().encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _source_token(episode: Path, stream_paths: dict[str, str]) -> str:
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


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _all_numbers_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    return True


def _vector(value: object, width: int) -> bool:
    return isinstance(value, list) and len(value) == width and all(_finite_number(item) for item in value)


def _pose(value: object) -> bool:
    if not isinstance(value, dict) or not _vector(value.get("position"), 3):
        return False
    rotation = value.get("rotation")
    return (
        _vector(rotation, 4)
        and sum(float(item) ** 2 for item in rotation) > 1e-24
    )


def _parse_completed(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _ranges(value: object, source_frames: int) -> tuple[list[tuple[int, int]], bool]:
    if not isinstance(value, list):
        return [], False
    result: list[tuple[int, int]] = []
    previous_end = -1
    for item in value:
        if (
            not isinstance(item, list) or len(item) != 2
            or isinstance(item[0], bool) or isinstance(item[1], bool)
            or not isinstance(item[0], int) or not isinstance(item[1], int)
        ):
            return [], False
        start, end = item
        if start < 0 or end <= start or end > source_frames or start <= previous_end:
            return [], False
        result.append((start, end))
        previous_end = end
    return result, True


def _kept_span_count(ranges: Iterable[tuple[int, int]], frame_count: int) -> int:
    cursor = 0
    count = 0
    for start, end in ranges:
        if cursor < start:
            count += 1
        cursor = end
    if cursor < frame_count:
        count += 1
    return count


def _read_task_meta(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest(path: Path, result: EpisodeValidation) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    object_index = 0
    header_count = 0
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    result.reasons.append(f"manifest_invalid_json:{line_number}")
                    continue
                object_index += 1
                if not isinstance(row, dict):
                    result.reasons.append(f"manifest_row_not_object:{line_number}")
                    continue
                if row.get("_type") == "session_header":
                    header_count += 1
                    if object_index != 1:
                        result.reasons.append("manifest_header_not_first")
                    if not header:
                        header = row
                elif row.get("_type") == "session_footer":
                    if row.get("aborted") is True:
                        result.reasons.append("manifest_session_aborted")
                elif isinstance(row.get("_type"), str):
                    continue
                else:
                    frames.append(row)
    except (OSError, UnicodeError):
        result.reasons.append("manifest_unreadable")
    if header_count != 1:
        result.reasons.append("manifest_header_missing" if header_count == 0 else "manifest_multiple_headers")
    if len(frames) < 2:
        result.reasons.append("manifest_too_few_frames")
    return header, frames


def _audit(
    root: Path,
    episode: Path,
    relative: str,
    result: EpisodeValidation,
    *,
    require_source: bool,
) -> dict[str, Any]:
    path = episode / "CUT_INFO.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        result.reasons.append("cut_info_missing")
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        result.reasons.append("cut_info_invalid")
        return {}
    if not isinstance(value, dict):
        result.reasons.append("cut_info_not_object")
        return {}
    audit_relative = value.get("episode")
    audit_parts: tuple[str, ...] = ()
    if isinstance(audit_relative, str):
        audit_path = Path(audit_relative)
        if (
            audit_relative
            and not audit_path.is_absolute()
            and all(part not in {"", ".", ".."} for part in audit_path.parts)
        ):
            audit_parts = audit_path.parts
    relative_parts = Path(relative).parts
    if not audit_parts or audit_parts[-len(relative_parts):] != relative_parts:
        result.reasons.append("cut_info_episode_mismatch")
    mode = value.get("mode")
    if mode not in {"trim", "no_trim"}:
        result.reasons.append("cut_info_mode_invalid")
    source_frames = value.get("sourceFrames")
    output_frames = value.get("outputFrames")
    if (
        isinstance(source_frames, bool) or not isinstance(source_frames, int) or source_frames <= 0
        or isinstance(output_frames, bool) or not isinstance(output_frames, int) or output_frames <= 0
        or output_frames > source_frames
    ):
        result.reasons.append("cut_info_frame_counts_invalid")
        source_frames = output_frames = 0
    ranges, ranges_valid = _ranges(value.get("removedRanges"), source_frames)
    if not ranges_valid:
        result.reasons.append("cut_info_ranges_invalid")
    elif source_frames and output_frames != source_frames - sum(end - start for start, end in ranges):
        result.reasons.append("cut_info_frame_counts_mismatch")
    if mode == "no_trim" and (ranges or source_frames != output_frames):
        result.reasons.append("cut_info_no_trim_mismatch")
    if not _parse_completed(value.get("completedAtUtc")):
        result.reasons.append("cut_info_completion_invalid")
    version = value.get("fingerprintVersion")
    if version is not None and version != FINGERPRINT_VERSION:
        result.reasons.append("cut_info_fingerprint_version_unsupported")
    try:
        result.output_fingerprint = output_fingerprint(episode)
    except OSError:
        result.reasons.append("output_fingerprint_failed")
    stored_fingerprint = value.get("outputFingerprint")
    if isinstance(stored_fingerprint, dict):
        stored_fingerprint = stored_fingerprint.get("digest")
    if not isinstance(stored_fingerprint, str) or stored_fingerprint != result.output_fingerprint:
        result.reasons.append("output_fingerprint_mismatch")
    if (
        source_frames and ranges_valid and _kept_span_count(ranges, source_frames) > 1
        and value.get("timestampRewriteVersion") is None
    ):
        result.reasons.append("unverified_nested_timestamp_stitching")
    elif value.get("timestampRewriteVersion") is not None and value.get("timestampRewriteVersion") != TIMESTAMP_REWRITE_VERSION:
        result.reasons.append("timestamp_rewrite_version_unsupported")

    source_root_value = value.get("sourceRoot")
    source_token_value = value.get("sourceToken")
    if not isinstance(source_root_value, str) or not source_root_value:
        result.reasons.append("cut_info_source_root_invalid")
    elif not isinstance(source_token_value, str) or not source_token_value:
        result.reasons.append("cut_info_source_token_invalid")
    else:
        source_root = Path(source_root_value).expanduser()
        if not source_root.is_dir():
            issue = "source_root_unavailable"
            (result.reasons if require_source else result.warnings).append(issue)
        else:
            try:
                source_relative = Path(*audit_parts) if audit_parts else Path(relative)
                source_episode = (source_root / source_relative).resolve()
                source_episode.relative_to(source_root.resolve())
            except (OSError, ValueError):
                result.reasons.append("source_episode_outside_root")
            else:
                if not source_episode.is_dir():
                    result.reasons.append("source_episode_missing")
                else:
                    source_paths = _stream_paths_from_manifest(source_episode / "manifest.jsonl")
                    current_token = _source_token(source_episode, source_paths)
                    result.source_token = current_token
                    if current_token != source_token_value:
                        result.reasons.append("source_token_mismatch")
    return value


def _stream_paths_from_manifest(path: Path) -> dict[str, str]:
    paths: dict[str, set[str]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or isinstance(row.get("_type"), str):
                    continue
                videos = row.get("videos")
                if not isinstance(videos, dict):
                    continue
                for name, entry in videos.items():
                    relative = entry.get("path") if isinstance(entry, dict) else None
                    if isinstance(relative, str) and relative:
                        paths.setdefault(name, set()).add(relative)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return {name: next(iter(values)) for name, values in paths.items() if len(values) == 1}


def _validate_frame(row: dict[str, Any], expected: int, result: EpisodeValidation) -> None:
    if row.get("frame_idx") != expected:
        result.reasons.append(f"frame_index_not_contiguous:{expected}")
    if not _all_numbers_finite(row):
        result.reasons.append(f"non_finite_numeric_value:{expected}")
    joints = row.get("joints")
    if not isinstance(joints, dict) or any(not _vector(joints.get(name), 23) for name in ("position", "velocity", "torque")):
        result.reasons.append(f"invalid_joints:{expected}")
    current_eef = row.get("current_eef_pose")
    if not isinstance(current_eef, dict) or any(not _pose(current_eef.get(name)) for name in ("left_eef_pose", "right_eef_pose", "head_pose")):
        result.reasons.append(f"invalid_current_eef_pose:{expected}")
    target_eef = row.get("target_eef_pose")
    if not isinstance(target_eef, dict) or any(not _pose(target_eef.get(name)) for name in ("left_eef_pose", "right_eef_pose", "head_pose")):
        result.reasons.append(f"invalid_target_eef_pose:{expected}")
    for field in ("current_height_z", "target_height_z"):
        value = row.get(field)
        if not isinstance(value, dict) or not _finite_number(value.get("height_z")):
            result.reasons.append(f"invalid_{field}:{expected}")
    robot_state = row.get("robot_state")
    if (
        not isinstance(robot_state, dict)
        or isinstance(robot_state.get("state"), bool)
        or not isinstance(robot_state.get("state"), int)
    ):
        result.reasons.append(f"invalid_robot_state:{expected}")
    if isinstance(row.get("t_ns"), bool) or not isinstance(row.get("t_ns"), int):
        result.reasons.append(f"invalid_t_ns:{expected}")
    control = row.get("control")
    if not isinstance(control, dict):
        result.reasons.append(f"invalid_control:{expected}")
        return
    nested_commands = control.get("commands")
    for command in CONTROL_COMMANDS:
        nested_key = command.removeprefix("commands.")
        present = False
        value: object = None
        if command in control:
            value = control[command]
            present = True
        elif isinstance(nested_commands, dict) and nested_key in nested_commands:
            value = nested_commands[nested_key]
            present = True
        if not present:
            continue
        if not _finite_number(value):
            result.reasons.append(f"non_finite_control_command:{command}:{expected}")
        else:
            result.control_counts[command] += 1


def _validate_videos(
    episode: Path,
    frames: list[dict[str, Any]],
    fps: float,
    result: EpisodeValidation,
    probe: VideoProbe,
) -> None:
    names: set[str] = set()
    for row in frames:
        videos = row.get("videos")
        if not isinstance(videos, dict):
            result.reasons.append("frame_videos_invalid")
            continue
        names.update(str(name) for name in videos)
    stream_paths: dict[str, str] = {}
    for name in sorted(names, key=natural_key):
        values: list[str | None] = []
        for row in frames:
            videos = row.get("videos")
            entry = videos.get(name) if isinstance(videos, dict) else None
            value = entry.get("path") if isinstance(entry, dict) else None
            values.append(value if isinstance(value, str) and value else None)
        referenced = {value for value in values if value is not None}
        if not referenced:
            continue
        if len(referenced) != 1 or any(value is None for value in values):
            result.reasons.append(f"video_reference_inconsistent:{name}")
            continue
        for expected, row in enumerate(frames):
            videos = row.get("videos")
            entry = videos.get(name) if isinstance(videos, dict) else None
            if not isinstance(entry, dict) or entry.get("frame_id") != expected:
                result.reasons.append(f"video_frame_id_mismatch:{name}:{expected}")
                break
        relative = next(iter(referenced))
        try:
            video_path = resolve_video(episode, relative)
        except WorkbenchError:
            result.reasons.append(f"video_path_outside_episode:{name}")
            continue
        stream_paths[name] = relative
        checked = probe(video_path)
        if not checked.get("valid"):
            result.reasons.append(f"video_decode_failed:{name}")
            continue
        frames_decoded = int(checked.get("frames") or 0)
        video_fps = float(checked.get("fps") or 0)
        width = int(checked.get("width") or 0)
        height = int(checked.get("height") or 0)
        if frames_decoded != len(frames):
            result.reasons.append(f"video_frame_count_mismatch:{name}")
        if not math.isclose(video_fps, fps, abs_tol=0.05):
            result.reasons.append(f"video_fps_mismatch:{name}")
        if width <= 0 or height <= 0:
            result.reasons.append(f"video_dimensions_invalid:{name}")
        result.videos[name] = VideoValidation(
            name=name,
            path=relative,
            width=width,
            height=height,
            fps=video_fps,
            frames=frames_decoded,
            codec=str(checked.get("codec") or "unknown"),
            pixel_format=str(checked.get("pixelFormat") or checked.get("pixel_format") or "unknown"),
            is_depth="depth" in name.casefold(),
        )
    for required in REQUIRED_COLOR_STREAMS:
        if required not in result.videos:
            result.reasons.append(f"required_video_missing:{required}")


def validate_episode(
    input_root: Path,
    episode: Path,
    *,
    require_source: bool = False,
    video_probe: VideoProbe | None = None,
) -> EpisodeValidation:
    root = input_root.resolve()
    path = episode.resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = episode.as_posix()
        result = EpisodeValidation(episode, relative, "")
        result.reasons.append("episode_outside_input_root")
        return result
    match = EPISODE_RE.fullmatch(path.name)
    task_relative = path.parent.relative_to(root).as_posix()
    task_relative = "" if task_relative == "." else task_relative
    result = EpisodeValidation(path, relative, task_relative)
    if match is None:
        result.reasons.append("episode_name_invalid")
    else:
        result.episode_id = int(match.group(1))

    audit = _audit(root, path, relative, result, require_source=require_source)
    header, frames = _manifest(path / "manifest.jsonl", result)
    result.header = header
    result.records = frames
    result.frame_count = len(frames)
    if audit and isinstance(audit.get("outputFrames"), int) and audit.get("outputFrames") != len(frames):
        result.reasons.append("manifest_cut_info_frame_count_mismatch")

    raw_fps = header.get("fps_target")
    if not _finite_number(raw_fps) or float(raw_fps) <= 0:
        result.reasons.append("manifest_fps_invalid")
    elif not math.isclose(float(raw_fps), round(float(raw_fps))):
        result.reasons.append("manifest_fps_not_integer")
    else:
        result.fps = float(raw_fps)
    meta = _read_task_meta(path / "task_meta.json")
    result.task_title = str(header.get("task_title") or meta.get("task_title") or "").strip()
    result.source_description = str(
        header.get("task_description")
        or meta.get("task_description")
        or meta.get("description")
        or ""
    ).strip()
    if not result.task_title:
        result.reasons.append("task_title_missing")
    if result.episode_id is not None and header.get("episode_id") != result.episode_id:
        result.reasons.append("manifest_episode_id_mismatch")

    for expected, row in enumerate(frames):
        _validate_frame(row, expected, result)
        task = row.get("task")
        if not isinstance(task, dict) or task.get("task_title") != result.task_title:
            result.reasons.append(f"frame_task_mismatch:{expected}")
    if result.fps > 0:
        decoded_probe = video_probe or (lambda video: probe_video(video, decoded=True))
        _validate_videos(path, frames, result.fps, result, decoded_probe)

    # Preserve the first occurrence order while keeping reports compact.
    result.reasons = list(dict.fromkeys(result.reasons))
    result.warnings = list(dict.fromkeys(result.warnings))
    result.valid = not result.reasons
    return result


def schema_key(schema: dict[str, Any]) -> str:
    normalized = {
        "fps": round(float(schema.get("fps") or 0), 6),
        "videos": schema.get("videos") or {},
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
