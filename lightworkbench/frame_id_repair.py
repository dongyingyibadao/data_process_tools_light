from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from lightworkbench.operations import (
    FINGERPRINT_EXCLUDES,
    FINGERPRINT_VERSION,
)
from lightworkbench.validation import output_fingerprint


FRAME_ID_REASON_PREFIX = "video_frame_id_mismatch:"
REPAIR_VERSION = 1


@dataclass(frozen=True)
class TargetAudit:
    relative_path: str
    frames: int
    streams: tuple[str, ...]
    frame_ids_to_change: int


@dataclass(frozen=True)
class RepairResult:
    backup_path: Path | None
    episodes: int
    frames: int
    frame_ids_changed: int
    repaired_targets: tuple[str, ...]
    already_correct_targets: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def has_frame_id_mismatch(reasons: Iterable[object]) -> bool:
    return any(
        isinstance(reason, str) and reason.startswith(FRAME_ID_REASON_PREFIX)
        for reason in reasons
    )


def targets_from_report(report_path: Path) -> list[str]:
    report = _read_json_object(report_path)
    targets: list[str] = []
    for item in report.get("skipped", []):
        if not isinstance(item, dict):
            continue
        reasons = item.get("reasons", [])
        if not has_frame_id_mismatch(reasons):
            continue
        relative = item.get("source_relative_path") or item.get("episode")
        if not isinstance(relative, str) or not relative:
            raise ValueError("frame-id report entry has no source_relative_path")
        targets.append(relative)
    if len(targets) != len(set(targets)):
        raise ValueError("report contains duplicate frame-id repair targets")
    return sorted(targets)


def _resolve_episode(raw_root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"unsafe relative Episode path: {relative}")
    episode = (raw_root / relative_path).resolve()
    try:
        episode.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(f"Episode is outside raw root: {relative}") from exc
    if not episode.is_dir():
        raise ValueError(f"Episode directory is missing: {episode}")
    return episode


def _iter_manifest(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            yield line_number, value, line


def _audit_target(raw_root: Path, relative: str) -> TargetAudit:
    episode = _resolve_episode(raw_root, relative)
    manifest_path = episode / "manifest.jsonl"
    cut_info_path = episode / "CUT_INFO.json"
    if not manifest_path.is_file() or not cut_info_path.is_file():
        raise ValueError(f"required repair files are missing: {episode}")

    cut_info = _read_json_object(cut_info_path)
    if cut_info.get("mode") != "no_trim":
        raise ValueError(f"repair target is not mode=no_trim: {relative}")
    if cut_info.get("removedRanges") not in ([], None):
        raise ValueError(f"no_trim target has removed ranges: {relative}")
    prior_fingerprint = cut_info.get("outputFingerprint")
    if not isinstance(prior_fingerprint, str) or output_fingerprint(episode) != prior_fingerprint:
        raise ValueError(f"stored output fingerprint does not match before repair: {relative}")

    frame_index = 0
    changes = 0
    streams: set[str] = set()
    stream_paths: dict[str, str] = {}
    previous_ids: dict[str, int] = {}
    legacy_pattern_valid = True
    for line_number, row, _ in _iter_manifest(manifest_path):
        if isinstance(row.get("_type"), str):
            continue
        if row.get("frame_idx") != frame_index:
            raise ValueError(
                f"{manifest_path}:{line_number}: frame_idx is {row.get('frame_idx')!r}, "
                f"expected {frame_index}"
            )
        videos = row.get("videos")
        if not isinstance(videos, dict):
            raise ValueError(f"{manifest_path}:{line_number}: videos is not an object")
        current_streams: set[str] = set()
        for name, entry in videos.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
                continue
            stream_name = str(name)
            current_streams.add(stream_name)
            path = str(entry["path"])
            if stream_name in stream_paths and stream_paths[stream_name] != path:
                raise ValueError(f"video path changes within {relative}: {stream_name}")
            stream_paths[stream_name] = path
            frame_id = entry.get("frame_id")
            if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
                raise ValueError(f"video frame_id is not a non-negative integer: {relative}:{stream_name}")
            if frame_id != frame_index:
                changes += 1
            is_repeat = entry.get("is_repeat")
            frames_dropped = entry.get("frames_dropped", 0)
            if not isinstance(is_repeat, bool):
                legacy_pattern_valid = False
            if (
                isinstance(frames_dropped, bool)
                or not isinstance(frames_dropped, int)
                or frames_dropped < 0
            ):
                legacy_pattern_valid = False
            if frame_index and stream_name in previous_ids and isinstance(is_repeat, bool) and isinstance(frames_dropped, int):
                expected = previous_ids[stream_name] if is_repeat else previous_ids[stream_name] + 1 + frames_dropped
                if frame_id != expected:
                    legacy_pattern_valid = False
            previous_ids[stream_name] = frame_id
        if frame_index == 0:
            streams = current_streams
        elif current_streams != streams:
            raise ValueError(f"referenced video streams change within {relative}")
        frame_index += 1

    source_frames = cut_info.get("sourceFrames")
    output_frames = cut_info.get("outputFrames")
    if (
        isinstance(source_frames, bool)
        or not isinstance(source_frames, int)
        or source_frames != frame_index
        or isinstance(output_frames, bool)
        or not isinstance(output_frames, int)
        or output_frames != frame_index
    ):
        raise ValueError(
            f"CUT_INFO frame counts do not match manifest for {relative}: "
            f"source={source_frames!r}, output={output_frames!r}, manifest={frame_index}"
        )
    if frame_index < 2 or not streams:
        raise ValueError(f"repair target has no usable frames or streams: {relative}")
    if changes and not legacy_pattern_valid:
        raise ValueError(f"video frame IDs do not match the legacy no_trim pattern: {relative}")
    return TargetAudit(relative, frame_index, tuple(sorted(streams)), changes)


def audit_frame_id_target(raw_root: Path, relative: str) -> TargetAudit:
    resolved_root = raw_root.expanduser().resolve()
    lock_path = resolved_root / ".lightworkbench-frame-id-repair.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        return _audit_target(resolved_root, relative)


def _write_backup(
    backup_path: Path,
    raw_root: Path,
    source_label: str,
    targets: list[TargetAudit],
) -> None:
    if backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = backup_path.with_name(f".{backup_path.name}.tmp-{uuid.uuid4().hex}")
    file_metadata = []
    for target in targets:
        episode = _resolve_episode(raw_root, target.relative_path)
        for name in ("manifest.jsonl", "CUT_INFO.json"):
            path = episode / name
            file_stat = path.stat()
            file_metadata.append(
                {
                    "path": f"{target.relative_path}/{name}",
                    "size": file_stat.st_size,
                    "mode": stat.S_IMODE(file_stat.st_mode),
                    "mtime_ns": file_stat.st_mtime_ns,
                    "sha256": _file_sha256(path),
                }
            )
    metadata = {
        "version": 1,
        "created_at_utc": _utc_now(),
        "raw_root": str(raw_root),
        "source": source_label,
        "target_count": len(targets),
        "targets": [target.relative_path for target in targets],
        "files": file_metadata,
    }
    try:
        with zipfile.ZipFile(
            temporary,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr("_repair_metadata.json", json.dumps(metadata, indent=2) + "\n")
            for target in targets:
                episode = _resolve_episode(raw_root, target.relative_path)
                for name in ("manifest.jsonl", "CUT_INFO.json"):
                    archive.write(episode / name, arcname=f"{target.relative_path}/{name}")
        with zipfile.ZipFile(temporary) as archive:
            broken = archive.testzip()
            if broken is not None:
                raise RuntimeError(f"backup ZIP verification failed at {broken}")
            for item in file_metadata:
                if hashlib.sha256(archive.read(item["path"])).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"backup content digest differs for {item['path']}")
        os.link(temporary, backup_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_file(path: Path, content: bytes, original_stat: os.stat_result) -> None:
    temporary = path.with_name(f".{path.name}.rollback-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, stat.S_IMODE(original_stat.st_mode))
        os.replace(temporary, path)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    finally:
        if temporary.exists():
            temporary.unlink()


def _restore_backup_file(path: Path, content: bytes, metadata: dict[str, Any]) -> None:
    if hashlib.sha256(content).hexdigest() != metadata.get("sha256"):
        raise RuntimeError(f"backup content digest differs for {metadata.get('path')}")
    temporary = path.with_name(f".{path.name}.rollback-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, int(metadata["mode"]))
        os.replace(temporary, path)
        mtime_ns = int(metadata["mtime_ns"])
        os.utime(path, ns=(mtime_ns, mtime_ns))
    finally:
        if temporary.exists():
            temporary.unlink()


def _rollback_batch(backup_path: Path, raw_root: Path, targets: list[TargetAudit]) -> None:
    with zipfile.ZipFile(backup_path) as archive:
        metadata = json.loads(archive.read("_repair_metadata.json"))
        files = {
            str(item["path"]): item
            for item in metadata.get("files", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        for target in reversed(targets):
            episode = _resolve_episode(raw_root, target.relative_path)
            for name in ("manifest.jsonl", "CUT_INFO.json"):
                member = f"{target.relative_path}/{name}"
                if member not in files:
                    raise RuntimeError(f"backup metadata is missing {member}")
                _restore_backup_file(episode / name, archive.read(member), files[member])
            cut_info = _read_json_object(episode / "CUT_INFO.json")
            if cut_info.get("outputFingerprint") != output_fingerprint(episode):
                raise RuntimeError(f"restored fingerprint does not validate: {target.relative_path}")


def _rewrite_manifest(path: Path) -> int:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    original_stat = path.stat()
    changed = 0
    frame_index = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for _, row, original_line in _iter_manifest(path):
                if isinstance(row.get("_type"), str):
                    output.write(original_line if original_line.endswith("\n") else original_line + "\n")
                    continue
                videos = row["videos"]
                row_changed = False
                for entry in videos.values():
                    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
                        continue
                    if entry.get("frame_id") != frame_index:
                        entry["frame_id"] = frame_index
                        changed += 1
                        row_changed = True
                if row_changed:
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                else:
                    output.write(original_line if original_line.endswith("\n") else original_line + "\n")
                frame_index += 1
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(original_stat.st_mode))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return changed


def _repair_target(raw_root: Path, target: TargetAudit) -> int:
    episode = _resolve_episode(raw_root, target.relative_path)
    manifest_path = episode / "manifest.jsonl"
    cut_info_path = episode / "CUT_INFO.json"
    manifest_stat = manifest_path.stat()
    cut_info_stat = cut_info_path.stat()
    original_manifest = manifest_path.read_bytes()
    original_cut_info = cut_info_path.read_bytes()
    cut_info = _read_json_object(cut_info_path)
    prior_fingerprint = cut_info.get("outputFingerprint")
    if not isinstance(prior_fingerprint, str) or output_fingerprint(episode) != prior_fingerprint:
        raise ValueError(f"stored output fingerprint does not match before repair: {target.relative_path}")

    try:
        changed = _rewrite_manifest(manifest_path)
        if changed != target.frame_ids_to_change:
            raise RuntimeError(
                f"changed frame-id count differs from dry-run for {target.relative_path}: "
                f"{changed} != {target.frame_ids_to_change}"
            )
        current_fingerprint = output_fingerprint(episode)
        repairs = cut_info.get("repairs")
        if repairs is None:
            repairs = []
        if not isinstance(repairs, list):
            raise ValueError(f"CUT_INFO repairs field is not a list: {target.relative_path}")
        repairs.append(
            {
                "type": "video_frame_id_reindex",
                "version": REPAIR_VERSION,
                "completedAtUtc": _utc_now(),
                "frames": target.frames,
                "streams": list(target.streams),
                "previousOutputFingerprint": prior_fingerprint,
            }
        )
        cut_info["repairs"] = repairs
        cut_info["fingerprintVersion"] = FINGERPRINT_VERSION
        cut_info["fingerprintExcludes"] = list(FINGERPRINT_EXCLUDES)
        cut_info["outputFingerprint"] = current_fingerprint
        _atomic_write_json(cut_info_path, cut_info)
        if output_fingerprint(episode) != current_fingerprint:
            raise RuntimeError(f"published fingerprint does not validate: {target.relative_path}")
        return changed
    except Exception:
        _restore_file(manifest_path, original_manifest, manifest_stat)
        _restore_file(cut_info_path, original_cut_info, cut_info_stat)
        raise


def repair_frame_ids(
    raw_root: Path,
    relative_targets: Sequence[str],
    backup_path: Path,
    *,
    source_label: str,
    progress: Callable[[int, int, TargetAudit], None] | None = None,
) -> RepairResult:
    raw_root = raw_root.expanduser().resolve()
    backup_path = backup_path.expanduser().resolve()
    lock_path = raw_root / ".lightworkbench-frame-id-repair.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if len(relative_targets) != len(set(relative_targets)):
            raise ValueError("frame-id repair targets contain duplicates")
        audits = [_audit_target(raw_root, relative) for relative in sorted(relative_targets)]
        pending = [target for target in audits if target.frame_ids_to_change]
        already_correct = tuple(
            target.relative_path for target in audits if not target.frame_ids_to_change
        )
        if not pending:
            return RepairResult(
                backup_path=None,
                episodes=0,
                frames=0,
                frame_ids_changed=0,
                repaired_targets=(),
                already_correct_targets=already_correct,
            )
        _write_backup(backup_path, raw_root, source_label, pending)

        changed = 0
        completed: list[TargetAudit] = []
        try:
            for index, target in enumerate(pending, 1):
                changed += _repair_target(raw_root, target)
                completed.append(target)
                if progress is not None:
                    progress(index, len(pending), target)
        except Exception:
            if completed:
                _rollback_batch(backup_path, raw_root, completed)
            raise

        try:
            verified = [_audit_target(raw_root, target.relative_path) for target in audits]
            remaining = sum(target.frame_ids_to_change for target in verified)
            if remaining:
                raise RuntimeError(f"repair completed with {remaining} mismatched frame IDs")
        except Exception:
            _rollback_batch(backup_path, raw_root, pending)
            raise
        return RepairResult(
            backup_path=backup_path,
            episodes=len(pending),
            frames=sum(target.frames for target in pending),
            frame_ids_changed=changed,
            repaired_targets=tuple(target.relative_path for target in pending),
            already_correct_targets=already_correct,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair Episode-local video frame_id values selected from a LeRobot "
            "conversion report. The default mode is read-only."
        )
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument(
        "--expected-count",
        required=True,
        type=int,
        help="refuse to continue unless the report selects exactly this many Episodes",
    )
    parser.add_argument("--apply", action="store_true", help="apply repairs; otherwise dry-run")
    parser.add_argument("--backup", type=Path, help="required ZIP destination with --apply")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report_path = args.report.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    if not report_path.is_file():
        raise SystemExit(f"report does not exist: {report_path}")
    if not raw_root.is_dir():
        raise SystemExit(f"raw root does not exist: {raw_root}")

    relative_targets = targets_from_report(report_path)
    if len(relative_targets) != args.expected_count:
        raise SystemExit(
            f"selected {len(relative_targets)} Episodes, expected {args.expected_count}; refusing to continue"
        )
    audits = [_audit_target(raw_root, relative) for relative in relative_targets]
    already_correct = [target.relative_path for target in audits if not target.frame_ids_to_change]
    if already_correct:
        raise SystemExit(
            f"report is stale: {len(already_correct)} selected Episodes already have local frame IDs"
        )
    total_frames = sum(target.frames for target in audits)
    total_changes = sum(target.frame_ids_to_change for target in audits)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "episodes": len(audits),
                "frames": total_frames,
                "frame_ids_to_change": total_changes,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.apply:
        return 0
    if args.backup is None:
        raise SystemExit("--backup is required with --apply")

    backup_path = args.backup.expanduser().resolve()
    print(f"backup: {backup_path}", flush=True)

    def progress(index: int, total: int, target: TargetAudit) -> None:
        print(f"repaired {index}/{total}: {target.relative_path}", flush=True)

    result = repair_frame_ids(
        raw_root,
        relative_targets,
        backup_path,
        source_label=str(report_path),
        progress=progress,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "episodes": result.episodes,
                "frames": result.frames,
                "frame_ids_changed": result.frame_ids_changed,
                "remaining_mismatches": 0,
                "backup": str(result.backup_path) if result.backup_path is not None else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
