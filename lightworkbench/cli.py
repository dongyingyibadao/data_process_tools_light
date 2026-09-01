from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import RESOURCES
from .validation import (
    REQUIRED_COLOR_STREAMS,
    EpisodeValidation,
    discover_episodes,
    natural_key,
    schema_key,
    validate_episode,
)


SUMMARY_NAME = "conversion_summary.json"
REPORT_NAME = "conversion_report.json"
STATE_NAME = "conversion_state.json"
DUAL_STATE_VERSION = 3
ACTION_MODES = ("body_joint_eef", "whole_body_joint")
NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _stored_schema_key(state: dict[str, Any]) -> str | None:
    schema = state.get("schema")
    if not isinstance(schema, dict):
        return None
    fps = schema.get("fps")
    videos = schema.get("videos")
    if isinstance(videos, dict):
        videos = {
            str(name): {
                "width": value.get("width"),
                "height": value.get("height"),
                "is_depth": bool(value.get("is_depth")),
            }
            for name, value in videos.items()
            if name in REQUIRED_COLOR_STREAMS and isinstance(value, dict)
        }
    else:
        # Accept the equivalent split representation used by early v2 state prototypes.
        keys = schema.get("video_keys")
        dimensions = schema.get("video_dimensions")
        depth = set(schema.get("depth_video_keys") or [])
        if not isinstance(keys, list) or not isinstance(dimensions, dict):
            return None
        videos = {}
        for name in keys:
            size = dimensions.get(name)
            if isinstance(name, str) and isinstance(size, (list, tuple)) and len(size) == 2:
                videos[name] = {"width": size[0], "height": size[1], "is_depth": name in depth}
    try:
        return schema_key({"fps": fps, "videos": videos})
    except (TypeError, ValueError):
        return None


def _choose_schema(
    episodes: list[EpisodeValidation],
    output_root: Path,
    relative: str,
    stored_task: str,
    action_mode: str,
) -> tuple[list[EpisodeValidation], list[EpisodeValidation], dict[str, Any] | None, str | None]:
    if not episodes:
        return [], [], None, None
    expected_key: str | None = None
    legacy = _read_json(output_root / relative / STATE_NAME)
    if legacy is not None:
        return [], episodes, None, "incompatible_v2_20d_action_schema_requires_new_output_root"
    requested = ACTION_MODES if action_mode == "both" else (action_mode,)
    if action_mode == "body_joint_eef":
        requested = ("whole_body_joint", "body_joint_eef")
    dimensions = {"body_joint_eef": 21, "whole_body_joint": 23}
    for mode in requested:
        state = _read_json(output_root / mode / relative / STATE_NAME)
        if state is None:
            if action_mode == "body_joint_eef" and mode == "whole_body_joint":
                return [], episodes, None, "body_joint_eef_requires_video_owner_use_both"
            continue
        if not state:
            return [], episodes, None, "conversion_state_invalid"
        config = state.get("conversion_config")
        if (
            state.get("version") != DUAL_STATE_VERSION
            or state.get("action_mode") != mode
            or not isinstance(config, dict)
            or config.get("action_dim") != dimensions[mode]
        ):
            return [], episodes, None, "incompatible_v2_20d_action_schema_requires_new_output_root"
        if state.get("stored_task") != stored_task:
            return [], episodes, None, "stored_task_language_or_text_conflict"
        current_key = _stored_schema_key(state)
        if current_key is None:
            return [], episodes, None, "conversion_state_schema_invalid"
        if expected_key is not None and current_key != expected_key:
            return [], episodes, None, "paired_conversion_state_schema_conflict"
        expected_key = current_key
    if expected_key is None:
        groups: OrderedDict[str, list[EpisodeValidation]] = OrderedDict()
        for episode in episodes:
            groups.setdefault(schema_key(episode.training_schema), []).append(episode)
        expected_key = max(groups, key=lambda key: len(groups[key]))
    accepted = [episode for episode in episodes if schema_key(episode.training_schema) == expected_key]
    outliers = [episode for episode in episodes if schema_key(episode.training_schema) != expected_key]
    selected_schema = accepted[0].training_schema if accepted else None
    return accepted, outliers, selected_schema, None


def _task_instruction(episode: EpisodeValidation) -> str:
    return " ".join(episode.task_title.replace("_", " ").split())


def _select_episodes(
    input_root: Path, discovered: list[Path], selectors: Sequence[str],
) -> tuple[list[Path], list[str]]:
    if not selectors:
        return discovered, []
    normalized: list[str] = []
    for raw in selectors:
        selector = Path(raw.strip())
        if not raw.strip() or selector.is_absolute() or ".." in selector.parts:
            raise ValueError(f"invalid included Episode path: {raw}")
        value = selector.as_posix().removeprefix("./")
        if value in normalized:
            raise ValueError(f"duplicate included Episode path: {value}")
        normalized.append(value)
    by_relative = {path.relative_to(input_root).as_posix(): path for path in discovered}
    missing = [value for value in normalized if value not in by_relative]
    if missing:
        raise ValueError(f"included Episode path was not discovered: {missing[0]}")
    return [by_relative[value] for value in normalized], normalized


def _validate_episodes(
    input_root: Path,
    episodes: Sequence[Path],
    *,
    require_source: bool,
    requested_workers: int,
) -> tuple[list[EpisodeValidation], int]:
    total = len(episodes)
    if total == 0:
        return [], 0
    workers = min(total, max(1, requested_workers), max(1, int(RESOURCES["budget"])))
    print(f"preflight: validating {total} Episodes with {workers} workers", file=sys.stderr, flush=True)

    checked: dict[int, EpisodeValidation] = {}
    progress_step = max(1, total // 20)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="preflight") as pool:
        futures = {
            pool.submit(
                validate_episode,
                input_root,
                episode,
                require_source=require_source,
            ): index
            for index, episode in enumerate(episodes)
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                checked[futures[future]] = future.result()
                if completed == total or completed % progress_step == 0:
                    print(f"preflight: validated {completed}/{total} Episodes", file=sys.stderr, flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return [checked[index] for index in range(total)], workers


def _conversion_payload(value: object) -> Any:
    if is_dataclass(value):
        return _conversion_payload(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _conversion_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_conversion_payload(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _report(
    *,
    input_root: Path,
    output_root: Path,
    relative: str,
    namespace: str,
    action_mode: str,
    encoder_config: dict[str, Any],
    task_title: str,
    source_description: str,
    stored_task: str,
    schema: dict[str, Any] | None,
    accepted: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    conversion: Any = None,
) -> dict[str, Any]:
    warnings = sorted({warning for item in accepted + skipped for warning in item.get("warnings", [])})
    payload = {
        "version": 1,
        "generated_at_utc": _now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "task_relative_path": relative,
        "namespace": namespace,
        "task_language": "english",
        "action_mode": action_mode,
        "encoder_config": encoder_config,
        "task_title": task_title,
        "source_description": source_description,
        "stored_task": stored_task,
        "schema": schema,
        "accepted": accepted,
        "skipped": skipped,
        "failed": failed,
        "freshness_warnings": warnings,
        "counts": {"accepted": len(accepted), "skipped": len(skipped), "failed": len(failed)},
    }
    if conversion is not None:
        payload["conversion"] = _conversion_payload(conversion)
    return payload


def _engine_import() -> Any:
    try:
        from . import lerobot_converter
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot conversion dependencies are unavailable; install "
            "data-autopro-tools-light[lerobot] in the documented Python 3.12 environment"
        ) from exc
    return lerobot_converter


def _run_conversion(
    engine: Any,
    episodes: list[EpisodeValidation],
    output_root: Path,
    relative: str,
    namespace: str,
    task_title: str,
    stored_task: str,
    source_task: Path,
    action_mode: str,
    encoder_config: dict[str, Any],
) -> Any:
    try:
        sources = [engine.load_source_episode(item.path, stored_task) for item in episodes]
        config = engine.ConverterConfig(**encoder_config)
        return engine.convert_task_bundle(
            sources,
            output_root,
            relative,
            namespace,
            source_task=source_task,
            config=config,
            action_mode=action_mode,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot conversion dependencies are unavailable; install "
            "data-autopro-tools-light[lerobot] in the documented Python 3.12 environment"
        ) from exc


def _conversion_reports(
    result: Any, episodes: list[EpisodeValidation]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requested = ACTION_MODES if getattr(result, "action_mode", "both") == "both" else (
        getattr(result, "action_mode", "both"),
    )
    ledger = getattr(result, "ledger", {})
    statuses = {
        int(item["source_episode_id"]): item.get("modes", {})
        for item in ledger.get("episodes", [])
        if isinstance(item, dict) and isinstance(item.get("source_episode_id"), int)
    }
    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for episode in episodes:
        mode_status = statuses.get(episode.episode_id, {})
        if all(mode_status.get(mode, {}).get("status") == "committed" for mode in requested):
            succeeded.append(episode.report())
            continue
        reasons = [
            f"{mode}:{mode_status.get(mode, {}).get('last_error') or 'pending'}"
            for mode in requested
            if mode_status.get(mode, {}).get("status") != "committed"
        ]
        reason = "; ".join(reasons) or "not committed"
        failed.append({
            **episode.report(),
            "status": "failed",
            "reasons": [f"conversion_failed:{reason}"],
        })
    return succeeded, failed


def _merged_stored_schema_key(
    output_root: Path, action_mode: str,
) -> tuple[str | None, str | None]:
    legacy = _read_json(output_root / STATE_NAME)
    if legacy is not None:
        return None, "incompatible_v2_20d_action_schema_requires_new_output_root"
    owner_state = _read_json(output_root / "whole_body_joint" / STATE_NAME)
    if owner_state is None:
        if action_mode == "body_joint_eef":
            return None, "body_joint_eef_requires_video_owner_use_both"
        return None, None
    if not owner_state:
        return None, "conversion_state_invalid"
    config = owner_state.get("conversion_config")
    if (
        owner_state.get("version") != DUAL_STATE_VERSION
        or owner_state.get("action_mode") != "whole_body_joint"
        or owner_state.get("dataset_layout") != "merged"
        or not isinstance(config, dict)
        or config.get("action_dim") != 23
    ):
        return None, "incompatible_v2_20d_action_schema_requires_new_output_root"
    expected_key = _stored_schema_key(owner_state)
    if expected_key is None:
        return None, "conversion_state_schema_invalid"

    requested = ACTION_MODES if action_mode == "both" else (action_mode,)
    for mode in requested:
        if mode == "whole_body_joint":
            continue
        state = _read_json(output_root / mode / STATE_NAME)
        if state is None:
            continue
        config = state.get("conversion_config") if state else None
        if (
            not state
            or state.get("version") != DUAL_STATE_VERSION
            or state.get("action_mode") != mode
            or state.get("dataset_layout") != "merged"
            or not isinstance(config, dict)
            or config.get("action_dim") != 21
        ):
            return None, "incompatible_v2_20d_action_schema_requires_new_output_root"
        current_key = _stored_schema_key(state)
        if current_key is None:
            return None, "conversion_state_schema_invalid"
        if current_key != expected_key:
            return None, "paired_conversion_state_schema_conflict"
    return expected_key, None


def _choose_merged_schema(
    episodes: list[EpisodeValidation], expected_key: str | None = None,
) -> tuple[list[EpisodeValidation], list[EpisodeValidation], dict[str, Any] | None]:
    if not episodes:
        return [], [], None
    if expected_key is None:
        groups: OrderedDict[str, list[EpisodeValidation]] = OrderedDict()
        for episode in episodes:
            groups.setdefault(schema_key(episode.training_schema), []).append(episode)
        expected_key = max(groups, key=lambda key: len(groups[key]))
    accepted = [episode for episode in episodes if schema_key(episode.training_schema) == expected_key]
    outliers = [episode for episode in episodes if schema_key(episode.training_schema) != expected_key]
    return accepted, outliers, accepted[0].training_schema if accepted else None


def _load_merged_sources(
    engine: Any,
    episodes: list[EpisodeValidation],
) -> tuple[list[Any], list[EpisodeValidation], list[dict[str, Any]]]:
    sources: list[Any] = []
    loaded: list[EpisodeValidation] = []
    skipped: list[dict[str, Any]] = []
    for item in episodes:
        try:
            source = engine.load_source_episode(item.path, _task_instruction(item))
        except (ImportError, ModuleNotFoundError):
            raise
        except Exception as exc:
            dependency_error = getattr(engine, "OptionalDependencyError", None)
            if isinstance(dependency_error, type) and isinstance(exc, dependency_error):
                raise
            report = _merged_episode_report(item)
            report["status"] = "skipped"
            reason = str(exc) or type(exc).__name__
            report["reasons"] = [f"source_load_failed:{reason}"]
            skipped.append(report)
            continue
        sources.append(source)
        loaded.append(item)
    return sources, loaded, skipped


def _run_merged_conversion(
    engine: Any,
    sources: list[Any],
    input_root: Path,
    output_root: Path,
    namespace: str,
    action_mode: str,
    encoder_config: dict[str, Any],
) -> Any:
    try:
        config = engine.ConverterConfig(**encoder_config)
        return engine.convert_merged_bundle(
            sources,
            output_root,
            namespace,
            source_root=input_root,
            config=config,
            action_mode=action_mode,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot conversion dependencies are unavailable; install "
            "data-autopro-tools-light[lerobot] in the documented Python 3.12 environment"
        ) from exc


def _merged_episode_report(episode: EpisodeValidation) -> dict[str, Any]:
    report = episode.report()
    report["source_relative_path"] = episode.relative_path
    report["stored_task"] = _task_instruction(episode) if episode.task_title else ""
    return report


def _merged_conversion_reports(
    result: Any, episodes: list[EpisodeValidation],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requested = ACTION_MODES if getattr(result, "action_mode", "both") == "both" else (
        getattr(result, "action_mode", "both"),
    )
    ledger = getattr(result, "ledger", {})
    entries = ledger.get("episodes", []) if isinstance(ledger, dict) else []
    statuses = {
        str(item["source_relative_path"]): item.get("modes", {})
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("source_relative_path"), str)
    }
    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for episode in episodes:
        raw_status = statuses.get(episode.relative_path, {})
        mode_status = raw_status if isinstance(raw_status, dict) else {}
        report = _merged_episode_report(episode)
        report["conversion"] = {"modes": _conversion_payload(mode_status)}
        if all(
            isinstance(mode_status.get(mode), dict)
            and mode_status[mode].get("status") == "committed"
            for mode in requested
        ):
            succeeded.append(report)
            continue
        reasons = [
            f"{mode}:{(mode_status.get(mode) or {}).get('last_error') or 'pending'}"
            for mode in requested
            if not isinstance(mode_status.get(mode), dict)
            or mode_status[mode].get("status") != "committed"
        ]
        report["status"] = "failed"
        report["reasons"] = [f"conversion_failed:{'; '.join(reasons) or 'not committed'}"]
        failed.append(report)
    return succeeded, failed


def convert(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        print(f"input root is not a directory: {input_root}", file=sys.stderr)
        return 2
    if not NAMESPACE_RE.fullmatch(args.namespace):
        print(f"invalid namespace: {args.namespace}", file=sys.stderr)
        return 2
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        print("input and output roots must be separate, non-overlapping directories", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)
    encoder_config = {
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "encoder_preset": args.encoder_preset,
        "encoder_threads": args.encoder_threads,
        "video_encoding_mode": args.video_encoding_mode,
    }

    discovered_all = discover_episodes(input_root)
    try:
        discovered, included = _select_episodes(input_root, discovered_all, args.include_episode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    checked_episodes, preflight_workers = _validate_episodes(
        input_root,
        discovered,
        require_source=args.require_source,
        requested_workers=args.preflight_workers,
    )
    grouped: OrderedDict[str, list[EpisodeValidation]] = OrderedDict()
    for checked in checked_episodes:
        grouped.setdefault(checked.task_relative_path, []).append(checked)

    reports: list[dict[str, Any]] = []
    rejected = False
    engine: Any = None
    for relative, checked_episodes in grouped.items():
        checked_episodes.sort(key=lambda item: natural_key(item.relative_path))
        report_output = output_root / relative
        valid = [item for item in checked_episodes if item.valid]
        skipped_reports = [item.report() for item in checked_episodes if not item.valid]
        rejected = rejected or bool(skipped_reports)

        task_title = valid[0].task_title if valid else (checked_episodes[0].task_title if checked_episodes else "")
        source_description = valid[0].source_description if valid else (
            checked_episodes[0].source_description if checked_episodes else ""
        )
        metadata_outliers = [item for item in valid if item.task_title != task_title]
        for item in metadata_outliers:
            report = item.report()
            report["status"] = "skipped"
            report["reasons"] = ["task_metadata_conflict"]
            skipped_reports.append(report)
        if metadata_outliers:
            rejected = True
            excluded = {item.relative_path for item in metadata_outliers}
            valid = [item for item in valid if item.relative_path not in excluded]

        stored_task = _task_instruction(valid[0]) if valid else ""

        id_counts = Counter(item.episode_id for item in valid)
        duplicate_ids = {episode_id for episode_id, count in id_counts.items() if count > 1}
        duplicate_episodes = [item for item in valid if item.episode_id in duplicate_ids]
        for item in duplicate_episodes:
            report = item.report()
            report["status"] = "skipped"
            report["reasons"] = [f"duplicate_episode_id:{item.episode_id}"]
            skipped_reports.append(report)
        if duplicate_episodes:
            rejected = True
            duplicates = {item.relative_path for item in duplicate_episodes}
            valid = [item for item in valid if item.relative_path not in duplicates]

        accepted, schema_outliers, selected_schema, state_error = _choose_schema(
            valid, output_root, relative, stored_task, args.action_mode,
        )
        for item in schema_outliers:
            report = item.report()
            report["status"] = "skipped"
            report["reasons"] = [state_error or "schema_outlier"]
            skipped_reports.append(report)
        if schema_outliers:
            rejected = True
        accepted_reports = [item.report() for item in accepted]
        failed_reports: list[dict[str, Any]] = []
        conversion_result: Any = None

        if accepted and not args.preflight_only:
            try:
                if engine is None:
                    engine = _engine_import()
                conversion_result = _run_conversion(
                    engine,
                    accepted,
                    output_root,
                    relative,
                    args.namespace,
                    task_title,
                    stored_task,
                    accepted[0].path.parent,
                    args.action_mode,
                    encoder_config,
                )
                if getattr(conversion_result, "failed", ()):
                    rejected = True
                    accepted_reports, failed_reports = _conversion_reports(conversion_result, accepted)

            except Exception as exc:  # Continue independent tasks and preserve the failure in the report.
                rejected = True
                reason = str(exc) or type(exc).__name__
                failed_reports = [
                    {**item.report(), "status": "failed", "reasons": [f"conversion_failed:{reason}"]}
                    for item in accepted
                ]
                accepted_reports = []

        report = _report(
            input_root=input_root,
            output_root=output_root,
            relative=relative,
            namespace=args.namespace,
            action_mode=args.action_mode,
            encoder_config=encoder_config,
            task_title=task_title,
            source_description=source_description,
            stored_task=stored_task,
            schema=selected_schema,
            accepted=accepted_reports,
            skipped=skipped_reports,
            failed=failed_reports,
            conversion=conversion_result,
        )
        _atomic_json(report_output / REPORT_NAME, report)
        reports.append(report)

    if not discovered:
        rejected = True
    summary = {
        "version": 1,
        "generated_at_utc": _now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "namespace": args.namespace,
        "preflight_only": bool(args.preflight_only),
        "task_language": "english",
        "action_mode": args.action_mode,
        "encoder_config": encoder_config,
        "preflight_workers": preflight_workers,
        "require_source": bool(args.require_source),
        "discovered_episodes": len(discovered_all),
        "selected_episodes": len(discovered),
        "included_episodes": included,
        "tasks": [
            {
                "task_relative_path": item["task_relative_path"],
                "report": str(output_root / item["task_relative_path"] / REPORT_NAME),
                "counts": item["counts"],
            }
            for item in reports
        ],
        "counts": {
            "tasks": len(reports),
            "accepted": sum(item["counts"]["accepted"] for item in reports),
            "skipped": sum(item["counts"]["skipped"] for item in reports),
            "failed": sum(item["counts"]["failed"] for item in reports),
        },
        "status": "completed_with_rejections" if rejected else "completed",
    }
    if not discovered:
        summary["errors"] = ["no_cleaned_episodes_found"]
    _atomic_json(output_root / SUMMARY_NAME, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if rejected else 0


def convert_merged(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else input_root / "lerobot_data"
    )
    if not input_root.is_dir():
        print(f"input root is not a directory: {input_root}", file=sys.stderr)
        return 2
    if not NAMESPACE_RE.fullmatch(args.namespace):
        print(f"invalid namespace: {args.namespace}", file=sys.stderr)
        return 2
    if input_root == output_root or output_root in input_root.parents:
        print("output root must not equal or contain the input root", file=sys.stderr)
        return 2

    encoder_config = {
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "encoder_preset": args.encoder_preset,
        "encoder_threads": args.encoder_threads,
        "video_encoding_mode": args.video_encoding_mode,
    }
    discovered_all = [
        path for path in discover_episodes(input_root)
        if output_root not in (path, *path.parents)
    ]
    if any(episode in output_root.parents for episode in discovered_all):
        print("output root must not be inside a cleaned Episode", file=sys.stderr)
        return 2
    try:
        discovered, included = _select_episodes(input_root, discovered_all, args.include_episode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    checked_episodes, preflight_workers = _validate_episodes(
        input_root,
        discovered,
        require_source=args.require_source,
        requested_workers=args.preflight_workers,
    )
    checked_episodes.sort(key=lambda item: natural_key(item.relative_path))
    valid = [item for item in checked_episodes if item.valid]
    skipped_reports = [_merged_episode_report(item) for item in checked_episodes if not item.valid]
    expected_key, state_error = _merged_stored_schema_key(output_root, args.action_mode)
    if state_error:
        accepted, schema_outliers, selected_schema = [], valid, None
    else:
        accepted, schema_outliers, selected_schema = _choose_merged_schema(valid, expected_key)
    for item in schema_outliers:
        report = _merged_episode_report(item)
        report["status"] = "skipped"
        report["reasons"] = [state_error or "schema_outlier"]
        skipped_reports.append(report)

    rejected = bool(skipped_reports)
    accepted_reports = [_merged_episode_report(item) for item in accepted]
    failed_reports: list[dict[str, Any]] = []
    conversion_result: Any = None
    if accepted and not args.preflight_only:
        conversion_episodes = accepted
        try:
            engine = _engine_import()
            sources, conversion_episodes, source_skipped = _load_merged_sources(engine, accepted)
            if source_skipped:
                rejected = True
                skipped_reports.extend(source_skipped)
            accepted_reports = [_merged_episode_report(item) for item in conversion_episodes]
            if not sources:
                conversion_episodes = []
                accepted_reports = []
            else:
                conversion_result = _run_merged_conversion(
                    engine, sources, input_root, output_root, args.namespace,
                    args.action_mode, encoder_config,
                )
                accepted_reports, failed_reports = _merged_conversion_reports(
                    conversion_result, conversion_episodes,
                )
                if failed_reports or getattr(conversion_result, "failed", ()):
                    rejected = True
        except Exception as exc:
            rejected = True
            reason = str(exc) or type(exc).__name__
            failed_reports = [
                {
                    **_merged_episode_report(item),
                    "status": "failed",
                    "reasons": [f"conversion_failed:{reason}"],
                }
                    for item in conversion_episodes
                ]
            accepted_reports = []

    task_titles = sorted({item.task_title for item in checked_episodes if item.task_title})
    stored_tasks = sorted({_task_instruction(item) for item in checked_episodes if item.task_title})
    warnings = sorted({
        warning
        for item in accepted_reports + skipped_reports
        for warning in item.get("warnings", [])
    })
    report = {
        "version": 1,
        "generated_at_utc": _now(),
        "command": "convert-merged",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "namespace": args.namespace,
        "task_language": "english",
        "action_mode": args.action_mode,
        "encoder_config": encoder_config,
        "preflight_workers": preflight_workers,
        "require_source": bool(args.require_source),
        "preflight_only": bool(args.preflight_only),
        "discovered_episodes": len(discovered_all),
        "selected_episodes": len(discovered),
        "included_episodes": included,
        "schema": selected_schema,
        "task_titles": task_titles,
        "stored_tasks": stored_tasks,
        "accepted": accepted_reports,
        "skipped": skipped_reports,
        "failed": failed_reports,
        "freshness_warnings": warnings,
        "counts": {
            "tasks": len(task_titles),
            "accepted": len(accepted_reports),
            "skipped": len(skipped_reports),
            "failed": len(failed_reports),
        },
    }
    if conversion_result is not None:
        report["conversion"] = _conversion_payload(conversion_result)
    _atomic_json(output_root / REPORT_NAME, report)

    if not discovered:
        rejected = True
    summary = {
        "version": 1,
        "generated_at_utc": _now(),
        "command": "convert-merged",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "namespace": args.namespace,
        "preflight_only": bool(args.preflight_only),
        "task_language": "english",
        "action_mode": args.action_mode,
        "encoder_config": encoder_config,
        "preflight_workers": preflight_workers,
        "require_source": bool(args.require_source),
        "discovered_episodes": len(discovered_all),
        "selected_episodes": len(discovered),
        "included_episodes": included,
        "report": str(output_root / REPORT_NAME),
        "counts": report["counts"],
        "status": "completed_with_rejections" if rejected else "completed",
    }
    if not discovered:
        summary["errors"] = ["no_cleaned_episodes_found"]
    _atomic_json(output_root / SUMMARY_NAME, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if rejected else 0


def promote_aux_videos(args: argparse.Namespace) -> int:
    try:
        engine = _engine_import()
        streams = [stream for group in args.streams for stream in group]
        consumers = [Path(path) for path in args.consumer_root] if args.consumer_root else None
        result = engine.promote_aux_videos(
            Path(args.owner_root), streams, consumer_roots=consumers,
        )
    except Exception as exc:
        print(f"promotion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_conversion_payload(result), ensure_ascii=False))
    return 0


def _encoder_preset(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_conversion_arguments(command: argparse.ArgumentParser, *, output_required: bool) -> None:
    command.add_argument("--input-root", required=True, help="cleaned data root")
    command.add_argument(
        "--output-root", required=output_required,
        help=(
            "LeRobot output root"
            if output_required
            else "merged LeRobot output root (default: INPUT_ROOT/lerobot_data)"
        ),
    )
    command.add_argument("--namespace", default="autolife", help="LeRobot repository namespace")
    command.add_argument("--preflight-only", action="store_true", help="validate and report without importing LeRobot")
    command.add_argument(
        "--preflight-workers", type=_positive_int, default=max(1, int(RESOURCES["budget"])),
        help="parallel Episode validators (default: CPU budget; capped by the CPU budget)",
    )
    command.add_argument(
        "--action-mode", choices=("both", *ACTION_MODES), default="both",
        help="action dataset(s) to create; body_joint_eef alone requires an existing owner",
    )
    command.add_argument(
        "--include-episode", action="append", default=[], metavar="RELATIVE_PATH",
        help="convert only this exact Episode path relative to input root; repeat for multiple Episodes",
    )
    command.add_argument("--video-codec", choices=("h264", "libsvtav1"), default="libsvtav1")
    command.add_argument("--video-crf", type=int, default=23)
    command.add_argument("--encoder-preset", type=_encoder_preset, default=8)
    command.add_argument("--encoder-threads", type=int, default=1)
    command.add_argument(
        "--video-encoding-mode", choices=("sequential", "parallel", "streaming"),
        default="sequential",
    )
    command.add_argument("--require-source", action="store_true", help="reject Episodes whose original source cannot be verified")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-autopro-light")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("convert", help="incrementally convert cleaned Episodes to LeRobot v3")
    _add_conversion_arguments(command, output_required=True)
    command.set_defaults(handler=convert)
    merged = commands.add_parser(
        "convert-merged", help="incrementally merge cleaned Episodes into shared LeRobot datasets",
    )
    _add_conversion_arguments(merged, output_required=False)
    merged.set_defaults(handler=convert_merged)
    promote = commands.add_parser(
        "promote-aux-videos", help="promote indexed auxiliary streams to standard LeRobot video features",
    )
    promote.add_argument("--owner-root", required=True, help="whole_body_joint task dataset root")
    promote.add_argument(
        "--stream", "--streams", dest="streams", action="append", nargs="+", required=True,
        help="auxiliary stream name(s) to promote",
    )
    promote.add_argument(
        "--consumer-root", action="append", default=[],
        help="hybrid task root to update; defaults to the paired body_joint_eef sibling",
    )
    promote.set_defaults(handler=promote_aux_videos)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
