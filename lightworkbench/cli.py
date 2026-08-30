from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter, OrderedDict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .validation import EpisodeValidation, discover_episodes, natural_key, schema_key, validate_episode


SUMMARY_NAME = "conversion_summary.json"
REPORT_NAME = "conversion_report.json"
STATE_NAME = "conversion_state.json"
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
            for name, value in videos.items() if isinstance(value, dict)
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
    task_output: Path,
    stored_task: str,
) -> tuple[list[EpisodeValidation], list[EpisodeValidation], dict[str, Any] | None, str | None]:
    if not episodes:
        return [], [], None, None
    state = _read_json(task_output / STATE_NAME)
    expected_key: str | None = None
    if state is not None:
        if not state:
            return [], episodes, None, "conversion_state_invalid"
        config = state.get("conversion_config")
        if state.get("version") != 2 or not isinstance(config, dict) or config.get("action_dim") != 20:
            return [], episodes, None, "incompatible_action_schema_requires_new_output_root"
        if state.get("stored_task") != stored_task:
            return [], episodes, None, "stored_task_language_or_text_conflict"
        expected_key = _stored_schema_key(state)
        if expected_key is None:
            return [], episodes, None, "conversion_state_schema_invalid"
    if expected_key is None:
        groups: OrderedDict[str, list[EpisodeValidation]] = OrderedDict()
        for episode in episodes:
            groups.setdefault(schema_key(episode.schema), []).append(episode)
        expected_key = max(groups, key=lambda key: len(groups[key]))
    accepted = [episode for episode in episodes if schema_key(episode.schema) == expected_key]
    outliers = [episode for episode in episodes if schema_key(episode.schema) != expected_key]
    selected_schema = accepted[0].schema if accepted else None
    return accepted, outliers, selected_schema, None


def _task_instruction(episode: EpisodeValidation, language: str) -> str:
    if language == "source":
        return episode.source_description
    return " ".join(episode.task_title.split("_"))


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
    language: str,
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
        "task_language": language,
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
    task_output: Path,
    namespace: str,
    task_title: str,
    stored_task: str,
    source_task: Path,
) -> Any:
    try:
        sources = [engine.load_source_episode(item.path, stored_task) for item in episodes]
        config = engine.ConverterConfig()
        repo_id = f"{namespace}/{task_title}"
        return engine.convert_task(
            sources,
            task_output,
            repo_id,
            source_task=source_task,
            config=config,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot conversion dependencies are unavailable; install "
            "data-autopro-tools-light[lerobot] in the documented Python 3.12 environment"
        ) from exc


def _conversion_reports(
    result: Any, episodes: list[EpisodeValidation]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures = {
        str(item.get("episode")): str(item.get("reason") or "unknown conversion failure")
        for item in (getattr(result, "failed", ()) or ())
        if isinstance(item, dict)
    }
    state = getattr(result, "state", {})
    converted = {
        str(item.get("source_episode_name"))
        for item in (state.get("episodes", []) if isinstance(state, dict) else [])
        if isinstance(item, dict)
    }
    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for episode in episodes:
        if episode.path.name in converted:
            succeeded.append(episode.report())
            continue
        reason = failures.get(episode.path.name, "not attempted after an earlier episode failed")
        failed.append({
            **episode.report(),
            "status": "failed",
            "reasons": [f"conversion_failed:{reason}"],
        })
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

    discovered = discover_episodes(input_root)
    grouped: OrderedDict[str, list[EpisodeValidation]] = OrderedDict()
    for episode in discovered:
        checked = validate_episode(input_root, episode, require_source=args.require_source)
        grouped.setdefault(checked.task_relative_path, []).append(checked)

    reports: list[dict[str, Any]] = []
    rejected = False
    engine: Any = None
    for relative, checked_episodes in grouped.items():
        checked_episodes.sort(key=lambda item: natural_key(item.relative_path))
        task_output = output_root / relative
        valid = [item for item in checked_episodes if item.valid]
        skipped_reports = [item.report() for item in checked_episodes if not item.valid]
        rejected = rejected or bool(skipped_reports)

        task_title = valid[0].task_title if valid else (checked_episodes[0].task_title if checked_episodes else "")
        source_description = valid[0].source_description if valid else (
            checked_episodes[0].source_description if checked_episodes else ""
        )
        metadata_outliers = [
            item for item in valid
            if item.task_title != task_title or item.source_description != source_description
        ]
        for item in metadata_outliers:
            report = item.report()
            report["status"] = "skipped"
            report["reasons"] = ["task_metadata_conflict"]
            skipped_reports.append(report)
        if metadata_outliers:
            rejected = True
            excluded = {item.relative_path for item in metadata_outliers}
            valid = [item for item in valid if item.relative_path not in excluded]

        stored_task = _task_instruction(valid[0], args.task_language) if valid else ""
        language_missing: list[EpisodeValidation] = []
        if args.task_language == "source":
            language_missing = [item for item in valid if not item.source_description]
            if language_missing:
                rejected = True
                for item in language_missing:
                    report = item.report()
                    report["status"] = "skipped"
                    report["reasons"] = ["source_description_missing"]
                    skipped_reports.append(report)
                missing = {item.relative_path for item in language_missing}
                valid = [item for item in valid if item.relative_path not in missing]

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

        accepted, schema_outliers, selected_schema, state_error = _choose_schema(valid, task_output, stored_task)
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
                    task_output,
                    args.namespace,
                    task_title,
                    stored_task,
                    accepted[0].path.parent,
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
            language=args.task_language,
            task_title=task_title,
            source_description=source_description,
            stored_task=stored_task,
            schema=selected_schema,
            accepted=accepted_reports,
            skipped=skipped_reports,
            failed=failed_reports,
            conversion=conversion_result,
        )
        _atomic_json(task_output / REPORT_NAME, report)
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
        "task_language": args.task_language,
        "require_source": bool(args.require_source),
        "discovered_episodes": len(discovered),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-autopro-light")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("convert", help="incrementally convert cleaned Episodes to LeRobot v3")
    command.add_argument("--input-root", required=True, help="cleaned data root")
    command.add_argument("--output-root", required=True, help="LeRobot output root")
    command.add_argument("--namespace", default="autolife", help="LeRobot repository namespace")
    command.add_argument("--preflight-only", action="store_true", help="validate and report without importing LeRobot")
    command.add_argument("--task-language", choices=("english", "source"), default="english")
    command.add_argument("--require-source", action="store_true", help="reject Episodes whose original source cannot be verified")
    command.set_defaults(handler=convert)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
