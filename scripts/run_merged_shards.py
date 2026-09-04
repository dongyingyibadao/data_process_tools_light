from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _partition_contiguous(episodes: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    if count < 1 or count > len(episodes):
        raise ValueError("shard count must be between 1 and the accepted episode count")
    frames = [int(item["frames"]) for item in episodes]
    if any(value <= 0 for value in frames):
        raise ValueError("accepted episode frame counts must be positive")

    def groups_needed(limit: int) -> int:
        groups = 1
        current = 0
        for value in frames:
            if current and current + value > limit:
                groups += 1
                current = 0
            current += value
        return groups

    lower, upper = max(frames), sum(frames)
    while lower < upper:
        middle = (lower + upper) // 2
        if groups_needed(middle) <= count:
            upper = middle
        else:
            lower = middle + 1
    limit = lower

    shards: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    end = len(episodes)
    for shard_index in range(count - 1, 0, -1):
        start = end
        total = 0
        while start > shard_index and total + frames[start - 1] <= limit:
            start -= 1
            total += frames[start]
        shards[shard_index] = episodes[start:end]
        end = start
    shards[0] = episodes[:end]
    return shards


def _quota_cpu_count() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(0.01, int(quota) / int(period))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(0.01, quota / period)
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return None


def _affinity_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def _spread_cpu_ids(cpu_ids: Sequence[int], count: int) -> list[int]:
    """Select CPUs across the full affinity range instead of one NUMA prefix."""

    if count < 1 or count > len(cpu_ids):
        raise ValueError("CPU count must be between 1 and the affinity CPU count")
    if count == len(cpu_ids):
        return list(cpu_ids)
    if count == 1:
        return [cpu_ids[0]]
    return [cpu_ids[index * (len(cpu_ids) - 1) // (count - 1)] for index in range(count)]


def _effective_cpu_ids(requested: int | None = None) -> tuple[list[int], dict[str, Any]]:
    affinity = _affinity_cpu_ids()
    quota = _quota_cpu_count()
    available = min(len(affinity), math.floor(quota) if quota is not None else len(affinity))
    available = max(1, available)
    count = available if requested is None else requested
    if count < 1 or count > available:
        raise ValueError(f"--cpus must be between 1 and the effective CPU budget ({available})")
    selected = _spread_cpu_ids(affinity, count)
    return selected, {
        "affinity_cpu_ids": affinity,
        "affinity_count": len(affinity),
        "quota_cpus": quota,
        "effective_cpu_budget": available,
        "selected_cpu_ids": selected,
    }


def _cpu_groups(cpu_ids: Sequence[int], shard_count: int) -> list[list[int]]:
    if len(cpu_ids) < shard_count:
        raise ValueError("CPU count must be at least the shard count")
    base, remainder = divmod(len(cpu_ids), shard_count)
    groups: list[list[int]] = []
    start = 0
    for index in range(shard_count):
        size = base + (1 if index < remainder else 0)
        groups.append(list(cpu_ids[start:start + size]))
        start += size
    return groups


def _parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    for raw_part in value.strip().split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start, end = int(raw_start), int(raw_end)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range: {part!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU id: {part!r}")
            cpus.add(cpu)
    return sorted(cpus)


def _numa_cpu_groups(
    cpu_ids: Sequence[int],
    topology_root: Path = Path("/sys/devices/system/node"),
) -> list[list[int]]:
    selected = set(cpu_ids)
    if not selected:
        raise ValueError("at least one effective CPU is required")
    node_paths = sorted(
        (
            path
            for path in topology_root.glob("node[0-9]*")
            if path.name.removeprefix("node").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("node")),
    )
    groups: list[list[int]] = []
    covered: set[int] = set()
    for node_path in node_paths:
        cpulist_path = node_path / "cpulist"
        if not cpulist_path.is_file():
            continue
        node_cpus = selected.intersection(_parse_cpu_list(cpulist_path.read_text(encoding="utf-8")))
        if node_cpus:
            groups.append(sorted(node_cpus))
            covered.update(node_cpus)
    if not groups:
        raise RuntimeError(f"no NUMA CPU topology found below {topology_root}")
    if covered != selected:
        missing = sorted(selected - covered)
        raise RuntimeError(f"effective CPUs are missing from NUMA topology: {missing}")
    return groups


def _cpu_assignments(
    cpu_ids: Sequence[int],
    shard_count: int,
    binding: str,
    *,
    topology_root: Path = Path("/sys/devices/system/node"),
) -> list[list[int]]:
    if shard_count < 1:
        raise ValueError("shard count must be at least 1")
    if binding == "exclusive":
        return _cpu_groups(cpu_ids, shard_count)
    if binding == "global-shared":
        if not cpu_ids:
            raise ValueError("at least one effective CPU is required")
        return [list(cpu_ids) for _ in range(shard_count)]
    if binding == "numa-shared":
        nodes = _numa_cpu_groups(cpu_ids, topology_root)
        return [
            list(nodes[min(index * len(nodes) // shard_count, len(nodes) - 1)])
            for index in range(shard_count)
        ]
    raise ValueError(f"unsupported CPU binding mode: {binding}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _normalized_relative_path(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} has no source_relative_path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{context} has invalid source_relative_path {value!r}")
    return relative.as_posix()


def _expected_owner_config(
    *, encoder_threads: int, video_workers: int, video_encoding_mode: str,
) -> dict[str, Any]:
    from lightworkbench.lerobot_converter import ConverterConfig, WHOLE_BODY_JOINT

    return ConverterConfig(
        video_codec="h264",
        video_crf=18,
        encoder_preset="fast",
        encoder_threads=encoder_threads,
        encoder_queue_maxsize=30,
        video_encoding_mode=video_encoding_mode,
        video_workers=video_workers,
    ).state_value(WHOLE_BODY_JOINT)


def _schema_compatibility_key(schema: Any) -> str | None:
    if not isinstance(schema, dict) or not isinstance(schema.get("videos"), dict):
        return None
    try:
        fps = round(float(schema.get("fps") or 0), 6)
    except (TypeError, ValueError):
        return None
    videos: dict[str, dict[str, Any]] = {}
    for name, value in schema["videos"].items():
        if not isinstance(value, dict):
            return None
        videos[str(name)] = {
            "width": value.get("width"),
            "height": value.get("height"),
            "is_depth": bool(value.get("is_depth")),
        }
    return json.dumps({"fps": fps, "videos": videos}, sort_keys=True, separators=(",", ":"))


def _filter_existing_episodes(
    episodes: Sequence[dict[str, Any]],
    *,
    input_root: Path,
    existing_owner: Path,
    expected_config: dict[str, Any],
    expected_schema: Any,
) -> tuple[list[dict[str, Any]], int]:
    resolved_owner = existing_owner.resolve()
    direct_state = resolved_owner / "conversion_state.json"
    bundled_state = resolved_owner / "whole_body_joint" / "conversion_state.json"
    state_path = (
        bundled_state
        if bundled_state.is_file()
        else direct_state
    )
    if not state_path.is_file():
        raise ValueError(f"--existing-owner has no whole_body_joint conversion state: {state_path}")
    state = _read_json(state_path)
    entries = state.get("episodes")
    if (
        state.get("action_mode") != "whole_body_joint"
        or state.get("dataset_layout") != "merged"
        or not isinstance(entries, list)
    ):
        raise ValueError(f"{state_path}: not a merged whole_body_joint conversion state")
    if state.get("pending_episode") is not None:
        raise ValueError(f"{state_path}: existing owner has an uncommitted pending episode")
    if state.get("conversion_config") != expected_config:
        raise ValueError(f"{state_path}: conversion config is incompatible with shard run parameters")
    report_schema_key = _schema_compatibility_key(expected_schema)
    if report_schema_key is None or _schema_compatibility_key(state.get("schema")) != report_schema_key:
        raise ValueError(f"{state_path}: schema is incompatible with the preflight report")

    fallback_root = state.get("source_root")
    committed: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{state_path}: episode {index} is not an object")
        source_root = entry.get("source_root") or fallback_root
        if not isinstance(source_root, str) or not source_root:
            raise ValueError(f"{state_path}: episode {index} has no source_root")
        relative = _normalized_relative_path(
            entry.get("source_relative_path"), context=f"{state_path}: episode {index}",
        )
        identity = (str(Path(source_root).expanduser().resolve()), relative)
        if identity in committed:
            raise ValueError(f"{state_path}: duplicate committed source identity {identity!r}")
        committed.add(identity)

    resolved_input_root = str(input_root.resolve())
    pending: list[dict[str, Any]] = []
    excluded = 0
    for index, episode in enumerate(episodes):
        relative = _normalized_relative_path(
            episode.get("source_relative_path"), context=f"preflight accepted episode {index}",
        )
        if (resolved_input_root, relative) in committed:
            excluded += 1
        else:
            pending.append(episode)
    return pending, excluded


def _shard_status(
    root: Path,
    selected: Sequence[str],
    expected_frames: int | None,
    *,
    source_root: Path,
    source_signatures: Sequence[str],
    expected_config: dict[str, Any],
    expected_schema: Any,
) -> tuple[bool, dict[str, Any]]:
    report_path = root / "conversion_report.json"
    state_path = root / "whole_body_joint" / "conversion_state.json"
    if not report_path.is_file() or not state_path.is_file():
        return False, {"reason": "report_or_state_missing"}
    report = _read_json(report_path)
    state = _read_json(state_path)
    accepted = [str(item.get("source_relative_path")) for item in report.get("accepted", [])]
    report_signatures = [str(item.get("source_signature")) for item in report.get("accepted", [])]
    skipped = [str(item.get("source_relative_path")) for item in report.get("skipped", [])]
    failed = report.get("failed", [])
    entries = state.get("episodes", [])
    state_paths = [str(item.get("source_relative_path")) for item in entries]
    resolved_source_root = str(source_root.resolve())
    state_source_roots = [
        str(Path(item.get("source_root") or state.get("source_root", "")).expanduser().resolve())
        for item in entries
    ]
    indices = [item.get("lerobot_episode_index") for item in entries]
    committed_frames = sum(int(item.get("output_frames") or 0) for item in entries)
    complete = (
        report.get("preflight_only") is False
        and report.get("action_mode") == "whole_body_joint"
        and str(Path(report.get("input_root", "")).expanduser().resolve()) == resolved_source_root
        and not failed
        and accepted == list(selected)
        and report_signatures == list(source_signatures)
        and not skipped
        and state.get("pending_episode") is None
        and state_paths == accepted
        and state_source_roots == [resolved_source_root] * len(entries)
        and state.get("conversion_config") == expected_config
        and _schema_compatibility_key(state.get("schema")) == _schema_compatibility_key(expected_schema)
        and _schema_compatibility_key(report.get("schema")) == _schema_compatibility_key(expected_schema)
        and indices == list(range(len(entries)))
        and (expected_frames is None or committed_frames == expected_frames)
    )
    return complete, {
        "accepted": len(accepted),
        "skipped": len(skipped),
        "failed": len(failed),
        "committed": len(entries),
        "committed_frames": committed_frames,
        "reason": None if complete else "terminal_state_validation_failed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent convert-merged shards in parallel")
    parser.add_argument("--report", type=Path, required=True, help="successful preflight conversion_report.json")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--existing-owner",
        type=Path,
        default=None,
        help="exclude episodes already committed in this merged owner dataset",
    )
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument(
        "--cpus", type=int, default=None,
        help="effective CPUs to divide across shards (default: min(affinity, cgroup quota))",
    )
    parser.add_argument(
        "--cpu-binding",
        choices=("exclusive", "numa-shared", "global-shared"),
        default="exclusive",
        help="CPU affinity policy for shard processes (default: exclusive)",
    )
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--video-workers", type=int, default=3)
    parser.add_argument("--preflight-workers", type=int, default=6)
    parser.add_argument(
        "--video-encoding-mode", choices=("sequential", "parallel", "streaming"), default="parallel",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _read_json(args.report.resolve())
    if not report.get("preflight_only") or report.get("failed"):
        raise ValueError("--report must be a completed preflight report without failed episodes")
    input_root = args.input_root.resolve()
    report_input_root = report.get("input_root")
    if (
        not isinstance(report_input_root, str)
        or Path(report_input_root).expanduser().resolve() != input_root
    ):
        raise ValueError("--report input_root differs from --input-root")
    accepted = list(report.get("accepted", []))
    if not accepted:
        raise ValueError("preflight report has no accepted episodes")
    if any(not isinstance(item.get("source_signature"), str) for item in accepted):
        raise ValueError("preflight accepted episode has no source_signature")
    expected_config = _expected_owner_config(
        encoder_threads=args.encoder_threads,
        video_workers=args.video_workers,
        video_encoding_mode=args.video_encoding_mode,
    )
    expected_schema = report.get("schema")
    episodes = accepted
    excluded_existing = 0
    existing_owner = args.existing_owner.resolve() if args.existing_owner is not None else None
    if existing_owner is not None:
        episodes, excluded_existing = _filter_existing_episodes(
            accepted,
            input_root=input_root,
            existing_owner=existing_owner,
            expected_config=expected_config,
            expected_schema=expected_schema,
        )
        if args.shards < 1:
            raise ValueError("shard count must be at least 1")
        shard_count = min(args.shards, len(episodes))
        shards = _partition_contiguous(episodes, shard_count) if shard_count else []
    else:
        shard_count = args.shards
        shards = _partition_contiguous(episodes, shard_count)
    cpu_ids, cpu_resources = _effective_cpu_ids(args.cpus if shard_count else None)
    cpu_groups = _cpu_assignments(cpu_ids, shard_count, args.cpu_binding) if shard_count else []
    taskset = shutil.which("taskset") if shard_count else None
    if shard_count and taskset is None:
        raise RuntimeError("taskset is required for bounded parallel shard conversion")

    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_root": str(input_root),
        "preflight_report": str(args.report.resolve()),
        "existing_owner": str(existing_owner) if existing_owner is not None else None,
        "excluded_existing": excluded_existing,
        "pending": len(episodes),
        "cpu_resources": cpu_resources,
        "cpu_binding": args.cpu_binding,
        "encoder_threads": args.encoder_threads,
        "video_workers": args.video_workers,
        "video_encoding_mode": args.video_encoding_mode,
        "shards": [
            {
                "index": index,
                "output_root": str(work_root / f"shard-{index:02d}"),
                "cpus": cpus,
                "episodes": [str(item["source_relative_path"]) for item in items],
                "source_signatures": [str(item["source_signature"]) for item in items],
                "frames": sum(int(item["frames"]) for item in items),
            }
            for index, (items, cpus) in enumerate(zip(shards, cpu_groups, strict=True))
        ],
    }
    _atomic_json(work_root / "shard_plan.json", plan)

    running: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OMP_THREAD_LIMIT": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENCV_FOR_THREADS_NUM": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "DATA_AUTOPRO_AUX_THREADS": "1",
    })
    child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    launcher_started_at = time.time()
    launcher_started_monotonic = time.monotonic()
    for item in plan["shards"]:
        index = int(item["index"])
        output_root = Path(str(item["output_root"]))
        selected = list(item["episodes"])
        complete, status = _shard_status(
            output_root,
            selected,
            int(item["frames"]),
            source_root=input_root,
            source_signatures=item["source_signatures"],
            expected_config=expected_config,
            expected_schema=expected_schema,
        )
        if complete:
            results.append({
                "index": index, "returncode": 0, "reused": True,
                "episodes": len(selected), "frames": int(item["frames"]),
                "cpus": list(item["cpus"]), "wall_seconds": 0.0, **status,
            })
            continue
        command = [
            str(taskset),
            "-c",
            ",".join(str(cpu) for cpu in item["cpus"]),
            args.python,
            "-m",
            "lightworkbench.cli",
            "convert-merged",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--action-mode",
            "whole_body_joint",
            "--video-codec",
            "h264",
            "--video-crf",
            "18",
            "--encoder-preset",
            "fast",
            "--encoder-threads",
            str(args.encoder_threads),
            "--video-encoding-mode",
            args.video_encoding_mode,
            "--video-workers",
            str(args.video_workers),
            "--preflight-workers",
            str(args.preflight_workers),
        ]
        for relative in selected:
            command.extend(("--include-episode", relative))
        log_path = work_root / f"shard-{index:02d}.log"
        log_handle = log_path.open("ab", buffering=0)
        started_monotonic = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        started_at = time.time()
        running[index] = {
            "process": process,
            "log_handle": log_handle,
            "output_root": output_root,
            "selected": selected,
            "frames": int(item["frames"]),
            "cpus": list(item["cpus"]),
            "started_monotonic": started_monotonic,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        }
        print(f"started shard {index:02d}: pid={process.pid} episodes={len(selected)} frames={item['frames']}", flush=True)

    failed = False
    while running:
        for index, item in list(running.items()):
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            finished_at = time.time()
            finished_monotonic = time.monotonic()
            item["log_handle"].close()
            complete, status = _shard_status(
                item["output_root"],
                item["selected"],
                item["frames"],
                source_root=input_root,
                source_signatures=plan["shards"][index]["source_signatures"],
                expected_config=expected_config,
                expected_schema=expected_schema,
            )
            results.append({
                "index": index,
                "returncode": returncode,
                "reused": False,
                "pid": process.pid,
                "episodes": len(item["selected"]),
                "frames": item["frames"],
                "cpus": item["cpus"],
                "started_at_utc": item["started_at_utc"],
                "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_at)),
                "wall_seconds": round(finished_monotonic - item["started_monotonic"], 6),
                **status,
            })
            failed = failed or returncode != 0 or not complete
            print(
                f"finished shard {index:02d}: rc={returncode} complete={complete} "
                f"committed={status.get('committed', 0)} skipped={status.get('skipped', 0)} "
                f"wall={finished_monotonic - item['started_monotonic']:.2f}s",
                flush=True,
            )
            del running[index]
        if running:
            time.sleep(0.1)
    results.sort(key=lambda item: int(item["index"]))
    launcher_finished_at = time.time()
    launcher_finished_monotonic = time.monotonic()
    child_usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    total_frames = sum(int(item["frames"]) for item in plan["shards"])
    wall_seconds = launcher_finished_monotonic - launcher_started_monotonic
    user_seconds = child_usage_after.ru_utime - child_usage_before.ru_utime
    system_seconds = child_usage_after.ru_stime - child_usage_before.ru_stime
    average_used_cores = (user_seconds + system_seconds) / wall_seconds if wall_seconds else 0.0
    effective_budget = int(cpu_resources["effective_cpu_budget"])
    _atomic_json(
        work_root / "shard_run_summary.json",
        {
            "version": 2,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(launcher_started_at)),
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(launcher_finished_at)),
            "wall_seconds": round(wall_seconds, 6),
            "user_seconds": round(user_seconds, 6),
            "system_seconds": round(system_seconds, 6),
            "average_used_cores": round(average_used_cores, 6),
            "cpu_budget_utilization_pct": round(average_used_cores / effective_budget * 100, 6),
            "max_rss_kib": child_usage_after.ru_maxrss,
            "frames": total_frames,
            "frames_per_second": round(total_frames / wall_seconds, 6) if wall_seconds else None,
            "cpu_resources": cpu_resources,
            "cpu_binding": args.cpu_binding,
            "encoder_threads": args.encoder_threads,
            "video_workers": args.video_workers,
            "video_encoding_mode": args.video_encoding_mode,
            "existing_owner": str(existing_owner) if existing_owner is not None else None,
            "excluded_existing": excluded_existing,
            "pending": len(episodes),
            "results": results,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
