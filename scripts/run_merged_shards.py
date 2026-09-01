from __future__ import annotations

import argparse
import json
import os
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
    total_frames = sum(int(item["frames"]) for item in episodes)
    shards: list[list[dict[str, Any]]] = []
    start = 0
    consumed = 0
    for shard_index in range(count - 1):
        remaining_shards = count - shard_index
        target = (total_frames - consumed) / remaining_shards
        frames = 0
        end = start
        maximum_end = len(episodes) - (remaining_shards - 1)
        while end < maximum_end:
            candidate = int(episodes[end]["frames"])
            if end > start and abs(frames - target) <= abs(frames + candidate - target):
                break
            frames += candidate
            end += 1
        shards.append(episodes[start:end])
        start = end
        consumed += frames
    shards.append(episodes[start:])
    return shards


def _cpu_groups(cpu_count: int, shard_count: int) -> list[list[int]]:
    if cpu_count < shard_count:
        raise ValueError("cpu count must be at least the shard count")
    base, remainder = divmod(cpu_count, shard_count)
    groups: list[list[int]] = []
    start = 0
    for index in range(shard_count):
        size = base + (1 if index < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _shard_status(root: Path, selected: Sequence[str]) -> tuple[bool, dict[str, Any]]:
    report_path = root / "conversion_report.json"
    state_path = root / "whole_body_joint" / "conversion_state.json"
    if not report_path.is_file() or not state_path.is_file():
        return False, {"reason": "report_or_state_missing"}
    report = _read_json(report_path)
    state = _read_json(state_path)
    accepted = [str(item.get("source_relative_path")) for item in report.get("accepted", [])]
    skipped = [str(item.get("source_relative_path")) for item in report.get("skipped", [])]
    failed = report.get("failed", [])
    entries = state.get("episodes", [])
    state_paths = [str(item.get("source_relative_path")) for item in entries]
    indices = [item.get("lerobot_episode_index") for item in entries]
    complete = (
        report.get("preflight_only") is False
        and report.get("action_mode") == "whole_body_joint"
        and not failed
        and sorted(accepted + skipped) == sorted(selected)
        and state.get("pending_episode") is None
        and state_paths == accepted
        and indices == list(range(len(entries)))
    )
    return complete, {
        "accepted": len(accepted),
        "skipped": len(skipped),
        "failed": len(failed),
        "committed": len(entries),
        "reason": None if complete else "terminal_state_validation_failed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent convert-merged shards in parallel")
    parser.add_argument("--report", type=Path, required=True, help="successful preflight conversion_report.json")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--cpus", type=int, default=54, help="CPU ids starting at zero to divide across shards")
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--preflight-workers", type=int, default=6)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _read_json(args.report.resolve())
    if not report.get("preflight_only") or report.get("failed"):
        raise ValueError("--report must be a completed preflight report without failed episodes")
    episodes = list(report.get("accepted", []))
    if not episodes:
        raise ValueError("preflight report has no accepted episodes")
    shards = _partition_contiguous(episodes, args.shards)
    cpu_groups = _cpu_groups(args.cpus, args.shards)
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for bounded parallel shard conversion")

    input_root = args.input_root.resolve()
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_root": str(input_root),
        "preflight_report": str(args.report.resolve()),
        "shards": [
            {
                "index": index,
                "output_root": str(work_root / f"shard-{index:02d}"),
                "cpus": cpus,
                "episodes": [str(item["source_relative_path"]) for item in items],
                "frames": sum(int(item["frames"]) for item in items),
            }
            for index, (items, cpus) in enumerate(zip(shards, cpu_groups, strict=True))
        ],
    }
    _atomic_json(work_root / "shard_plan.json", plan)

    running: list[tuple[int, subprocess.Popen[bytes], Any, Path, list[str]]] = []
    results: list[dict[str, Any]] = []
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    for item in plan["shards"]:
        index = int(item["index"])
        output_root = Path(str(item["output_root"]))
        selected = list(item["episodes"])
        complete, status = _shard_status(output_root, selected)
        if complete:
            results.append({"index": index, "returncode": 0, "reused": True, **status})
            continue
        command = [
            taskset,
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
            "parallel",
            "--preflight-workers",
            str(args.preflight_workers),
        ]
        for relative in selected:
            command.extend(("--include-episode", relative))
        log_path = work_root / f"shard-{index:02d}.log"
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        running.append((index, process, log_handle, output_root, selected))
        print(f"started shard {index:02d}: pid={process.pid} episodes={len(selected)} frames={item['frames']}", flush=True)

    failed = False
    for index, process, log_handle, output_root, selected in running:
        returncode = process.wait()
        log_handle.close()
        complete, status = _shard_status(output_root, selected)
        results.append({"index": index, "returncode": returncode, "reused": False, **status})
        failed = failed or not complete
        print(
            f"finished shard {index:02d}: rc={returncode} complete={complete} "
            f"committed={status.get('committed', 0)} skipped={status.get('skipped', 0)}",
            flush=True,
        )
    results.sort(key=lambda item: int(item["index"]))
    _atomic_json(work_root / "shard_run_summary.json", {"version": 1, "results": results})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
