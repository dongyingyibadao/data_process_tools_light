from __future__ import annotations

import math
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
INTERNAL_NAMES = {
    ".lightworkbench-staging",
    ".lightworkbench-backups",
    ".lightworkbench-trash",
}


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
        return None
    return None


def resource_budget(apply: bool = True) -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    available = min(len(affinity), _quota_cpu_count() or len(affinity))
    budget = max(1, math.floor(available * 0.70))
    allowed = affinity[:budget]
    if apply and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, allowed)
    web_slots = min(2, budget)
    return {
        "affinityAvailable": len(affinity),
        "quotaAvailable": _quota_cpu_count(),
        "available": available,
        "budget": budget,
        "cpus": allowed,
        "webSlots": web_slots,
        "ffmpegSlots": max(1, budget - web_slots),
    }


RESOURCES = resource_budget()
IO_WORKERS = max(2, min(8, int(RESOURCES["budget"])))
