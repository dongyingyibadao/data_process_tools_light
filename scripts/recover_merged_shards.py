from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish interrupted initial shard staging roots for normal incremental recovery"
    )
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    work_root = args.work_root.expanduser().resolve()
    bundles = sorted(path for path in work_root.glob("shard-[0-9][0-9]") if path.is_dir())
    if not bundles:
        raise ValueError(f"no shard bundle directories found under {work_root}")

    for bundle in bundles:
        owner = bundle / "whole_body_joint"
        if owner.is_dir():
            print(f"{bundle.name}: owner already published")
            continue
        staging = list(bundle.glob(".whole_body_joint.staging-*"))
        if len(staging) != 1:
            raise ValueError(f"{bundle}: expected exactly one interrupted staging root, found {len(staging)}")
        root = staging[0]
        state = _read_object(root / "conversion_state.json")
        info = _read_object(root / "meta/info.json")
        episodes = state.get("episodes")
        pending = state.get("pending_episode")
        if not isinstance(episodes, list) or not isinstance(pending, dict):
            raise ValueError(f"{root}: interrupted state must contain episodes and pending_episode")
        known = len(episodes)
        total = int(info.get("total_episodes", -1))
        if total not in {known, known + 1}:
            raise ValueError(
                f"{root}: unsupported recovery boundary metadata={total}, committed_state={known}"
            )
        if pending.get("lerobot_episode_index") != known:
            raise ValueError(f"{root}: pending episode index is not the next contiguous index")
        os.replace(root, owner)
        print(
            f"{bundle.name}: published recovery root committed={known} "
            f"metadata={total} pending={pending.get('source_relative_path')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
