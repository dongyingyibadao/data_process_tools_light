from pathlib import Path

from lightworkbench.shard_merge import merge_whole_body_joint_shards


WORK_ROOT = Path(
    "/inspire/qb-ilm/project/robot-decision/public/demo2/"
    "lerobot_data_08_29_shards_20260901"
)
OUTPUT_ROOT = Path(
    "/inspire/qb-ilm/project/robot-decision/public/demo2/"
    "lerobot_data_08_29/whole_body_joint"
)


def main() -> None:
    shard_roots = [WORK_ROOT / f"shard-{index:02d}" for index in range(6)]
    print(f"merging {len(shard_roots)} shards into {OUTPUT_ROOT}", flush=True)
    result = merge_whole_body_joint_shards(shard_roots, OUTPUT_ROOT)
    print(f"merge completed: {result}", flush=True)


if __name__ == "__main__":
    main()
