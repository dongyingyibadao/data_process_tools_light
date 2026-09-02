from __future__ import annotations

import json

import pytest

from scripts import run_merged_shards as runner


def test_partition_contiguous_preserves_order_and_balances_frames() -> None:
    episodes = [
        {"source_relative_path": f"task/episode_{index:06d}", "frames": frames}
        for index, frames in enumerate((100, 200, 100, 250, 150, 200))
    ]

    shards = runner._partition_contiguous(episodes, 3)

    assert [item for shard in shards for item in shard] == episodes
    assert all(shards)
    frame_counts = [sum(int(item["frames"]) for item in shard) for shard in shards]
    assert max(frame_counts) - min(frame_counts) <= 150


def test_partition_contiguous_minimizes_the_largest_shard() -> None:
    episodes = [{"frames": value} for value in (79, 17, 52, 50, 13, 61, 69)]

    shards = runner._partition_contiguous(episodes, 4)

    assert [sum(item["frames"] for item in shard) for shard in shards] == [96, 102, 74, 69]


@pytest.mark.parametrize("count", (0, 3))
def test_partition_rejects_invalid_shard_count(count: int) -> None:
    with pytest.raises(ValueError, match="shard count"):
        runner._partition_contiguous([{"frames": 1}, {"frames": 2}], count)


def test_effective_cpu_ids_use_real_affinity_and_full_cgroup_budget(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_affinity_cpu_ids", lambda: [3, 5, 7, 11, 13, 17])
    monkeypatch.setattr(runner, "_quota_cpu_count", lambda: 3.7)

    selected, resources = runner._effective_cpu_ids()

    assert selected == [3, 7, 17]
    assert resources["effective_cpu_budget"] == 3
    assert resources["affinity_count"] == 6
    assert resources["quota_cpus"] == 3.7


def test_requested_cpu_count_is_bounded_by_effective_budget(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_affinity_cpu_ids", lambda: [2, 4, 6, 8])
    monkeypatch.setattr(runner, "_quota_cpu_count", lambda: 3.0)

    selected, _ = runner._effective_cpu_ids(2)
    assert selected == [2, 8]

    try:
        runner._effective_cpu_ids(4)
    except ValueError as exc:
        assert "effective CPU budget (3)" in str(exc)
    else:
        raise AssertionError("CPU request above the cgroup budget was accepted")


def test_cpu_groups_preserve_non_contiguous_cpu_ids() -> None:
    groups = runner._cpu_groups([1, 4, 7, 10, 13], 2)
    assert groups == [[1, 4, 7], [10, 13]]


def test_parse_cpu_list_supports_ranges_gaps_and_duplicates() -> None:
    assert runner._parse_cpu_list("0-2, 5,7-8,2\n") == [0, 1, 2, 5, 7, 8]


def test_cpu_assignments_default_exclusive_preserves_existing_behavior() -> None:
    assignments = runner._cpu_assignments([1, 4, 7, 10, 13], 2, "exclusive")
    assert assignments == [[1, 4, 7], [10, 13]]


def test_cpu_binding_cli_defaults_to_exclusive() -> None:
    args = runner._parser().parse_args([
        "--report", "report.json",
        "--input-root", "input",
        "--work-root", "work",
    ])
    assert args.cpu_binding == "exclusive"


def test_cpu_assignments_global_shared_give_every_shard_all_effective_cpus() -> None:
    assignments = runner._cpu_assignments([1, 4, 7], 4, "global-shared")
    assert assignments == [[1, 4, 7], [1, 4, 7], [1, 4, 7], [1, 4, 7]]
    assert len({id(item) for item in assignments}) == 4


def test_cpu_assignments_numa_shared_intersect_effective_cpus(
    tmp_path,
) -> None:
    node0 = tmp_path / "node0"
    node1 = tmp_path / "node1"
    node0.mkdir()
    node1.mkdir()
    (node0 / "cpulist").write_text("0-7\n", encoding="utf-8")
    (node1 / "cpulist").write_text("8-15\n", encoding="utf-8")

    assignments = runner._cpu_assignments(
        [1, 4, 7, 10, 13], 4, "numa-shared", topology_root=tmp_path,
    )

    assert assignments == [
        [1, 4, 7],
        [1, 4, 7],
        [10, 13],
        [10, 13],
    ]


def test_numa_cpu_groups_reject_incomplete_topology(tmp_path) -> None:
    node0 = tmp_path / "node0"
    node0.mkdir()
    (node0 / "cpulist").write_text("0-3\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing from NUMA topology"):
        runner._numa_cpu_groups([1, 5], tmp_path)


def test_shard_status_requires_exact_episode_order_and_frame_count(tmp_path) -> None:
    root = tmp_path / "shard"
    owner = root / "whole_body_joint"
    owner.mkdir(parents=True)
    paths = ["task/episode_000001", "task/episode_000002"]
    (root / "conversion_report.json").write_text(
        json.dumps({
            "preflight_only": False,
            "action_mode": "whole_body_joint",
            "accepted": [{"source_relative_path": value} for value in paths],
            "skipped": [],
            "failed": [],
        }),
        encoding="utf-8",
    )
    state = {
        "episodes": [
            {"source_relative_path": value, "lerobot_episode_index": index, "output_frames": frames}
            for index, (value, frames) in enumerate(zip(paths, (10, 12), strict=True))
        ]
    }
    (owner / "conversion_state.json").write_text(json.dumps(state), encoding="utf-8")

    complete, status = runner._shard_status(root, paths, 22)
    assert complete
    assert status["committed_frames"] == 22
    assert not runner._shard_status(root, paths, 21)[0]
    assert not runner._shard_status(root, list(reversed(paths)), 22)[0]
