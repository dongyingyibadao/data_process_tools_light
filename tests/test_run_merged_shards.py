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
    assert args.existing_owner is None


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
    source_root = tmp_path / "source"
    source_root.mkdir()
    owner = root / "whole_body_joint"
    owner.mkdir(parents=True)
    paths = ["task/episode_000001", "task/episode_000002"]
    signatures = ["signature-1", "signature-2"]
    schema = {"fps": 30, "videos": {}}
    (root / "conversion_report.json").write_text(
        json.dumps({
            "preflight_only": False,
            "action_mode": "whole_body_joint",
            "input_root": str(source_root),
            "schema": schema,
            "accepted": [
                {"source_relative_path": value, "source_signature": signature}
                for value, signature in zip(paths, signatures, strict=True)
            ],
            "skipped": [],
            "failed": [],
        }),
        encoding="utf-8",
    )
    state = {
        "source_root": str(source_root),
        "conversion_config": _owner_config(),
        "schema": schema,
        "episodes": [
            {
                "source_relative_path": value,
                "source_signature": {"digest": signature},
                "lerobot_episode_index": index,
                "output_frames": frames,
            }
            for index, (value, signature, frames) in enumerate(
                zip(paths, signatures, (10, 12), strict=True)
            )
        ]
    }
    (owner / "conversion_state.json").write_text(json.dumps(state), encoding="utf-8")

    kwargs = {
        "source_root": source_root,
        "source_signatures": signatures,
        "expected_config": _owner_config(),
        "expected_schema": schema,
    }
    complete, status = runner._shard_status(root, paths, 22, **kwargs)
    assert complete
    assert status["committed_frames"] == 22
    assert not runner._shard_status(root, paths, 21, **kwargs)[0]
    assert not runner._shard_status(root, list(reversed(paths)), 22, **kwargs)[0]
    assert not runner._shard_status(
        root, paths, 22, **{**kwargs, "source_root": tmp_path / "other-source"},
    )[0]
    assert not runner._shard_status(
        root, paths, 22, **{**kwargs, "source_signatures": ["wrong", signatures[1]]},
    )[0]


def _owner_config(*, encoder_threads: int = 4, video_encoding_mode: str = "parallel") -> dict:
    return runner._expected_owner_config(
        encoder_threads=encoder_threads,
        video_workers=3,
        video_encoding_mode=video_encoding_mode,
    )


def _write_existing_owner(
    root, *, source_root, episodes, schema, config=None, pending_episode=None,
) -> None:
    owner = root / "whole_body_joint"
    owner.mkdir(parents=True)
    (owner / "conversion_state.json").write_text(
        json.dumps({
            "version": 3,
            "action_mode": "whole_body_joint",
            "dataset_layout": "merged",
            "source_root": str(source_root),
            "conversion_config": config or _owner_config(),
            "schema": schema,
            "episodes": episodes,
            "pending_episode": pending_episode,
        }),
        encoding="utf-8",
    )


def test_filter_existing_episodes_uses_composite_source_identity_and_legacy_fallback(
    tmp_path,
) -> None:
    input_root = tmp_path / "input"
    other_root = tmp_path / "other"
    input_root.mkdir()
    other_root.mkdir()
    report_schema = {
        "fps": 30.0,
        "videos": {"rgb": {"width": 64, "height": 48, "is_depth": False}},
    }
    state_schema = {
        "fps": 30,
        "videos": {"rgb": {
            "key": "observation.images.rgb",
            "width": 64,
            "height": 48,
            "channels": 3,
            "is_depth": False,
        }},
    }
    existing_owner = tmp_path / "existing"
    _write_existing_owner(
        existing_owner,
        source_root=input_root,
        schema=state_schema,
        episodes=[
            {"source_relative_path": "task/episode_000001"},
            {"source_root": str(input_root), "source_relative_path": "task/episode_000002"},
            {"source_root": str(other_root), "source_relative_path": "task/episode_000003"},
        ],
    )
    accepted = [
        {"source_relative_path": f"task/episode_{index:06d}", "frames": 10}
        for index in range(1, 5)
    ]

    pending, excluded = runner._filter_existing_episodes(
        accepted,
        input_root=input_root,
        existing_owner=existing_owner,
        expected_config=_owner_config(),
        expected_schema=report_schema,
    )

    assert excluded == 2
    assert [item["source_relative_path"] for item in pending] == [
        "task/episode_000003",
        "task/episode_000004",
    ]


@pytest.mark.parametrize("owner_form", ("bundle", "direct"))
def test_filter_existing_owner_accepts_bundle_or_direct_owner_path(
    tmp_path, owner_form,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    schema = {"fps": 30, "videos": {}}
    bundle = tmp_path / "existing"
    _write_existing_owner(
        bundle,
        source_root=input_root,
        schema=schema,
        episodes=[{"source_relative_path": "task/episode_000001"}],
    )
    existing_owner = bundle if owner_form == "bundle" else bundle / "whole_body_joint"

    pending, excluded = runner._filter_existing_episodes(
        [{"source_relative_path": "task/episode_000001", "frames": 10}],
        input_root=input_root,
        existing_owner=existing_owner,
        expected_config=_owner_config(),
        expected_schema=schema,
    )

    assert pending == []
    assert excluded == 1


@pytest.mark.parametrize(
    ("config", "schema", "message"),
    [
        (_owner_config(encoder_threads=2), {"fps": 30}, "conversion config"),
        (_owner_config(), {"fps": 25}, "schema"),
    ],
)
def test_filter_existing_episodes_rejects_incompatible_owner_before_work(
    tmp_path, config, schema, message,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    report_schema = {"fps": 30}
    existing_owner = tmp_path / "existing"
    _write_existing_owner(
        existing_owner,
        source_root=input_root,
        schema=schema,
        config=config,
        episodes=[],
    )

    with pytest.raises(ValueError, match=message):
        runner._filter_existing_episodes(
            [{"source_relative_path": "task/episode_000001", "frames": 10}],
            input_root=input_root,
            existing_owner=existing_owner,
            expected_config=_owner_config(),
            expected_schema=report_schema,
        )


def test_main_rejects_incompatible_owner_before_launching_process(
    tmp_path, monkeypatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    schema = {"fps": 30, "videos": {}}
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        json.dumps({
            "preflight_only": True,
            "input_root": str(input_root),
            "failed": [],
            "accepted": [{
                "source_relative_path": "task/episode_000001",
                "source_signature": "signature-1",
                "frames": 10,
            }],
            "schema": schema,
        }),
        encoding="utf-8",
    )
    existing_owner = tmp_path / "existing"
    _write_existing_owner(
        existing_owner,
        source_root=input_root,
        schema=schema,
        config=_owner_config(encoder_threads=2),
        episodes=[],
    )
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("incompatible owner launched a shard process"),
    )
    work_root = tmp_path / "work"

    with pytest.raises(ValueError, match="conversion config"):
        runner.main([
            "--report", str(report_path),
            "--input-root", str(input_root),
            "--work-root", str(work_root),
            "--existing-owner", str(existing_owner),
        ])

    assert not work_root.exists()


def test_all_existing_episodes_write_zero_pending_plan_and_summary(
    tmp_path, monkeypatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    schema = {"fps": 30, "videos": {}}
    accepted = [
        {
            "source_relative_path": f"task/episode_{index:06d}",
            "source_signature": f"signature-{index}",
            "frames": frames,
        }
        for index, frames in enumerate((10, 12), start=1)
    ]
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        json.dumps({
            "preflight_only": True,
            "input_root": str(input_root),
            "failed": [],
            "accepted": accepted,
            "schema": schema,
        }),
        encoding="utf-8",
    )
    existing_owner = tmp_path / "existing"
    _write_existing_owner(
        existing_owner,
        source_root=input_root,
        schema=schema,
        episodes=[{"source_relative_path": item["source_relative_path"]} for item in accepted],
    )
    cpu_requests = []

    def effective_cpu_ids(requested=None):
        cpu_requests.append(requested)
        return [1], {"effective_cpu_budget": 1}

    monkeypatch.setattr(runner, "_effective_cpu_ids", effective_cpu_ids)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("no shard process should be launched"),
    )
    work_root = tmp_path / "work"

    result = runner.main([
        "--report", str(report_path),
        "--input-root", str(input_root),
        "--work-root", str(work_root),
        "--existing-owner", str(existing_owner),
        "--cpus", "999",
    ])

    assert result == 0
    plan = json.loads((work_root / "shard_plan.json").read_text(encoding="utf-8"))
    summary = json.loads((work_root / "shard_run_summary.json").read_text(encoding="utf-8"))
    assert plan["excluded_existing"] == summary["excluded_existing"] == 2
    assert plan["pending"] == summary["pending"] == 0
    assert plan["shards"] == []
    assert summary["frames"] == 0
    assert summary["results"] == []
    assert cpu_requests == [None]


def test_second_source_root_launches_only_new_episodes_and_shrinks_shards(
    tmp_path, monkeypatch,
) -> None:
    first_root = tmp_path / "source-a"
    second_root = tmp_path / "source-b"
    first_root.mkdir()
    second_root.mkdir()
    schema = {"fps": 30, "videos": {}}
    bundle = tmp_path / "existing"
    _write_existing_owner(
        bundle,
        source_root=first_root,
        schema=schema,
        episodes=[{"source_relative_path": "task/episode_000001"}],
    )
    accepted = [
        {
            "source_relative_path": "task/episode_000001",
            "source_signature": "signature-1",
            "frames": 10,
        },
        {
            "source_relative_path": "other/episode_000001",
            "source_signature": "signature-2",
            "frames": 12,
        },
    ]
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        json.dumps({
            "preflight_only": True,
            "input_root": str(second_root),
            "failed": [],
            "accepted": accepted,
            "schema": schema,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_effective_cpu_ids", lambda requested=None: ([1, 2], {
        "effective_cpu_budget": 2,
    }))
    status_calls = {}

    def shard_status(root, selected, expected_frames=None, **kwargs):
        calls = status_calls.get(root, 0)
        status_calls[root] = calls + 1
        return calls > 0, {
            "accepted": len(selected), "skipped": 0, "failed": 0,
            "committed": len(selected), "committed_frames": expected_frames,
            "reason": None if calls > 0 else "report_or_state_missing",
        }

    commands = []

    class CompletedProcess:
        pid = 12345

        def poll(self):
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        return CompletedProcess()

    monkeypatch.setattr(runner, "_shard_status", shard_status)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/taskset")
    work_root = tmp_path / "work"

    result = runner.main([
        "--report", str(report_path),
        "--input-root", str(second_root),
        "--work-root", str(work_root),
        "--existing-owner", str(bundle),
        "--shards", "6",
    ])

    assert result == 0
    plan = json.loads((work_root / "shard_plan.json").read_text(encoding="utf-8"))
    assert plan["excluded_existing"] == 0
    assert plan["pending"] == 2
    assert len(plan["shards"]) == 2
    assert len(commands) == 2
    included = [
        command[index + 1]
        for command in commands
        for index, value in enumerate(command)
        if value == "--include-episode"
    ]
    assert included == [item["source_relative_path"] for item in accepted]
