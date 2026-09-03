from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightworkbench.operations import (
    FINGERPRINT_EXCLUDES,
    FINGERPRINT_VERSION,
    TIMESTAMP_REWRITE_VERSION,
    output_fingerprint,
    rewrite_manifest,
)


WALL_TIMES = [100.0, 100.101, 100.202, 100.306, 100.407, 100.509, 100.608, 100.710, 100.813]


def _source_row(index: int) -> dict:
    wall = WALL_TIMES[index]
    intended = 10.0 + wall - WALL_TIMES[0]
    jitter = (index + 1) * 0.0001
    return {
        "frame_idx": index,
        "t_wall": wall,
        "t_monotonic": intended + jitter,
        "t_intended": intended,
        "t_ns": int(round(wall * 1_000_000_000)),
        "t_jitter_ms": -999.0,
        "session_start_t": 99.5,
        "robot_state": {"timestamp": wall - 0.011},
        "imu": {"stamp": wall - 0.007},
        "control": {"ts": wall * 1000.0 - 2.5},
        "topics_t": {
            "sensor_t": wall - 0.013,
            "sensor_age_ms": -1.0,
            "unavailable_t": 0.0,
            "unavailable_age_ms": -1.0,
        },
        "videos": {
            "head": {
                "path": "videos/head/episode.mp4",
                "frame_id": 1000 + index,
                "is_repeat": True,
                "frames_dropped": 3,
            }
        },
    }


def _rewrite(tmp_path: Path, keep: set[int], *, virtual_video: bool = False) -> tuple[dict, list[dict], dict]:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    rows = [
        {"_type": "session_header", "t_wall": 99.5, "fps_target": 10.0},
        *[_source_row(index) for index in range(len(WALL_TIMES))],
        {"_type": "session_footer", "t_wall": WALL_TIMES[-1] + 0.1},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit = {
        "timestampRewriteVersion": TIMESTAMP_REWRITE_VERSION,
        "fingerprintVersion": FINGERPRINT_VERSION,
        "fingerprintExcludes": list(FINGERPRINT_EXCLUDES),
    }

    assert rewrite_manifest(
        source, destination, keep, 10.0, audit, virtual_video=virtual_video,
    ) == len(keep)
    output = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    return output[0], output[1:-1], output[-1]


def _assert_shifted(row: dict, source_index: int, shift: float, output_index: int) -> None:
    source = _source_row(source_index)
    assert row["frame_idx"] == output_index
    assert row["t_wall"] == pytest.approx(source["t_wall"] - shift)
    assert row["t_monotonic"] == pytest.approx(source["t_monotonic"] - shift)
    assert row["t_intended"] == pytest.approx(source["t_intended"] - shift)
    assert row["t_ns"] == source["t_ns"] - round(shift * 1_000_000_000)
    assert row["robot_state"]["timestamp"] == pytest.approx(source["robot_state"]["timestamp"] - shift)
    assert row["imu"]["stamp"] == pytest.approx(source["imu"]["stamp"] - shift)
    assert row["control"]["ts"] == pytest.approx(source["control"]["ts"] - shift * 1000.0)
    assert row["topics_t"]["sensor_t"] == pytest.approx(source["topics_t"]["sensor_t"] - shift)
    assert row["topics_t"]["sensor_age_ms"] == pytest.approx(13.0)
    assert row["topics_t"]["unavailable_t"] == 0.0
    assert row["topics_t"]["unavailable_age_ms"] == pytest.approx(row["t_wall"] * 1000.0)
    assert row["t_jitter_ms"] == pytest.approx((source_index + 1) * 0.1)
    assert row["videos"]["head"] == {
        "path": "videos/head/episode.mp4",
        "frame_id": output_index,
        "is_repeat": False,
        "frames_dropped": 0,
    }


def test_prefix_removal_moves_all_time_families_and_start_times(tmp_path: Path) -> None:
    kept = set(range(2, len(WALL_TIMES)))
    header, rows, footer = _rewrite(tmp_path, kept)
    shift = WALL_TIMES[2] - WALL_TIMES[0]

    assert header["t_wall"] == pytest.approx(WALL_TIMES[0])
    assert header["session_start_t"] == pytest.approx(WALL_TIMES[0])
    assert header["lightworkbench"]["timestampRewriteVersion"] == 2
    assert footer["t_wall"] == pytest.approx(WALL_TIMES[-1] + 0.1 - shift)
    for output_index, source_index in enumerate(sorted(kept)):
        _assert_shifted(rows[output_index], source_index, shift, output_index)
        assert rows[output_index]["session_start_t"] == pytest.approx(WALL_TIMES[0])
    assert rows[1]["t_wall"] - rows[0]["t_wall"] == pytest.approx(WALL_TIMES[3] - WALL_TIMES[2])


def test_suffix_removal_keeps_original_jitter_and_timestamps(tmp_path: Path) -> None:
    kept = set(range(6))
    header, rows, footer = _rewrite(tmp_path, kept)

    assert header["t_wall"] == pytest.approx(WALL_TIMES[0])
    for output_index in range(6):
        _assert_shifted(rows[output_index], output_index, 0.0, output_index)
    assert rows[-1]["t_wall"] - rows[-2]["t_wall"] == pytest.approx(WALL_TIMES[5] - WALL_TIMES[4])
    assert footer["t_wall"] == pytest.approx(WALL_TIMES[-1] + 0.1)


def test_multiple_internal_removals_use_cumulative_segment_shifts(tmp_path: Path) -> None:
    kept_indices = [0, 1, 4, 5, 7, 8]
    _, rows, _ = _rewrite(tmp_path, set(kept_indices))
    first_internal_shift = WALL_TIMES[4] - WALL_TIMES[2]
    second_internal_shift = first_internal_shift + WALL_TIMES[7] - WALL_TIMES[6]
    shifts = [0.0, 0.0, first_internal_shift, first_internal_shift, second_internal_shift, second_internal_shift]

    for output_index, (source_index, shift) in enumerate(zip(kept_indices, shifts, strict=True)):
        _assert_shifted(rows[output_index], source_index, shift, output_index)
    assert rows[2]["t_wall"] - rows[1]["t_wall"] == pytest.approx(WALL_TIMES[2] - WALL_TIMES[1])
    assert rows[4]["t_wall"] - rows[3]["t_wall"] == pytest.approx(WALL_TIMES[6] - WALL_TIMES[5])
    assert rows[3]["t_wall"] - rows[2]["t_wall"] == pytest.approx(WALL_TIMES[5] - WALL_TIMES[4])
    assert rows[5]["t_wall"] - rows[4]["t_wall"] == pytest.approx(WALL_TIMES[8] - WALL_TIMES[7])


def test_virtual_manifest_records_physical_source_frames(tmp_path: Path) -> None:
    kept_indices = [0, 1, 4, 5, 8]
    _, rows, _ = _rewrite(tmp_path, set(kept_indices), virtual_video=True)

    assert [row["frame_idx"] for row in rows] == list(range(len(kept_indices)))
    assert [row["videos"]["head"]["frame_id"] for row in rows] == list(range(len(kept_indices)))
    assert [row["videos"]["head"]["source_frame_id"] for row in rows] == kept_indices


def test_fingerprint_excludes_only_root_cut_info(tmp_path: Path) -> None:
    (tmp_path / "manifest.jsonl").write_text("manifest\n", encoding="utf-8")
    before = output_fingerprint(tmp_path)
    (tmp_path / "CUT_INFO.json").write_text('{"revision":1}\n', encoding="utf-8")
    assert output_fingerprint(tmp_path) == before
    (tmp_path / "CUT_INFO.json").write_text('{"revision":200}\n', encoding="utf-8")
    assert output_fingerprint(tmp_path) == before

    nested = tmp_path / "metadata"
    nested.mkdir()
    (nested / "CUT_INFO.json").write_text("nested\n", encoding="utf-8")
    assert output_fingerprint(tmp_path) != before
