from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from lightworkbench import frame_id_repair
from lightworkbench.validation import output_fingerprint


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def test_repairs_report_targets_and_updates_audit(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    relative = Path("2026-09-02/site/task/episode_000001")
    episode = raw_root / relative
    episode.mkdir(parents=True)
    manifest = [
        {"_type": "session_header", "episode_id": 1, "fps_target": 30},
        {
            "frame_idx": 0,
            "videos": {
                "camera": {
                    "path": "videos/camera/video.mp4",
                    "frame_id": 100,
                    "is_repeat": False,
                    "frames_dropped": 0,
                },
                "unused": {"path": None, "frame_id": None},
            },
        },
        {
            "frame_idx": 1,
            "videos": {
                "camera": {
                    "path": "videos/camera/video.mp4",
                    "frame_id": 101,
                    "is_repeat": False,
                    "frames_dropped": 0,
                },
                "unused": {"path": None, "frame_id": None},
            },
        },
        {"_type": "session_footer", "aborted": False},
    ]
    (episode / "manifest.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in manifest),
        encoding="utf-8",
    )
    _write_json(episode / "task_meta.json", {"task_title": "task"})
    cut_info = {
        "mode": "no_trim",
        "removedRanges": [],
        "sourceFrames": 2,
        "outputFrames": 2,
    }
    cut_info["outputFingerprint"] = output_fingerprint(episode)
    _write_json(episode / "CUT_INFO.json", cut_info)
    report = {
        "skipped": [
            {
                "source_relative_path": relative.as_posix(),
                "reasons": ["video_frame_id_mismatch:camera:0"],
            }
        ]
    }
    report_path = tmp_path / "conversion_report.json"
    _write_json(report_path, report)
    backup = tmp_path / "backup.zip"
    script = Path(__file__).parents[1] / "scripts" / "repair_video_frame_ids.py"
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))

    dry_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--report",
            str(report_path),
            "--raw-root",
            str(raw_root),
            "--expected-count",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert '"frame_ids_to_change": 2' in dry_run.stdout
    assert not backup.exists()

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--report",
            str(report_path),
            "--raw-root",
            str(raw_root),
            "--expected-count",
            "1",
            "--apply",
            "--backup",
            str(backup),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    rows = [json.loads(line) for line in (episode / "manifest.jsonl").read_text().splitlines()]
    assert rows[1]["videos"]["camera"]["frame_id"] == 0
    assert rows[2]["videos"]["camera"]["frame_id"] == 1
    assert rows[1]["videos"]["unused"]["frame_id"] is None
    repaired_info = json.loads((episode / "CUT_INFO.json").read_text())
    assert repaired_info["outputFingerprint"] == output_fingerprint(episode)
    assert repaired_info["repairs"][0]["type"] == "video_frame_id_reindex"
    with zipfile.ZipFile(backup) as archive:
        assert f"{relative.as_posix()}/manifest.jsonl" in archive.namelist()
        original_rows = [
            json.loads(line)
            for line in archive.read(f"{relative.as_posix()}/manifest.jsonl").decode().splitlines()
        ]
    assert original_rows[1]["videos"]["camera"]["frame_id"] == 100


def test_stale_fingerprint_prevents_the_entire_batch_before_writes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    relatives = [
        Path("2026-09-02/site/task/episode_000001"),
        Path("2026-09-02/site/task/episode_000002"),
    ]
    for episode_index, relative in enumerate(relatives, 1):
        episode = raw_root / relative
        episode.mkdir(parents=True)
        rows = [{"_type": "session_header", "episode_id": episode_index, "fps_target": 30}]
        rows.extend(
            [
                {
                    "frame_idx": frame_index,
                    "videos": {
                        "camera": {
                            "path": "videos/camera/video.mp4",
                            "frame_id": 100 + frame_index,
                            "is_repeat": False,
                            "frames_dropped": 0,
                        }
                    },
                }
                for frame_index in range(2)
            ]
        )
        rows.append({"_type": "session_footer", "aborted": False})
        (episode / "manifest.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_json(episode / "task_meta.json", {"task_title": "task"})
        cut_info = {
            "mode": "no_trim",
            "removedRanges": [],
            "sourceFrames": 2,
            "outputFrames": 2,
            "outputFingerprint": output_fingerprint(episode),
        }
        if episode_index == 2:
            cut_info["outputFingerprint"] = "stale"
        _write_json(episode / "CUT_INFO.json", cut_info)

    report_path = tmp_path / "conversion_report.json"
    _write_json(
        report_path,
        {
            "skipped": [
                {
                    "source_relative_path": relative.as_posix(),
                    "reasons": ["video_frame_id_mismatch:camera:0"],
                }
                for relative in relatives
            ]
        },
    )
    backup = tmp_path / "backup.zip"
    script = Path(__file__).parents[1] / "scripts" / "repair_video_frame_ids.py"
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--report",
            str(report_path),
            "--raw-root",
            str(raw_root),
            "--expected-count",
            "2",
            "--apply",
            "--backup",
            str(backup),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode != 0
    first = raw_root / relatives[0]
    first_rows = [json.loads(line) for line in (first / "manifest.jsonl").read_text().splitlines()]
    assert first_rows[1]["videos"]["camera"]["frame_id"] == 100
    first_cut_info = json.loads((first / "CUT_INFO.json").read_text())
    assert "repairs" not in first_cut_info
    assert first_cut_info["outputFingerprint"] == output_fingerprint(first)


def test_final_audit_failure_rolls_back_the_repaired_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    relative = Path("2026-09-02/site/task/episode_000001")
    episode = raw_root / relative
    episode.mkdir(parents=True)
    rows = [{"_type": "session_header", "episode_id": 1, "fps_target": 30}]
    rows.extend(
        [
            {
                "frame_idx": frame_index,
                "videos": {
                    "camera": {
                        "path": "videos/camera/video.mp4",
                        "frame_id": 100 + frame_index,
                        "is_repeat": False,
                        "frames_dropped": 0,
                    }
                }
            }
            for frame_index in range(2)
        ]
    )
    rows.append({"_type": "session_footer", "aborted": False})
    (episode / "manifest.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(episode / "task_meta.json", {"task_title": "task"})
    cut_info = {
        "mode": "no_trim",
        "removedRanges": [],
        "sourceFrames": 2,
        "outputFrames": 2,
        "outputFingerprint": output_fingerprint(episode),
    }
    _write_json(episode / "CUT_INFO.json", cut_info)
    original_manifest = (episode / "manifest.jsonl").read_bytes()
    original_cut_info = (episode / "CUT_INFO.json").read_bytes()
    real_audit = frame_id_repair._audit_target
    audit_calls = 0

    def fail_final_audit(root: Path, target: str) -> frame_id_repair.TargetAudit:
        nonlocal audit_calls
        audit_calls += 1
        result = real_audit(root, target)
        if audit_calls == 2:
            raise RuntimeError("synthetic final audit failure")
        return result

    monkeypatch.setattr(frame_id_repair, "_audit_target", fail_final_audit)
    backup = tmp_path / "backup.zip"

    with pytest.raises(RuntimeError, match="synthetic final audit failure"):
        frame_id_repair.repair_frame_ids(
            raw_root,
            [relative.as_posix()],
            backup,
            source_label="test",
        )

    assert backup.is_file()
    assert (episode / "manifest.jsonl").read_bytes() == original_manifest
    assert (episode / "CUT_INFO.json").read_bytes() == original_cut_info
    restored_info = json.loads((episode / "CUT_INFO.json").read_text())
    assert restored_info["outputFingerprint"] == output_fingerprint(episode)
