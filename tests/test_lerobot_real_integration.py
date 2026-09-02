from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from lightworkbench import lerobot_converter as converter


BUNDLE_ENV = "LEROBOT_DUAL_BUNDLE"
TRAINING_KEYS = {f"observation.images.{name}" for name in converter.TRAINING_VIDEO_STREAMS}
AUXILIARY_STREAMS = {"head_left", "head_right", "rgbd_head_depth"}
STAT_VECTORS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def _bundle_root() -> Path:
    value = os.environ.get(BUNDLE_ENV)
    if not value:
        pytest.skip(f"set {BUNDLE_ENV} to run output-driven LeRobot integration checks")
    root = Path(value).resolve()
    if not root.is_dir():
        pytest.fail(f"{BUNDLE_ENV} is not a directory: {root}")
    return root


def _task_roots(bundle: Path, mode: str) -> list[Path]:
    return sorted(path.parent for path in (bundle / mode).rglob(converter.CONVERSION_STATE_FILENAME))


def _load_dataset(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    state = json.loads((root / converter.CONVERSION_STATE_FILENAME).read_text(encoding="utf-8"))
    dataset = LeRobotDataset(
        repo_id=state["repo_id"],
        root=root,
        video_backend="pyav",
        return_uint8=True,
        download_videos=False,
        force_cache_sync=False,
    )
    return state, dataset


def _assert_footer_schema(root: Path, action_dim: int) -> None:
    import pyarrow.parquet as pq

    for path in sorted((root / "data").glob("*/*.parquet")):
        schema = pq.read_schema(path)
        assert schema.field("action").type.list_size == action_dim
        assert schema.field("observation.state").type.list_size == 38
        metadata = json.loads(schema.metadata[b"huggingface"])
        features = metadata["info"]["features"]
        assert features["action"]["length"] == action_dim
        assert features["observation.state"]["length"] == 38
    for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            for name in STAT_VECTORS:
                assert len(row[f"stats/action/{name}"]) == action_dim
            assert len(row["stats/action/count"]) == 1
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    for name in STAT_VECTORS:
        assert len(stats["action"][name]) == action_dim


def _assert_auxiliary_index(owner_root: Path, expected_episodes: int) -> None:
    import av
    import pyarrow.parquet as pq

    rows = pq.read_table(owner_root / converter.AUXILIARY_INDEX_PATH).to_pylist()
    assert len(rows) == expected_episodes * len(AUXILIARY_STREAMS)
    assert {(int(row["episode_index"]), row["stream"]) for row in rows} == {
        (episode, stream)
        for episode in range(expected_episodes)
        for stream in AUXILIARY_STREAMS
    }
    for row in rows:
        relative = Path(row["relative_path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = owner_root / relative
        assert path.is_file()
        feature_info = json.loads(row["feature_info_json"])
        assert bool(feature_info.get("is_depth_map")) is bool(row["is_depth"])
        if row["is_depth"]:
            assert feature_info["depth_unit"] == "mm"
            for key in ("video.depth_min", "video.depth_max", "video.shift", "video.use_log"):
                assert key in feature_info
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            decoded = sum(1 for _ in container.decode(video=0))
            assert stream.codec_context.name == row["codec"]
            assert stream.codec_context.pix_fmt == row["pixel_format"]
            assert stream.width == row["width"] and stream.height == row["height"]
            assert float(stream.average_rate) == pytest.approx(row["fps"])
            assert decoded == row["frame_count"]
        finally:
            container.close()


def _assert_pair(owner_root: Path, hybrid_root: Path) -> int:
    owner_state, owner = _load_dataset(owner_root)
    hybrid_state, hybrid = _load_dataset(hybrid_root)
    try:
        assert owner_state["stored_task"] == hybrid_state["stored_task"]
        assert owner_state["episodes"] == hybrid_state["episodes"]
        assert owner_state["stored_task"].isascii()
        assert "_" not in owner_state["stored_task"]
        assert tuple(owner.meta.features["action"]["shape"]) == (23,)
        assert tuple(hybrid.meta.features["action"]["shape"]) == (21,)
        assert tuple(owner.meta.features["observation.state"]["shape"]) == (38,)
        assert tuple(hybrid.meta.features["observation.state"]["shape"]) == (38,)
        assert owner.meta.features["action"]["names"]["axes"] == converter.WHOLE_BODY_JOINT_ACTION_NAMES
        assert hybrid.meta.features["action"]["names"]["axes"] == converter.BODY_JOINT_EEF_ACTION_NAMES
        assert set(owner.meta.video_keys) == TRAINING_KEYS
        assert set(hybrid.meta.video_keys) == TRAINING_KEYS
        assert list(owner.meta.tasks.index) == [owner_state["stored_task"]]
        assert list(hybrid.meta.tasks.index) == [owner_state["stored_task"]]

        owner_raw = owner.hf_dataset.with_format("numpy")
        hybrid_raw = hybrid.hf_dataset.with_format("numpy")
        owner_actions = np.asarray(owner_raw["action"])
        hybrid_actions = np.asarray(hybrid_raw["action"])
        states = np.asarray(owner_raw["observation.state"])
        expected_frames = sum(int(entry["output_frames"]) for entry in owner_state["episodes"])
        assert len(owner) == len(hybrid) == expected_frames
        assert owner_actions.shape == (expected_frames, 23)
        assert hybrid_actions.shape == (expected_frames, 21)
        assert states.shape == (expected_frames, 38)
        assert owner_actions.dtype == hybrid_actions.dtype == states.dtype == np.float32
        assert np.isfinite(owner_actions).all() and np.isfinite(hybrid_actions).all()

        shared_columns = set(owner.hf_dataset.column_names) - {"action"}
        assert shared_columns == set(hybrid.hf_dataset.column_names) - {"action"}
        for key in shared_columns:
            np.testing.assert_array_equal(np.asarray(owner_raw[key]), np.asarray(hybrid_raw[key]))

        for episode_index, entry in enumerate(owner_state["episodes"]):
            metadata = owner.meta.episodes[episode_index]
            start = int(metadata["dataset_from_index"])
            end = int(metadata["dataset_to_index"])
            assert end - start == int(entry["source_frames"]) == int(entry["output_frames"])
            qpos = states[start:end, :23]
            whole = owner_actions[start:end]
            hybrid_action = hybrid_actions[start:end]
            np.testing.assert_allclose(whole[:-1], qpos[1:], rtol=0, atol=1e-6)
            np.testing.assert_allclose(whole[-1], qpos[-1], rtol=0, atol=1e-6)
            np.testing.assert_allclose(hybrid_action[:-1, :4], qpos[1:, :4], rtol=0, atol=1e-6)
            np.testing.assert_allclose(hybrid_action[:-1, 10], qpos[1:, 18], rtol=0, atol=1e-6)
            np.testing.assert_allclose(hybrid_action[:-1, 17], qpos[1:, 19], rtol=0, atol=1e-6)
            np.testing.assert_allclose(hybrid_action[:-1, 18:21], qpos[1:, 20:23], rtol=0, atol=1e-6)
            for position in range(end - start - 1):
                expected_left = converter.relative_pose_delta(
                    states[start + position, 23:30].tolist(),
                    states[start + position + 1, 23:30].tolist(),
                )
                expected_right = converter.relative_pose_delta(
                    states[start + position, 30:37].tolist(),
                    states[start + position + 1, 30:37].tolist(),
                )
                np.testing.assert_allclose(hybrid_action[position, 4:10], expected_left, rtol=0, atol=1e-6)
                np.testing.assert_allclose(hybrid_action[position, 11:17], expected_right, rtol=0, atol=1e-6)
            terminal = hybrid_action[-1]
            np.testing.assert_array_equal(terminal[4:10], np.zeros(6, dtype=np.float32))
            np.testing.assert_array_equal(terminal[11:17], np.zeros(6, dtype=np.float32))
            np.testing.assert_allclose(terminal[:4], qpos[-1, :4], rtol=0, atol=1e-6)
            np.testing.assert_allclose(terminal[[10, 17]], qpos[-1, [18, 19]], rtol=0, atol=1e-6)
            np.testing.assert_allclose(terminal[18:21], qpos[-1, 20:23], rtol=0, atol=1e-6)

            for position in (start, end - 1):
                owner_item = owner[position]
                hybrid_item = hybrid[position]
                assert owner_item["task"] == hybrid_item["task"] == owner_state["stored_task"]
                assert tuple(owner_item["action"].shape) == (23,)
                assert tuple(hybrid_item["action"].shape) == (21,)
                for key in TRAINING_KEYS:
                    assert tuple(owner_item[key].shape) == tuple(hybrid_item[key].shape) == (3, 480, 640)

            for key in TRAINING_KEYS:
                owner_video = owner_root / owner.meta.get_video_file_path(episode_index, key)
                hybrid_video = hybrid_root / hybrid.meta.get_video_file_path(episode_index, key)
                assert os.path.samefile(owner_video, hybrid_video)

        for name in ("videos", "auxiliary"):
            link = hybrid_root / name
            assert link.is_symlink() and not os.path.isabs(os.readlink(link))
            assert link.resolve(strict=True) == (owner_root / name).resolve(strict=True)
        _assert_footer_schema(owner_root, 23)
        _assert_footer_schema(hybrid_root, 21)
        _assert_auxiliary_index(owner_root, len(owner_state["episodes"]))
        return expected_frames
    finally:
        owner.finalize()
        hybrid.finalize()


def test_real_dual_bundle_contracts_and_decode() -> None:
    bundle = _bundle_root()
    owner_roots = _task_roots(bundle, converter.WHOLE_BODY_JOINT)
    hybrid_roots = _task_roots(bundle, converter.BODY_JOINT_EEF)
    assert len(owner_roots) == len(hybrid_roots) == 3
    total = 0
    for owner_root, hybrid_root in zip(owner_roots, hybrid_roots):
        assert owner_root.relative_to(bundle / converter.WHOLE_BODY_JOINT) == hybrid_root.relative_to(
            bundle / converter.BODY_JOINT_EEF
        )
        total += _assert_pair(owner_root, hybrid_root)
    assert total == 582


def test_relocated_bundle_and_promoted_auxiliary_readback(tmp_path: Path) -> None:
    bundle = _bundle_root()
    relocated = tmp_path / "relocated_bundle"
    shutil.copytree(bundle, relocated, symlinks=True)

    for owner_root in _task_roots(relocated, converter.WHOLE_BODY_JOINT):
        relative = owner_root.relative_to(relocated / converter.WHOLE_BODY_JOINT)
        hybrid_root = relocated / converter.BODY_JOINT_EEF / relative
        for name in ("videos", "auxiliary"):
            assert (hybrid_root / name).resolve(strict=True) == (owner_root / name).resolve(strict=True)
        result = converter.promote_aux_videos(owner_root, sorted(AUXILIARY_STREAMS))
        assert result["consumer_roots"] == [str(hybrid_root.resolve())]
        rows = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
            owner_root / converter.AUXILIARY_INDEX_PATH
        ).to_pylist()
        state, owner = _load_dataset(owner_root)
        _, hybrid = _load_dataset(hybrid_root)
        try:
            expected_keys = TRAINING_KEYS | {
                f"observation.images.{stream}" for stream in AUXILIARY_STREAMS
            }
            assert set(owner.meta.video_keys) == set(hybrid.meta.video_keys) == expected_keys
            for row in rows:
                official = owner_root / owner.meta.get_video_file_path(
                    int(row["episode_index"]), row["feature_key"],
                )
                assert official.is_symlink()
                assert os.path.samefile(official, owner_root / row["relative_path"])
            for episode_index, entry in enumerate(state["episodes"]):
                metadata = owner.meta.episodes[episode_index]
                for position in (
                    int(metadata["dataset_from_index"]), int(metadata["dataset_to_index"]) - 1,
                ):
                    owner_item = owner[position]
                    hybrid_item = hybrid[position]
                    for key in expected_keys:
                        assert tuple(owner_item[key].shape) == tuple(hybrid_item[key].shape)
                        assert owner_item[key].shape[0] == (1 if key.endswith("rgbd_head_depth") else 3)
                assert int(entry["source_frames"]) == int(entry["output_frames"])
        finally:
            owner.finalize()
            hybrid.finalize()
