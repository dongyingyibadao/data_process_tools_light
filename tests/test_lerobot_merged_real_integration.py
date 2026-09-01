from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from lightworkbench import lerobot_converter as converter


BUNDLE_ENV = "LEROBOT_MERGED_BUNDLE"
TRAINING_KEYS = {f"observation.images.{name}" for name in converter.TRAINING_VIDEO_STREAMS}
EXPECTED_AUXILIARY_STREAMS = {"rgbd_head_depth", "head_left", "head_right"}


def _bundle_root() -> Path:
    value = os.environ.get(BUNDLE_ENV)
    if not value:
        pytest.skip(f"set {BUNDLE_ENV} to run merged LeRobot integration checks")
    root = Path(value).resolve()
    if not root.is_dir():
        pytest.fail(f"{BUNDLE_ENV} is not a directory: {root}")
    return root


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


def _assert_merged_pair(bundle: Path) -> None:
    owner_root = bundle / converter.WHOLE_BODY_JOINT
    hybrid_root = bundle / converter.BODY_JOINT_EEF
    owner_state, owner = _load_dataset(owner_root)
    hybrid_state, hybrid = _load_dataset(hybrid_root)
    try:
        assert owner_state["dataset_layout"] == hybrid_state["dataset_layout"] == "merged"
        assert owner_state["source_root"] == hybrid_state["source_root"]
        assert owner_state["episodes"] == hybrid_state["episodes"]
        entries = owner_state["episodes"]
        relatives = [entry["source_relative_path"] for entry in entries]
        assert len(relatives) == len(set(relatives))
        assert all(not Path(value).is_absolute() and ".." not in Path(value).parts for value in relatives)
        expected_tasks = {entry["stored_task"] for entry in entries}
        assert set(owner.meta.tasks.index) == set(hybrid.meta.tasks.index) == expected_tasks

        assert tuple(owner.meta.features["action"]["shape"]) == (23,)
        assert tuple(hybrid.meta.features["action"]["shape"]) == (21,)
        assert tuple(owner.meta.features["observation.state"]["shape"]) == (38,)
        assert tuple(hybrid.meta.features["observation.state"]["shape"]) == (38,)
        assert set(owner.meta.video_keys) == set(hybrid.meta.video_keys) == TRAINING_KEYS
        assert len(owner) == len(hybrid) == sum(int(entry["output_frames"]) for entry in entries)

        owner_raw = owner.hf_dataset.with_format("numpy")
        hybrid_raw = hybrid.hf_dataset.with_format("numpy")
        owner_actions = np.asarray(owner_raw["action"])
        hybrid_actions = np.asarray(hybrid_raw["action"])
        states = np.asarray(owner_raw["observation.state"])
        assert owner_actions.shape == (len(owner), 23)
        assert hybrid_actions.shape == (len(hybrid), 21)
        assert states.shape == (len(owner), 38)
        assert np.isfinite(owner_actions).all() and np.isfinite(hybrid_actions).all()

        for episode_index, entry in enumerate(entries):
            metadata = owner.meta.episodes[episode_index]
            start = int(metadata["dataset_from_index"])
            end = int(metadata["dataset_to_index"])
            assert end - start == int(entry["output_frames"])
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
            np.testing.assert_array_equal(hybrid_action[-1, 4:10], np.zeros(6, dtype=np.float32))
            np.testing.assert_array_equal(hybrid_action[-1, 11:17], np.zeros(6, dtype=np.float32))
            np.testing.assert_allclose(hybrid_action[-1, :4], qpos[-1, :4], rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                hybrid_action[-1, [10, 17]], qpos[-1, [18, 19]], rtol=0, atol=1e-6,
            )
            np.testing.assert_allclose(hybrid_action[-1, 18:21], qpos[-1, 20:23], rtol=0, atol=1e-6)
            assert owner[start]["task"] == hybrid[start]["task"] == entry["stored_task"]
            assert owner[end - 1]["task"] == hybrid[end - 1]["task"] == entry["stored_task"]
            for key in TRAINING_KEYS:
                owner_video = owner_root / owner.meta.get_video_file_path(episode_index, key)
                hybrid_video = hybrid_root / hybrid.meta.get_video_file_path(episode_index, key)
                assert os.path.samefile(owner_video, hybrid_video)

        for name in ("videos", "auxiliary"):
            link = hybrid_root / name
            assert link.is_symlink()
            assert not os.path.isabs(os.readlink(link))
            assert link.resolve(strict=True) == (owner_root / name).resolve(strict=True)

        import av
        import pyarrow.parquet as pq

        auxiliary_rows = pq.read_table(owner_root / converter.AUXILIARY_INDEX_PATH).to_pylist()
        assert len(auxiliary_rows) == len(entries) * len(EXPECTED_AUXILIARY_STREAMS)
        assert {(int(row["episode_index"]), row["stream"]) for row in auxiliary_rows} == {
            (episode_index, stream)
            for episode_index in range(len(entries))
            for stream in EXPECTED_AUXILIARY_STREAMS
        }
        for row in auxiliary_rows:
            video_path = owner_root / row["relative_path"]
            assert video_path.is_file()
            container = av.open(str(video_path))
            try:
                assert sum(1 for _ in container.decode(video=0)) == int(row["frame_count"])
            finally:
                container.close()
    finally:
        owner.finalize()
        hybrid.finalize()


def test_real_merged_bundle_contracts_and_shared_videos() -> None:
    _assert_merged_pair(_bundle_root())


def test_real_merged_bundle_survives_relocation(tmp_path: Path) -> None:
    relocated = tmp_path / "relocated"
    shutil.copytree(_bundle_root(), relocated, symlinks=True)
    _assert_merged_pair(relocated)


def test_real_merged_auxiliary_promotion_after_relocation(tmp_path: Path) -> None:
    bundle = _bundle_root()
    import pyarrow.parquet as pq

    relocated = tmp_path / "promoted"
    shutil.copytree(bundle, relocated, symlinks=True)
    owner_root = relocated / converter.WHOLE_BODY_JOINT
    hybrid_root = relocated / converter.BODY_JOINT_EEF
    result = converter.promote_aux_videos(owner_root, sorted(EXPECTED_AUXILIARY_STREAMS))
    assert result["consumer_roots"] == [str(hybrid_root.resolve())]

    rows = pq.read_table(owner_root / converter.AUXILIARY_INDEX_PATH).to_pylist()
    _, owner = _load_dataset(owner_root)
    _, hybrid = _load_dataset(hybrid_root)
    try:
        expected_keys = TRAINING_KEYS | {
            f"observation.images.{stream}" for stream in EXPECTED_AUXILIARY_STREAMS
        }
        assert set(owner.meta.video_keys) == set(hybrid.meta.video_keys) == expected_keys
        for row in rows:
            official = owner_root / owner.meta.get_video_file_path(
                int(row["episode_index"]), row["feature_key"],
            )
            assert official.is_symlink()
            assert os.path.samefile(official, owner_root / row["relative_path"])
        for episode_index in range(owner.meta.total_episodes):
            position = int(owner.meta.episodes[episode_index]["dataset_from_index"])
            owner_item = owner[position]
            hybrid_item = hybrid[position]
            for key in expected_keys:
                assert tuple(owner_item[key].shape) == tuple(hybrid_item[key].shape)
                assert owner_item[key].shape[0] == (1 if key.endswith("rgbd_head_depth") else 3)
    finally:
        owner.finalize()
        hybrid.finalize()
