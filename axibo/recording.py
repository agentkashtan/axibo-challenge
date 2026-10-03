"""LeRobot recording for batched collection.

LeRobot's writer keeps a single open episode buffer, but a batch runs N episodes at once. Since every executed episode
is saved, episode indices are known before execution, so each env writes its frames straight to LeRobot's temporary
image paths and later hands a complete buffer to `dataset.save_episode(episode_data=...)`, in index order.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.feature_utils import validate_frame
from lerobot.datasets.image_writer import write_image
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH

from axibo.sim import CAMERA_NAMES, OBJECT_NAMES, Observation, SimConfig

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
IMAGE_KEYS = {name: f"observation.images.{name}" for name in CAMERA_NAMES}


def make_features(cfg: SimConfig) -> dict:
    width, height = cfg.cam_res
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        # Not policy inputs (no "observation." prefix): raw finger joints and object poses, for analysis/eval.
        "sim.qpos": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES[:6] + ["joint7", "joint8"]},
        "sim.object_poses": {
            "dtype": "float32",
            "shape": (7 * len(OBJECT_NAMES),),
            "names": [f"{o.replace(' ', '_')}_{c}" for o in OBJECT_NAMES for c in ("x", "y", "z", "qw", "qx", "qy", "qz")],
        },
    }
    for key in IMAGE_KEYS.values():
        features[key] = {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    return features


class EpisodeWriter:
    """Accumulates one episode for a known episode index; images go directly to LeRobot's temp paths."""

    def __init__(self, dataset: LeRobotDataset, episode_index: int, task: str, pool: ThreadPoolExecutor):
        self.dataset, self.episode_index, self.task, self.pool = dataset, episode_index, task, pool
        self.numeric = {key: [] for key in ("observation.state", "action", "sim.qpos", "sim.object_poses")}
        self.image_paths = {key: [] for key in IMAGE_KEYS.values()}
        self.futures = []
        self.size = 0

    def add(self, obs: Observation, action: np.ndarray, qpos: np.ndarray, object_poses: np.ndarray):
        """obs is a single-env Observation; action is the command issued at this frame."""
        values = {
            "observation.state": obs.state.astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "sim.qpos": qpos.astype(np.float32),
            "sim.object_poses": object_poses.astype(np.float32),
        }
        if self.size == 0:
            frame = {**values, **{IMAGE_KEYS[n]: obs.images[n] for n in CAMERA_NAMES}, "task": self.task}
            validate_frame(frame, self.dataset.meta.features)
        for key, value in values.items():
            self.numeric[key].append(value)
        for name in CAMERA_NAMES:
            key = IMAGE_KEYS[name]
            path = self.dataset.root / DEFAULT_IMAGE_PATH.format(
                image_key=key, episode_index=self.episode_index, frame_index=self.size
            )
            if self.size == 0:
                path.parent.mkdir(parents=True, exist_ok=True)
            self.futures.append(self.pool.submit(write_image, obs.images[name], path, compress_level=1))
            self.image_paths[key].append(str(path))
        self.size += 1

    def episode_buffer(self) -> dict:
        for future in self.futures:  # wait for PNGs and surface write errors
            future.result()
        buffer = {
            "size": self.size,
            "task": [self.task] * self.size,
            "episode_index": self.episode_index,
            "frame_index": list(range(self.size)),
            "timestamp": [i / self.dataset.meta.fps for i in range(self.size)],
            "index": [],  # filled in by save_episode
            "task_index": [],  # filled in by save_episode
        }
        buffer.update({key: list(vals) for key, vals in self.numeric.items()})
        buffer.update(self.image_paths)
        return buffer
