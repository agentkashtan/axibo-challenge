"""Smoke test for axibo.sim.PiperStackingScene.

Spawns the three objects at given poses, holds the arm at home while opening/closing the gripper for 10 s,
prints robot state + object poses once per second, then saves one frame per camera to outputs/.

    source .venv/bin/activate
    python test_scene.py            # with viewer
    python test_scene.py --no-viewer
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs

from axibo.backend import gs_backend

DURATION = 10.0  # [s]
OUT_DIR = Path("outputs")


def yaw_quat(deg: float) -> tuple[float, float, float, float]:
    """Rotation about world z as a Genesis (w, x, y, z) quaternion."""
    x, y, z, w = Rotation.from_euler("z", deg, degrees=True).as_quat()
    return (w, x, y, z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs_backend(), logging_level="warning")
    from axibo.sim import CAMERA_NAMES, ObjectPose, PiperStackingScene

    object_poses = {  # pos = object center, so z = half height to rest on the floor
        "red cube": ObjectPose(pos=(0.28, 0.12, 0.02), quat=yaw_quat(30)),
        "red cylinder": ObjectPose(pos=(0.36, -0.02, 0.025)),
        "blue cube": ObjectPose(pos=(0.25, -0.12, 0.02), quat=yaw_quat(-45)),
    }
    sim = PiperStackingScene(object_poses, show_viewer=not args.no_viewer)
    sim.reset()

    home = sim.observe().state[0]
    n_steps = int(DURATION * sim.cfg.control_hz)
    for step in range(n_steps):
        t = step / sim.cfg.control_hz
        action = home.copy()
        action[6] = 0.025 * (1 - np.cos(2 * np.pi * t / 4.0))  # gripper: open and close every 4 s
        clipped = bool(sim.apply_action(action[None])[0])
        sim.step_control()

        if step % sim.cfg.control_hz == 0:
            state = sim.observe().state[0]
            print(f"t={t:4.1f}s  state={np.round(state, 3)}  gripper cmd={action[6]:.3f}  clipped={clipped}")
            for name, entity in sim.objects.items():
                print(f"          {name:13s} pos={np.round(sim.object_pose(name)[0][0], 3)}")

    OUT_DIR.mkdir(exist_ok=True)
    obs = sim.observe().env(0)
    for name in CAMERA_NAMES:
        path = OUT_DIR / f"test_scene_{name}.png"
        imageio.imwrite(path, obs.images[name])
        print(f"saved {path} {obs.images[name].shape}")


if __name__ == "__main__":
    main()
