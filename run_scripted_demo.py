"""Plan and execute one scripted pick-and-place demo, then print a quick geometric check of the result.

    source .venv/bin/activate
    python run_scripted_demo.py --layout-id 0 --source "red cylinder" --destination "blue cube" [--no-viewer]
    python run_scripted_demo.py --layout-id 0 --live-cameras   # watch what the policy sees, not the free camera
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs

from axibo.backend import gs_backend

SETTLE_S = 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", default="data/layouts/pilot_50.csv")
    p.add_argument("--layout-id", type=int, default=0)
    p.add_argument("--source", default="red cylinder")
    p.add_argument("--destination", default="blue cube")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--live-cameras", action="store_true",
                   help="show the three policy cameras in an OpenCV window while the demo runs (implies --no-viewer)")
    p.add_argument("--live-every", type=int, default=2, help="render the live view every N control steps")
    p.add_argument("--live-size", type=int, default=340, help="pixel size of each panel in the live window")
    p.add_argument("--squeeze", type=float, default=None, help="[m] grip interference per side (PlannerConfig)")
    p.add_argument("--object-friction", type=float, default=None, help="tangential friction of the 3 objects")
    args = p.parse_args()

    gs.init(backend=gs_backend(), logging_level="warning")
    from axibo.scene_builder import load_layouts_csv
    from axibo.scripted import PlannerConfig, execute_plan, plan_demo
    from axibo.sim import OBJECT_NAMES, PiperStackingScene, SimConfig, object_height

    sim_cfg = SimConfig()
    if args.object_friction is not None:
        sim_cfg.object_friction = args.object_friction
    planner_cfg = PlannerConfig()
    if args.squeeze is not None:
        planner_cfg.squeeze = args.squeeze

    layout = load_layouts_csv(args.layouts)[args.layout_id]
    show_viewer = not (args.no_viewer or args.live_cameras)  # both windows at once fight over the main thread
    sim = PiperStackingScene(layout, cfg=sim_cfg, show_viewer=show_viewer)
    sim.reset()
    for _ in range(10):  # let objects settle on the table
        sim.step_control()

    pair = (OBJECT_NAMES.index(args.source), OBJECT_NAMES.index(args.destination))
    plan = plan_demo(sim, args.layout_id, pair, args.layouts, cfg=planner_cfg)
    print(plan.summary())

    on_step = None
    if args.live_cameras:
        import cv2

        from axibo.sim import CAMERA_NAMES

        def on_step(step, _action):
            if step % args.live_every:
                return
            obs = sim.observe().env(0)
            panels = []
            for name in CAMERA_NAMES:
                img = cv2.resize(obs.images[name], (args.live_size, args.live_size))
                cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(img)
            cv2.imshow("policy cameras (q to quit)", cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    n_clipped = execute_plan(sim, plan, on_step=on_step)
    for _ in range(int(SETTLE_S * sim.cfg.control_hz)):
        sim.step_control()

    src_pos, dst_pos = sim.object_pose(plan.source)[0][0], sim.object_pose(plan.destination)[0][0]
    w, x, y, z = sim.object_pose(plan.destination)[1][0]
    dst_tilt = np.degrees(np.arccos(np.clip(Rotation.from_quat([x, y, z, w]).as_matrix()[2, 2], -1, 1)))
    expected_dz = object_height(plan.destination, sim.cfg) / 2 + object_height(plan.source, sim.cfg) / 2
    print(f"clipped actions: {n_clipped}")
    print(f"source {np.round(src_pos, 4)}  destination {np.round(dst_pos, 4)}")
    print(f"xy offset {np.linalg.norm(src_pos[:2] - dst_pos[:2]) * 1000:.1f} mm | "
          f"dz {(src_pos[2] - dst_pos[2]) * 1000:.1f} mm (expected {expected_dz * 1000:.1f}) | "
          f"destination tilt {dst_tilt:.1f} deg")


if __name__ == "__main__":
    main()
