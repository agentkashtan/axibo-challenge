"""Synchronous SmolVLA closed loop on PiperX in Genesis: observe -> predict a chunk -> execute it -> repeat.

Zero-shot smolvla_base was never trained on PiperX, so the motion is not expected to solve the task; this script
verifies the observation/action plumbing and gives a first latency number.

    source .venv/bin/activate
    python run_smolvla.py --task "put the red cylinder on the blue cube" --n-chunks 5
    python run_smolvla.py --layouts data/layouts/testv2_10.csv --layout-id 3 \
        --source "red cube" --destination "blue cube" --checkpoint outputs/train/.../pretrained_model
"""

import argparse

import numpy as np

import genesis as gs

from axibo.backend import gs_backend, torch_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layouts", default=None, help="layouts CSV; without it the placeholder scene below is used")
    p.add_argument("--layout-id", type=int, default=0)
    p.add_argument("--source", default=None, help="with --layouts: builds the task string from source/destination")
    p.add_argument("--destination", default=None)
    p.add_argument("--task", default="put the red cylinder on the blue cube")
    p.add_argument("--checkpoint", default="lerobot/smolvla_base")
    p.add_argument("--n-chunks", type=int, default=5, help="number of inference calls")
    p.add_argument("--exec-steps", type=int, default=50, help="actions executed per chunk (<= chunk size)")
    p.add_argument("--device", default=torch_device())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-viewer", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gs.init(backend=gs_backend(), seed=args.seed, logging_level="warning")

    # Imported after gs.init so Genesis is initialized before any scene code runs.
    from axibo.policy import SmolVLARunner
    from axibo.scene_builder import load_layouts_csv
    from axibo.sim import ObjectPose, PiperStackingScene

    task = args.task
    if args.layouts:
        object_poses = load_layouts_csv(args.layouts)[args.layout_id]
        if args.source and args.destination:
            task = f"put the {args.source} on the {args.destination}"
    else:
        # Fixed placeholder scene; objects spawn 5 mm above the floor and settle. Note the cylinder sits at x = 0.35,
        # outside the sampled workspace (x <= 0.34), so it is out of distribution for a policy trained on layouts.
        object_poses = {
            "red cube": ObjectPose(pos=(0.30, 0.10, 0.025)),
            "red cylinder": ObjectPose(pos=(0.35, 0.00, 0.030)),
            "blue cube": ObjectPose(pos=(0.30, -0.10, 0.025)),
        }
    sim = PiperStackingScene(object_poses, show_viewer=not args.no_viewer)
    runner = SmolVLARunner(sim.action_low, sim.action_high, checkpoint=args.checkpoint, device=args.device)
    exec_steps = min(args.exec_steps, runner.chunk_size)

    sim.reset()
    runner.reset()
    scene = f"{args.layouts}#{args.layout_id}" if args.layouts else "placeholder scene"
    print(f"task: {task!r} | {scene} | control {sim.cfg.control_hz} Hz | "
          f"executing {exec_steps}/{runner.chunk_size} per chunk")

    for i in range(args.n_chunks):
        obs = sim.observe().env(0)
        chunk = runner.predict_chunk(obs, task)

        n_clipped = 0
        for action in chunk[:exec_steps]:
            n_clipped += int(sim.apply_action(action[None])[0])
            sim.step_control()

        print(
            f"chunk {i}: {runner.latencies_s[-1] * 1000:7.1f} ms | shape {chunk.shape} | "
            f"range [{chunk.min():+.3f}, {chunk.max():+.3f}] | clipped {n_clipped}/{exec_steps} | "
            f"state at chunk start {np.round(obs.state, 3)}"
        )

    lat = np.array(runner.latencies_s[1:] or runner.latencies_s) * 1000  # first call includes warmup
    print(f"latency (excluding first call): p50 {np.percentile(lat, 50):.1f} ms, p99 {np.percentile(lat, 99):.1f} ms")


if __name__ == "__main__":
    main()
