"""Replay a saved rollout in the simulator, at real-time speed, from the actions it recorded.

`eval_async.py --save-traces` stores every commanded action plus the layout id and seed, so an episode can be
re-executed exactly: reset the scene with the same layout and jitter seed, then feed the recorded actions back in.
No policy is loaded, so replay is fast and deterministic, and stalled steps show up naturally because the recorded
action simply repeats.

    source .venv/bin/activate
    python replay_trace.py synctest                     # first saved trial of outputs/eval/synctest, real time
    python replay_trace.py synctest --trial 3           # a specific trial
    python replay_trace.py synctest --live-cameras      # what the policy saw, with STALL / SWITCH labels
    python replay_trace.py synctest --save-video outputs/eval/synctest/k0.mp4 --no-viewer
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import genesis as gs

from axibo.backend import gs_backend


def resolve_trace(arg: str, trial: int) -> Path:
    """Accept a run name (outputs/eval/<name>), a run dir, or a direct npz path."""
    path = Path(arg)
    if path.suffix == ".npz":
        return path
    run = path if path.is_dir() else Path("outputs/eval") / arg
    traces = sorted((run / "traces").glob("trial_*.npz"))
    if not traces:
        raise SystemExit(f"no traces in {run / 'traces'} - was the run made with --save-traces?")
    by_index = {int(t.stem.split("_")[1]): t for t in traces}
    if trial not in by_index:
        raise SystemExit(f"trial {trial} not saved in {run}; available: {sorted(by_index)}")
    return by_index[trial]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="run name under outputs/eval/, a run dir, or a trial_XXXX.npz path")
    p.add_argument("--trial", type=int, default=0, help="which trial, when a run name/dir is given")
    p.add_argument("--layouts", default=None, help="layouts CSV (default: read from the run's summary.json)")
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 0.5 = half speed, 0 = as fast as possible")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--live-cameras", action="store_true", help="show the three policy cameras instead of the viewer")
    p.add_argument("--save-video", type=Path, default=None, help="write an mp4 of the top+side cameras")
    p.add_argument("--fps", type=int, default=30, help="video frame rate (sim runs at 30 Hz)")
    return p.parse_args()


def main():
    args = parse_args()
    args.trace = resolve_trace(args.trace, args.trial)
    d = np.load(args.trace, allow_pickle=True)
    actions = d["action"]
    layout_id, seed = int(d["layout_id"]), int(d["seed"])
    task, outcome = str(d["task"]), str(d["outcome"])
    k = int(d["queue_threshold"]) if "queue_threshold" in d else -1
    stalled, switch = d["stalled"], d["switch"]

    layouts_csv = args.layouts
    if layouts_csv is None:
        summary = args.trace.parent.parent / "summary.json"
        if not summary.exists():
            raise SystemExit(f"no --layouts given and no {summary}")
        layouts_csv = json.loads(summary.read_text())["layouts"]

    print(f"{args.trace.name}: {task!r} | layout {layout_id} seed {seed} | k={k} | outcome {outcome}\n"
          f"  {len(actions)} steps ({len(actions) / 30:.1f} s), stalled {stalled.mean():.1%}, "
          f"{int(switch.sum())} chunk switches | layouts {layouts_csv}")

    gs.init(backend=gs_backend(), seed=0, logging_level="warning")
    from axibo.scene_builder import load_layouts_csv
    from axibo.sim import CAMERA_NAMES, PiperStackingScene

    show_viewer = not (args.no_viewer or args.live_cameras or args.save_video)
    layout = load_layouts_csv(layouts_csv)[layout_id]
    sim = PiperStackingScene(layout, show_viewer=show_viewer)
    sim.reseed(seed)
    sim.reset(object_poses=layout)
    for _ in range(10):
        sim.step_control()

    cv2 = writer = None
    if args.live_cameras:
        import cv2
    if args.save_video:
        import imageio

        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.save_video, fps=args.fps)

    period = (1.0 / 30.0) / args.speed if args.speed > 0 else 0.0
    t_next = time.perf_counter()
    for i, action in enumerate(actions):
        sim.apply_action(action[None])
        sim.step_control()

        if args.live_cameras or writer is not None:
            obs = sim.observe().env(0)
            names = CAMERA_NAMES if args.live_cameras else ("top", "side")
            panels = [np.ascontiguousarray(obs.images[n]) for n in names]
            board = np.hstack([p if p.shape[0] == panels[0].shape[0] else p for p in panels])
            if args.live_cameras:
                label = f"step {i}  {'STALL' if stalled[i] else ''}{'  SWITCH' if switch[i] else ''}"
                cv2.putText(board, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(f"replay k={k} ({outcome}) - q to quit", cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer is not None:
                writer.append_data(board)

        if period:
            t_next += period
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                t_next = time.perf_counter()  # fell behind (rendering); do not try to catch up

    if writer is not None:
        writer.close()
        print(f"wrote {args.save_video}")


if __name__ == "__main__":
    main()
