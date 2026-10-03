"""Evaluate a fine-tuned SmolVLA checkpoint in Genesis: rollouts, outcome taxonomy, per-pair success.

One trial = one layout x one (source, destination) pair x one seed. The policy runs closed-loop until the
completion detector fires, after which it keeps acting for --settle-s seconds before the final state is judged
(the demos end with a retreat, so a policy that learned them should retreat on its own and leave the stack
standing). Trials that never complete stop at --max-chunks and count as failures.

    source .venv/bin/activate
    python eval_policy.py --checkpoint outputs/train/main_250v3/checkpoints/045000/pretrained_model \\
        --layouts data/layouts/eval_50.csv --pairs train --seeds 0 --name v3_45k

    python eval_policy.py --checkpoint ... --layouts ... --name v3_45k_heldout --pairs held-out
    python eval_policy.py --checkpoint ... --layouts ... --live-cameras     # watch what the policy sees
    python eval_policy.py --checkpoint ... --layouts ... --pairs reversal --layout-id 0  # both directions
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import genesis as gs

from axibo.backend import gs_backend, torch_device

LOG_FIELDS = [
    "trial", "seed", "layout_id", "source", "destination", "task", "outcome", "chunks", "steps",
    "lifted_step", "transported_step", "stacked_step", "release_step", "placed_step",
    "centered_at_release", "release_dz_mm", "release_xy_offset_mm", "release_speed_mm_s" "release_dst_moved_mm", "max_src_z_mm", "lift_threshold_mm", "stack_height_mm",
    "final_xy_offset_mm", "final_dz_error_mm", "final_src_speed_mm_s", "final_settle_disp_mm",
    "final_gripper_mm", "dst_tilt_deg",
    "dst_moved_mm", "third_moved_mm", "third_disturbed", "dst_toppled", "latency_p50_ms", "clipped_steps",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="pretrained_model dir from train_smolvla.py")
    p.add_argument("--layouts", required=True, help="eval layouts CSV (must not be the training layouts)")
    p.add_argument("--num-layouts", type=int, default=None, help="use only the first K layouts")
    p.add_argument("--layout-id", type=int, default=0, help="reversal mode: which layout to use")
    p.add_argument("--pairs", choices=("train", "held-out", "all", "reversal"), default="train")
    p.add_argument("--seeds", default="0", help="comma-separated, one rollout per seed per layout/pair")
    p.add_argument("--max-chunks", type=int, default=10, help="inference calls before a trial is given up on")
    p.add_argument("--exec-steps", type=int, default=50, help="actions executed per chunk (<= chunk size)")
    p.add_argument("--settle-s", type=float, default=4.0, help="policy keeps running this long after detection")
    p.add_argument("--name", required=True, help="run name; results go to outputs/eval/<name>")
    p.add_argument("--device", default=torch_device())
    p.add_argument("--viewer", action="store_true", help="Genesis free camera (not what the policy sees)")
    p.add_argument("--live-cameras", action="store_true",
                   help="show the three policy cameras + live milestone state in an OpenCV window")
    p.add_argument("--live-every", type=int, default=3, help="render the live view every N control steps")
    p.add_argument("--live-size", type=int, default=340, help="pixel size of each panel in the live window")
    return p.parse_args()


def main():
    args = parse_args()
    gs.init(backend=gs_backend(), seed=0, logging_level="warning")

    from axibo.outcome import Trace, analyse
    from axibo.policy import SmolVLARunner
    from axibo.scene_builder import load_layouts_csv
    from axibo.scripted import HELD_OUT_PAIR, TRAIN_PAIRS
    from axibo.sim import OBJECT_NAMES, PiperStackingScene

    layouts = load_layouts_csv(args.layouts)[: args.num_layouts]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.pairs == "reversal":
        src, dst = TRAIN_PAIRS[2]  # red cylinder <-> blue cube: both directions are in training
        pairs, layout_ids = [(src, dst), (dst, src)], [args.layout_id]
    else:
        pairs = {"train": list(TRAIN_PAIRS), "held-out": [HELD_OUT_PAIR],
                 "all": list(TRAIN_PAIRS) + [HELD_OUT_PAIR]}[args.pairs]
        layout_ids = list(range(len(layouts)))

    out = Path("outputs/eval") / args.name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "eval_log.csv"

    sim = PiperStackingScene(layouts[layout_ids[0]], show_viewer=args.viewer and not args.live_cameras)

    live = None
    if args.live_cameras:
        import cv2

        from axibo.sim import CAMERA_NAMES

        def live(trace, frame, label: str, status: str):
            """Three camera panels with the trial and milestone state drawn on the top one."""
            obs = sim.observe().env(0)
            panels = []
            for name in CAMERA_NAMES:
                img = np.ascontiguousarray(cv2.resize(obs.images[name], (args.live_size, args.live_size)))
                cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(img)
            board = np.hstack(panels)
            for i, line in enumerate((label, status)):
                cv2.putText(board, line, (8, args.live_size - 34 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("eval: policy cameras (q aborts)", cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
    runner = SmolVLARunner(sim.action_low, sim.action_high, checkpoint=args.checkpoint, device=args.device)
    exec_steps = min(args.exec_steps, runner.chunk_size)
    settle_steps = int(args.settle_s * sim.cfg.control_hz)

    trials = [(lid, pair, seed) for lid in layout_ids for pair in pairs for seed in seeds]
    print(f"{len(trials)} trials = {len(layout_ids)} layouts x {len(pairs)} pairs x {len(seeds)} seeds | "
          f"{exec_steps}/{runner.chunk_size} actions per chunk, max {args.max_chunks} chunks "
          f"({args.max_chunks * exec_steps / sim.cfg.control_hz:.1f} s) + {args.settle_s:.0f} s settle -> {out}")

    rows, t_start = [], time.time()
    with log_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    for trial, (layout_id, (s_i, d_i), seed) in enumerate(trials):
        source, destination = OBJECT_NAMES[s_i], OBJECT_NAMES[d_i]
        task = f"put the {source} on the {destination}"
        sim.reseed(seed)
        sim.reset(object_poses=layouts[layout_id])
        for _ in range(10):
            sim.step_control()
        runner.reset()
        runner.latencies_s.clear()
        import torch

        torch.manual_seed(seed)  # flow-matching sampling noise

        trace = Trace.start(sim, 0, source, destination)
        obs = sim.observe().env(0)
        trace.record(sim, 0, obs.state[6])
        countdown, chunks, clipped, step = None, 0, 0, 0
        transported = False
        seen = {"lift": False, "transp": False, "release": False}
        while True:
            chunk = runner.predict_chunk(obs, task)
            chunks += 1
            for action in chunk[:exec_steps]:
                clipped += int(sim.apply_action(action[None])[0])
                sim.step_control()
                step += 1
                frame = trace.record(sim, 0, (sim.qpos()[0][6] - sim.qpos()[0][7]) / 2)
                transported = transported or trace.arrived_now()
                if countdown is None:
                    if transported and trace.release_detected_now():
                        countdown = settle_steps  # keep acting; the release only starts the clock
                else:
                    countdown -= 1
                    if frame.gripper < trace.tol.closed_grip and trace.above_dst(frame):
                        countdown = None  # a genuine retry cancels the window (see axibo/rollout.py)
                if live:
                    seen["lift"] |= trace.lifted(frame)
                    seen["transp"] |= trace.arrived(frame)
                    seen["release"] |= countdown is not None
                    if step % args.live_every == 0:
                        done = " ".join(f"{k}:{'Y' if v else '-'}" for k, v in seen.items())
                        live(trace, frame, f"[{trial + 1}/{len(trials)}] {task}",
                             f"chunk {chunks}/{args.max_chunks} step {step}  {done}"
                             + (f"  settle {countdown}" if countdown is not None else ""))
                if countdown is not None and countdown <= 0:
                    break
            if (countdown is not None and countdown <= 0) or chunks >= args.max_chunks:
                break
            obs = sim.observe().env(0)

        if countdown is None:
            # Budget expired mid-motion: hold the last action and let the scene settle before judging, otherwise a
            # placement made on the final chunk is assessed while the object is still coming to rest.
            for _ in range(settle_steps):
                sim.step_control()
                step += 1
                trace.record(sim, 0, (sim.qpos()[0][6] - sim.qpos()[0][7]) / 2)

        a = analyse(trace)
        lat = np.array(runner.latencies_s[1:] or runner.latencies_s) * 1000
        row = {
            "trial": trial, "seed": seed, "layout_id": layout_id, "source": source, "destination": destination,
            "task": task, "outcome": a.outcome, "chunks": chunks, "clipped_steps": clipped,
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 1), **a.metrics,
        }
        rows.append(row)
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
        print(f"  [{trial + 1}/{len(trials)}] layout {layout_id} seed {seed} {task:38} -> {a.outcome}")

    # --- summary (same tables as eval_report.py) ---
    from axibo.report import print_run, summarise

    meta = {
        "checkpoint": args.checkpoint,
        "layouts": args.layouts,
        "pairs_mode": args.pairs,
        "seeds": seeds,
        "exec_steps": exec_steps,
        "chunk_size": runner.chunk_size,
        "max_chunks": args.max_chunks,
        "settle_s": args.settle_s,
    }
    print_run(rows, title=out.name, meta=meta)
    summary = {**meta, **summarise(rows), "wall_s": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {log_path} and {out / 'summary.json'} ({summary['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
