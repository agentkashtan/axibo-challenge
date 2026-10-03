"""Task 3: evaluate the policy with a queue-based controller and a simulated inference latency.

Same protocol, outcome taxonomy and outputs as eval_policy.py (which is left untouched, so the Task 2 numbers
cannot shift), but the rollout runs through axibo/rollout.py: actions live in a queue, a new chunk is requested
when the queue drops to --queue-threshold, and the measured inference time is converted into control steps during
which the robot keeps executing what it already had. One knob covers both conditions:

    --queue-threshold 0    naive synchronous: the queue empties, the arm holds while the chunk is computed
    --queue-threshold 10   asynchronous: the chunk lands before the queue runs dry, so there is no stall

    source .venv/bin/activate
    python eval_async.py --checkpoint outputs/train/main_250v3/checkpoints/045000/pretrained_model \\
        --layouts data/layouts/eval_100_v228.csv --queue-threshold 0  --name async_k0
    python eval_async.py --checkpoint ... --layouts ... --queue-threshold 10 --name async_k10 --save-traces 20
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import genesis as gs

from axibo.backend import gs_backend, torch_device
from axibo.value import ADVANTAGE_POSITIVE

LOG_FIELDS = [
    "trial", "seed", "layout_id", "source", "destination", "task", "outcome", "chunks", "steps",
    "lifted_step", "transported_step", "stacked_step", "release_step", "placed_step",
    "centered_at_release", "release_dz_mm", "release_xy_offset_mm", "release_speed_mm_s" "release_dst_moved_mm", "max_src_z_mm", "lift_threshold_mm", "stack_height_mm",
    "final_xy_offset_mm", "final_dz_error_mm", "final_src_speed_mm_s", "final_settle_disp_mm",
    "final_gripper_mm", "dst_tilt_deg", "dst_moved_mm", "third_moved_mm", "third_disturbed", "dst_toppled",
    "latency_p50_ms", "clipped_steps",
    # Task 3 columns
    "queue_threshold", "latency_steps", "stall_fraction", "stalled_steps", "switches",
    "vel_disc_boundary_mean", "vel_disc_boundary_max", "vel_disc_mean",
    "jerk_boundary_rms", "jerk_rms", "jerk_max", "chunk_disagreement", "discarded_mean",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layouts", required=True)
    p.add_argument("--name", required=True, help="results go to outputs/eval/<name>")
    p.add_argument("--num-layouts", type=int, default=None)
    p.add_argument("--layout-id", type=int, default=0, help="reversal mode: which layout")
    p.add_argument("--pairs", choices=("train", "held-out", "all", "reversal"), default="train")
    p.add_argument("--seeds", default="0")
    p.add_argument("--queue-threshold", type=int, default=0,
                   help="request a new chunk when the queue has this many actions left; 0 = synchronous")
    p.add_argument("--latency", choices=("fixed", "measured"), default="fixed",
                   help="fixed: --latency-ms for every call (reproducible); measured: real per-call wall clock")
    p.add_argument("--latency-ms", type=float, default=200.0)
    p.add_argument("--blend", choices=("none", "linear"), default="none",
                   help="overlap handling on arrival: none = hard switch, linear = ramp over the overlap "
                        "(needs k > L for a usable window: overlap = k - L)")
    p.add_argument("--add-tag", action="store_true",
                   help='append the RECAP improvement indicator ("Advantage: positive") to the prompt: the '
                        "policy is asked for high-advantage behaviour, which is how an advantage-conditioned "
                        "checkpoint is meant to be run. Pointless on a checkpoint not trained with it")
    p.add_argument("--max-steps", type=int, default=500, help="control-step budget per episode (500 = 16.7 s)")
    p.add_argument("--settle-s", type=float, default=4.0)
    p.add_argument("--save-traces", type=int, nargs="?", const=10**9, default=0,
                   help="dump per-step npz traces (optionally only the first N trials, plus every failure)")
    p.add_argument("--device", default=torch_device())
    p.add_argument("--log-switches", action="store_true",
                   help="print a line per chunk switch: latency, actions pulled, held, dropped, resume index")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--live-cameras", action="store_true")
    p.add_argument("--live-every", type=int, default=3)
    p.add_argument("--live-size", type=int, default=340)
    return p.parse_args()


def main():
    args = parse_args()
    gs.init(backend=gs_backend(), seed=0, logging_level="warning")

    from axibo.outcome import Trace, analyse  # noqa: F401  (Trace used via rollout)
    from axibo.policy import SmolVLARunner
    from axibo.report import print_run, summarise
    from axibo.rollout import RolloutConfig, run_episode
    from axibo.scene_builder import load_layouts_csv
    from axibo.scripted import HELD_OUT_PAIR, TRAIN_PAIRS
    from axibo.sim import OBJECT_NAMES, PiperStackingScene
    from axibo.smoothness import episode_metrics, trace_arrays

    layouts = load_layouts_csv(args.layouts)[: args.num_layouts]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.pairs == "reversal":
        src, dst = TRAIN_PAIRS[2]
        pairs, layout_ids = [(src, dst), (dst, src)], [args.layout_id]
    else:
        pairs = {"train": list(TRAIN_PAIRS), "held-out": [HELD_OUT_PAIR],
                 "all": list(TRAIN_PAIRS) + [HELD_OUT_PAIR]}[args.pairs]
        layout_ids = list(range(len(layouts)))

    out = Path("outputs/eval") / args.name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "eval_log.csv"
    if args.save_traces:
        (out / "traces").mkdir(exist_ok=True)

    sim = PiperStackingScene(layouts[layout_ids[0]], show_viewer=args.viewer and not args.live_cameras)

    live = None
    if args.live_cameras:
        import cv2

        from axibo.sim import CAMERA_NAMES

        def live(label: str, status: str):
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
            cv2.imshow("eval_async: policy cameras (q aborts)", cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    runner = SmolVLARunner(sim.action_low, sim.action_high, checkpoint=args.checkpoint, device=args.device)
    cfg = RolloutConfig(
        queue_threshold=args.queue_threshold,
        latency_mode=args.latency,
        latency_ms=args.latency_ms,
        max_steps=args.max_steps,
        settle_steps=int(args.settle_s * sim.cfg.control_hz),
        blend=args.blend,
        log_switches=args.log_switches,
    )
    if args.latency == "fixed":
        lat_steps = int(np.ceil(args.latency_ms * sim.cfg.control_hz / 1000.0))
        mode = "synchronous (stalls)" if args.queue_threshold < lat_steps else "asynchronous (no stall)"
        lat_desc = f"fixed {args.latency_ms:.0f} ms = {lat_steps} steps -> {mode}"
    else:
        lat_steps = None  # only known per call
        lat_desc = (f"measured per call (L = ceil(wall_clock * {sim.cfg.control_hz})); "
                    f"stall-free needs k >= L, and k < {runner.chunk_size} - L for periodic inference")

    trials = [(lid, pair, seed) for lid in layout_ids for pair in pairs for seed in seeds]
    print(f"{len(trials)} trials = {len(layout_ids)} layouts x {len(pairs)} pairs x {len(seeds)} seeds\n"
          f"  k={args.queue_threshold}, latency {lat_desc} | budget {args.max_steps} steps "
          f"({args.max_steps / sim.cfg.control_hz:.1f} s) + {args.settle_s:.0f} s settle -> {out}")

    rows, t_start = [], time.time()
    with log_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    for trial, (layout_id, (s_i, d_i), seed) in enumerate(trials):
        source, destination = OBJECT_NAMES[s_i], OBJECT_NAMES[d_i]
        task = f"put the {source} on the {destination}"
        # The policy sees the tag; the CSV, the per-pair grouping and the live overlay keep the bare task, so
        # runs with and without it stay directly comparable in eval_report.py.
        prompt = f"{task} {ADVANTAGE_POSITIVE}" if args.add_tag else task
        sim.reseed(seed)
        sim.reset(object_poses=layouts[layout_id])
        for _ in range(10):
            sim.step_control()
        runner.reset()
        runner.latencies_s.clear()
        import torch

        torch.manual_seed(seed)

        on_step = None
        if live:
            def on_step(step, frame, stats, trial=trial, task=task):
                if step % args.live_every == 0:
                    live(f"[{trial + 1}/{len(trials)}] {task}",
                         f"k={args.queue_threshold} step {step} calls {stats.calls} "
                         f"stalled {stats.stalled_steps} switches {len(stats.switch_steps)}")

        trace, stats = run_episode(sim, runner, prompt, source, destination, cfg, on_step=on_step)
        a = analyse(trace)
        sm = episode_metrics(trace, sim.cfg.control_hz)
        lat_ms = np.array(stats.latency_s[1:] or stats.latency_s) * 1000

        row = {
            "trial": trial, "seed": seed, "layout_id": layout_id, "source": source, "destination": destination,
            "task": task, "outcome": a.outcome, "chunks": stats.calls, "clipped_steps": stats.clipped_steps,
            "latency_p50_ms": round(float(np.percentile(lat_ms, 50)), 1),
            "queue_threshold": args.queue_threshold,
            "latency_steps": int(np.median(stats.latency_steps)) if stats.latency_steps else 0,
            "chunk_disagreement": round(float(np.mean(stats.chunk_disagreement)), 5) if stats.chunk_disagreement else 0.0,
            "discarded_mean": round(float(np.mean(stats.discarded)), 1) if stats.discarded else 0.0,
            **a.metrics,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in sm.items()},
        }
        row["steps"] = stats.steps
        rows.append(row)
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore").writerow(row)

        if args.save_traces and (trial < args.save_traces or a.outcome != "success"):
            np.savez_compressed(out / "traces" / f"trial_{trial:04d}.npz",
                                outcome=a.outcome, task=task, layout_id=layout_id, seed=seed,
                                queue_threshold=args.queue_threshold, **trace_arrays(trace, stats))

        print(f"  [{trial + 1}/{len(trials)}] L{layout_id} seed {seed} {task:38} -> {a.outcome:20} "
              f"calls {stats.calls:3d} stall {sm.get('stalled_steps', 0):3d}/{stats.steps:3d} ({sm.get('stall_fraction', 0):4.1%}) "
              f"velΔ@bnd {sm.get('vel_disc_boundary_max', 0):.3f}")

    # --- summary ---
    meta = {
        "checkpoint": args.checkpoint, "layouts": args.layouts, "pairs_mode": args.pairs, "seeds": seeds,
        "queue_threshold": args.queue_threshold, "latency_mode": args.latency,
        "latency_ms": args.latency_ms if args.latency == "fixed" else None,
        "latency_steps": lat_steps, "max_steps": args.max_steps, "settle_s": args.settle_s,
        "chunk_size": runner.chunk_size, "blend": args.blend, "add_tag": args.add_tag,
    }
    print_run(rows, title=out.name, meta=meta)

    all_lat = np.concatenate([[r["latency_p50_ms"]] for r in rows]) if rows else np.zeros(1)
    mean = lambda key: float(np.mean([r.get(key, 0.0) for r in rows]))  # noqa: E731
    smooth = {
        "stall_fraction": mean("stall_fraction"),
        "stalled_steps": mean("stalled_steps"),
        "steps": mean("steps"),
        "vel_disc_boundary_mean": mean("vel_disc_boundary_mean"),
        "vel_disc_boundary_max": mean("vel_disc_boundary_max"),
        "vel_disc_mean": mean("vel_disc_mean"),
        "jerk_boundary_rms": mean("jerk_boundary_rms"),
        "jerk_rms": mean("jerk_rms"),
        "chunk_disagreement": mean("chunk_disagreement"),
        "calls_per_episode": mean("chunks"),
    }
    print("\n  smoothness (mean over trials)")
    for k, v in smooth.items():
        print(f"    {k:26} {v:10.4f}")
    print(f"    {'latency p50/p99 ms':26} {np.percentile(all_lat, 50):6.0f} / {np.percentile(all_lat, 99):.0f}")

    summary = {**meta, **summarise(rows), "smoothness": smooth, "wall_s": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {log_path} and {out / 'summary.json'} ({summary['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
