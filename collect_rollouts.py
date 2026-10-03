"""RECAP stage A: collect the policy's own rollouts, plus a scripted correction for each placement failure.

Two passes per trial.

    pass 1   run the policy, record every frame, label the episode with axibo.outcome.analyse
    pass 2   only if pass 1 failed as `misplaced` / `released_in_air`: reset the same layout, REPLAY pass 1's
             recorded actions (no inference) until the transport milestone latches, then hand to IK, which
             places the object compensating for however off-centre it is in the jaws

Pass 2 needs no determinism. The correction only has to be self-consistent, so the handover uses the transport
moment detected in the replay itself and plans from whatever state the replay actually reached - which matters,
because identical runs of this sim diverge from trial 2 onward (Genesis solver state appears to survive reset).
Replaying costs no inference, so a correction is ~190 sim steps plus ~100 of scripted placement.

Collected synchronously (k=0) on purpose. The queue empties before each new chunk, so every recorded
50-frame block is exactly one chunk prediction and the stalls fall at the block boundaries - skipping them
concatenates the chunks with nothing interleaved, which is the structure the policy is trained to produce
(measured: chunk_disagreement is 0.0 at k=0, i.e. no splices anywhere). At k>0 the replan period is 50-k and
chunks are consumed from index L, so any 50-action training window spans a splice. k=0 is also the controller
the Task 4 baseline is measured with, which is what RECAP's advantage is defined relative to.

Why those two outcomes: after the taxonomy fix, 18 of 20 failures on `eval_100_v228` are placement failures
where the object did reach the destination (release offsets 19.7-37.3 mm, destination untouched at release), and
inspection of the replays showed the arm aiming correctly but holding the object off-centre. Both have a
transport moment to hand over at; `no_grasp` has none, so it gets no correction.

Outputs, next to each other so neither can drift from the other:

    <out>/                 a LeRobot dataset, one episode per rollout plus one per accepted correction
    <out>/rollout_log.csv  per-episode outcome, milestones and provenance (LeRobot has no slot for an outcome)

Returns are deliberately NOT stored: R_t is a closed-form function of frame_index plus the per-episode scalars
in the CSV, so C_fail and the reward definition stay changeable without re-collecting.

    source .venv/bin/activate
    python collect_rollouts.py --checkpoint outputs/train/main_250v3/checkpoints/045000/pretrained_model \\
        --layouts data/layouts/rollout_150_v771.csv --num-layouts 2 --out data/lerobot/recap_v1 --viewer
"""

import argparse
import bisect
import csv
import time
from pathlib import Path

import numpy as np

import genesis as gs

from axibo.backend import gs_backend, torch_device

# Failures worth correcting: the object reached the destination, so there is a transport moment to take over at.
CORRECTABLE = ("misplaced", "released_in_air")

LOG_FIELDS = [
    "episode_index", "kind", "outcome", "parent_episode", "steps",
    "lifted_step", "transported_step", "stacked_step", "release_step",
    "layout_id", "source", "destination", "task", "seed",
    "delta_mm", "release_xy_offset_mm", "release_dz_mm", "final_xy_offset_mm", "final_dz_error_mm",
    "ik_err_mm", "note",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layouts", required=True)
    p.add_argument("--out", required=True, help="dataset directory; must not exist")
    p.add_argument("--repo-id", default="local/axibo_recap")
    p.add_argument("--num-layouts", type=int, default=None)
    p.add_argument("--seeds", default="0")
    p.add_argument("--queue-threshold", type=int, default=0,
                   help="k for the rollout controller; 0 (synchronous) makes each recorded 50-frame block one "
                        "chunk prediction, matching what the policy is trained to produce")
    p.add_argument("--latency", choices=("fixed", "measured"), default="measured")
    p.add_argument("--latency-ms", type=float, default=130.0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--settle-s", type=float, default=4.0)
    p.add_argument("--post-release-frames", type=int, default=30,
                   help="frames recorded after the release; 0 would drop the gripper-opening action itself")
    p.add_argument("--no-corrections", action="store_true", help="pass 1 only")
    p.add_argument("--device", default=torch_device())
    p.add_argument("--viewer", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; choose another --out or remove it")
    gs.init(backend=gs_backend(), seed=0, logging_level="warning")

    from concurrent.futures import ThreadPoolExecutor

    from lerobot.datasets import LeRobotDataset

    from axibo.outcome import analyse
    from axibo.policy import SmolVLARunner
    from axibo.recording import EpisodeWriter, make_features
    from axibo.rollout import RolloutConfig, run_episode
    from axibo.scene_builder import load_layouts_csv
    from axibo.scripted import TRAIN_PAIRS, PlanningError, plan_place_from_here
    from axibo.sim import OBJECT_NAMES, PiperStackingScene, SimConfig

    layouts = load_layouts_csv(args.layouts)[: args.num_layouts]
    seeds = [int(s) for s in args.seeds.split(",")]
    trials = [(lid, pair, seed) for lid in range(len(layouts)) for pair in TRAIN_PAIRS for seed in seeds]

    cfg = SimConfig()
    sim = PiperStackingScene(layouts[0], cfg=cfg, show_viewer=args.viewer)
    runner = SmolVLARunner(sim.action_low, sim.action_high, checkpoint=args.checkpoint, device=args.device)
    rcfg = RolloutConfig(
        queue_threshold=args.queue_threshold, latency_mode=args.latency, latency_ms=args.latency_ms,
        max_steps=args.max_steps, settle_steps=int(args.settle_s * cfg.control_hz), blend="none",
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=cfg.control_hz, features=make_features(cfg), root=out, robot_type="piperx",
        use_videos=True, video_backend="pyav",
    )
    log_path = out / "rollout_log.csv"
    with log_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
    pool = ThreadPoolExecutor(max_workers=8)

    def log(row: dict):
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore").writerow(row)

    def log_dropped(kind: str, parent: int, note: str, **extra):
        """Record a correction that never entered the dataset. Counted in the summary either way, but without a
        row there is no way to tell a skip from a rejection, or to check a rejection's geometry - which is how a
        sign error in the delta compensation would show up."""
        log({"episode_index": "", "kind": kind, "parent_episode": parent, "note": note,
             "layout_id": layout_id, "source": source, "destination": destination, "task": task, "seed": seed,
             **extra})

    def written_index(kept: list[int], sim_step) -> int | None:
        """A milestone's sim step translated into the recorded episode's frame index.

        Stalled frames are skipped and recording stops after the release, so the two numbering schemes differ.
        Returns None past the end of what was recorded - the milestone happened, but not inside the episode the
        dataset holds, so no return can be anchored to it.
        """
        if sim_step is None:
            return None
        i = bisect.bisect_left(kept, sim_step)
        return i if i < len(kept) else None

    def record_frame(writer: EpisodeWriter, action: np.ndarray):
        """Capture the observation the action is being issued from, pairing obs_t with action_t."""
        parts = []
        for name in OBJECT_NAMES:
            pos, quat = sim.object_pose(name)
            parts += [pos[0], quat[0]]
        writer.add(sim.observe().env(0), action, sim.qpos()[0], np.concatenate(parts))

    print(f"{len(trials)} trials = {len(layouts)} layouts x {len(TRAIN_PAIRS)} train pairs x {len(seeds)} seeds "
          f"-> {out}\n  k={args.queue_threshold}, corrections for {CORRECTABLE}")
    stats = {"rollouts": 0, "failures": 0, "attempted": 0, "accepted": 0, "rejected": 0, "skipped": 0}
    t_start = time.time()

    for trial, (layout_id, (s_i, d_i), seed) in enumerate(trials):
        source, destination = OBJECT_NAMES[s_i], OBJECT_NAMES[d_i]
        task = f"put the {source} on the {destination}"

        # ---- pass 1: the policy's own rollout -------------------------------------------------------------
        sim.reseed(seed)
        sim.reset(object_poses=layouts[layout_id])
        for _ in range(10):
            sim.step_control()
        runner.reset()
        import torch

        torch.manual_seed(seed)

        writer = EpisodeWriter(dataset, dataset.meta.total_episodes, task, pool)
        # `kept` maps written frames back to sim steps. Stalled frames are skipped and recording stops after the
        # release, so written index != sim step, and every MC return is computed from written indices.
        kept: list[int] = []
        cut: dict[str, int | None] = {"after": None}

        def on_step(step, frame, stats, cut=cut):
            # stats.detected_step is the step the settle window opened on, i.e. the release after transport.
            if cut["after"] is None and stats.detected_step is not None:
                cut["after"] = stats.detected_step + args.post_release_frames

        def on_action(step, action, stalled, writer=writer, kept=kept, cut=cut):
            if stalled:  # the policy was not queried here; the previous command was re-sent
                return
            if cut["after"] is not None and step > cut["after"]:
                return  # the episode keeps running (the verdict needs the full settle), we just stop recording
            record_frame(writer, action)
            kept.append(step)

        trace, rstats = run_episode(sim, runner, task, source, destination, rcfg,
                                    on_step=on_step, on_action=on_action)
        a = analyse(trace)
        stats["rollouts"] += 1
        stats["failures"] += a.outcome != "success"

        ep = writer.episode_index
        dataset.save_episode(episode_data=writer.episode_buffer())
        log({"episode_index": ep, "kind": "rollout", "outcome": a.outcome, "parent_episode": "",
             "steps": writer.size,
             **{k: written_index(kept, a.metrics.get(k))
                for k in ("lifted_step", "transported_step", "stacked_step", "release_step")},
             # release_* is the geometry when the jaws opened; final_* is after the object landed and rolled.
             # Only release_xy_offset_mm is comparable with a correction's delta_mm, which is what tests
             # whether a misplacement is the grasp offset carried forward or the arm also mis-aiming.
             **{k: a.metrics.get(k) for k in ("release_xy_offset_mm", "release_dz_mm",
                                             "final_xy_offset_mm", "final_dz_error_mm")},
             "layout_id": layout_id, "source": source, "destination": destination, "task": task, "seed": seed})

        print(f"  [{trial + 1}/{len(trials)}] L{layout_id} {task:38} -> {a.outcome:18} "
              f"ep {ep:4d} {writer.size:4d} frames", end="")

        if args.no_corrections or a.outcome not in CORRECTABLE:
            print()
            continue

        # ---- pass 2: replay to the transport moment, then scripted placement ------------------------------
        stats["attempted"] += 1
        actions = np.asarray([f.action for f in trace.frames], dtype=np.float32)
        sim.reseed(seed)
        sim.reset(object_poses=layouts[layout_id])
        for _ in range(10):
            sim.step_control()

        cwriter = EpisodeWriter(dataset, dataset.meta.total_episodes, task, pool)
        from axibo.outcome import Trace

        ctrace = Trace.start(sim, 0, source, destination, tol=trace.tol)
        handover = None
        for step, action in enumerate(actions):
            record_frame(cwriter, action)
            sim.apply_action(action[None])
            sim.step_control()
            q = sim.qpos()[0]
            ctrace.record(sim, 0, (q[6] - q[7]) / 2, qpos=q, action=action)
            if ctrace.arrived_now():  # the replay's own transport moment, not pass 1's
                handover = step
                break

        if handover is None:
            stats["skipped"] += 1
            log_dropped("correction_skipped", ep, "replay never reached the destination")
            print("  | correction skipped: replay never reached the destination")
            continue

        try:
            plan = plan_place_from_here(sim, source, destination, env_idx=0)
        except PlanningError as e:
            stats["skipped"] += 1
            log_dropped("correction_skipped", ep, f"planning: {e}", note=f"handover@{handover}")
            print(f"  | correction skipped: {e}")
            continue

        for action in plan.actions():
            record_frame(cwriter, action)
            sim.apply_action(action[None])
            sim.step_control()
            q = sim.qpos()[0]
            ctrace.record(sim, 0, (q[6] - q[7]) / 2, qpos=q, action=action)
        for _ in range(int(args.settle_s * cfg.control_hz)):  # let it settle before judging
            sim.step_control()
            q = sim.qpos()[0]
            ctrace.record(sim, 0, (q[6] - q[7]) / 2, qpos=q, action=plan.actions()[-1])

        # The correction's recorded frames map 1:1 onto ctrace indices (record_frame and ctrace.record run in
        # lockstep, and only the settle loop records without writing), so its milestones need no translation.
        # stacked_step lands past the recorded length, which is correct: every written frame precedes the latch.
        c = analyse(ctrace)
        delta_mm = float(np.linalg.norm(plan.delta_xy)) * 1000
        if c.outcome != "success":  # only keep corrections that actually worked
            stats["rejected"] += 1
            log_dropped("correction_rejected", ep, f"handover@{handover} outcome={c.outcome}",
                        outcome=c.outcome, delta_mm=round(delta_mm, 2),
                        ik_err_mm=round(plan.max_ik_pos_error * 1000, 2),
                        final_xy_offset_mm=c.metrics["final_xy_offset_mm"],
                        final_dz_error_mm=c.metrics["final_dz_error_mm"])
            print(f"  | correction rejected: {c.outcome} (delta {delta_mm:.1f} mm, "
                  f"xy {c.metrics['final_xy_offset_mm']:.1f} mm)")
            continue

        stats["accepted"] += 1
        cep = cwriter.episode_index
        dataset.save_episode(episode_data=cwriter.episode_buffer())
        log({"episode_index": cep, "kind": "correction", "outcome": c.outcome, "parent_episode": ep,
             "steps": cwriter.size, **{k: c.metrics.get(k) for k in
                                       ("lifted_step", "transported_step", "stacked_step", "release_step",
                                        "release_xy_offset_mm", "release_dz_mm",
                                        "final_xy_offset_mm", "final_dz_error_mm")},
             "layout_id": layout_id, "source": source, "destination": destination, "task": task, "seed": seed,
             "delta_mm": round(delta_mm, 2), "ik_err_mm": round(plan.max_ik_pos_error * 1000, 2),
             "note": f"handover@{handover}"})
        print(f"  | corrected ep {cep} (delta {delta_mm:.1f} mm -> xy {c.metrics['final_xy_offset_mm']:.1f} mm)")

    pool.shutdown(wait=True)
    print(f"\n  {stats['rollouts']} rollouts, {stats['failures']} failures "
          f"({stats['failures'] / max(1, stats['rollouts']):.1%})")
    print(f"  corrections: {stats['attempted']} attempted, {stats['accepted']} accepted, "
          f"{stats['rejected']} rejected (ran but did not stack), {stats['skipped']} skipped (no handover)")
    print(f"  {dataset.meta.total_episodes} episodes in {out} ({time.time() - t_start:.0f} s)")
    print(f"  wrote {log_path}")


if __name__ == "__main__":
    main()
