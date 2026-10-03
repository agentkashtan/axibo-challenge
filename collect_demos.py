"""Collect scripted stacking demos in parallel Genesis envs and save them as a LeRobot dataset.

Jobs = each loaded layout x the 5 training pairs (the held-out pair is never collected). Every executed episode is
saved; success and metrics per episode go to <out>/collection_log.csv so failures can be filtered before training.

    source .venv/bin/activate
    python collect_demos.py --layouts data/layouts/pilot_50.csv --n-envs 5 --num-layouts 2
"""

import argparse
import csv
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

import genesis as gs

from axibo.backend import gs_backend

SETTLE_S = 2.0  # recorded after the plan; success is measured at the end of it
LOG_FIELDS = [
    "episode_index", "layout_id", "source", "destination", "task", "success", "reason", "n_frames",
    "xy_offset_mm", "dz_error_mm", "dst_tilt_deg", "third_moved_mm", "grasp_yaw_deg", "drop_yaw_deg", "clipped_steps",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layouts", required=True, help="layouts CSV from sample_layouts.py")
    p.add_argument("--n-envs", type=int, default=5, help="parallel Genesis environments")
    p.add_argument("--num-layouts", type=int, default=None, help="use only the first K layouts of the file")
    p.add_argument("--out", default=None, help="dataset root (default: data/lerobot/<layouts file stem>)")
    p.add_argument("--repo-id", default="local/piperx_stack")
    p.add_argument("--viewer", action="store_true", help="show the Genesis viewer")
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-record plays the demos (success still checked) without rendering or saving anything")
    return p.parse_args()


def object_poses_flat(sim, object_names) -> np.ndarray:
    """(N, 7 * n_objects): xyz + wxyz per object, for all envs at once."""
    parts = []
    for name in object_names:
        pos, quat = sim.object_pose(name)
        parts += [pos, quat]
    return np.concatenate(parts, axis=1)


def main():
    args = parse_args()
    gs.init(backend=gs_backend(), logging_level="warning")

    from lerobot.datasets import LeRobotDataset

    from axibo.recording import EpisodeWriter, make_features
    from axibo.scene_builder import load_layouts_csv
    from axibo.scripted import TRAIN_PAIRS, PlanningError, plan_demo
    from axibo.sim import OBJECT_NAMES, PiperStackingScene, SimConfig
    from axibo.success import stack_metrics, third_object

    layouts = load_layouts_csv(args.layouts)[: args.num_layouts]
    jobs = [(layout_id, pair) for layout_id in range(len(layouts)) for pair in TRAIN_PAIRS]
    out = Path(args.out or f"data/lerobot/{Path(args.layouts).stem}")
    if args.record and out.exists():
        raise SystemExit(f"{out} already exists; choose another --out or remove it")

    cfg = SimConfig()
    n_envs = min(args.n_envs, len(jobs))
    sim = PiperStackingScene(layouts[0], n_envs=n_envs, cfg=cfg, show_viewer=args.viewer)
    dataset, log_path = None, out / "collection_log.csv"
    if args.record:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=cfg.control_hz, features=make_features(cfg), root=out, robot_type="piperx",
            use_videos=True, video_backend="pyav",  # torchcodec needs system FFmpeg libs, which aren't installed
        )
        with log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    settle_steps = int(SETTLE_S * cfg.control_hz)
    pool = ThreadPoolExecutor(max_workers=8)
    t_start, results = time.time(), []
    print(f"{len(jobs)} jobs ({len(layouts)} layouts x {len(TRAIN_PAIRS)} pairs), {n_envs} envs -> "
          f"{out if args.record else 'not recording'}")

    progress = tqdm(total=len(jobs), desc="collect", unit="demo", dynamic_ncols=True)
    writers = {}
    try:
        for batch_start in range(0, len(jobs), n_envs):
            batch = jobs[batch_start : batch_start + n_envs]
            active = list(range(len(batch)))  # padded envs (last partial batch) just idle
            batch_layouts = [layouts[lid] for lid, _ in batch] + [layouts[batch[0][0]]] * (n_envs - len(batch))
            sim.reset(object_poses=batch_layouts)
            for _ in range(10):
                sim.step_control()

            plans, writers, rows, third_initial = {}, {}, {}, {}
            for env, (layout_id, pair) in enumerate(batch):
                src, dst = OBJECT_NAMES[pair[0]], OBJECT_NAMES[pair[1]]
                rows[env] = {"episode_index": -1, "layout_id": layout_id, "source": src, "destination": dst,
                             "task": f"put the {src} on the {dst}", "success": False, "reason": "", "n_frames": 0,
                             "clipped_steps": 0}
                try:
                    plans[env] = plan_demo(sim, layout_id, pair, args.layouts, env_idx=env)
                except PlanningError as e:
                    rows[env]["reason"] = f"planning: {e}"
                    active.remove(env)
                    continue
                rows[env].update(grasp_yaw_deg=plans[env].grasp_yaw_deg, drop_yaw_deg=plans[env].drop_yaw_deg)
                third_initial[env] = sim.object_pose(third_object(src, dst))[0][env].copy()

            if args.record:  # episode indices follow the order envs are saved in
                for i, env in enumerate(active):
                    writers[env] = EpisodeWriter(dataset, dataset.meta.total_episodes + i, plans[env].task, pool)
                    rows[env]["episode_index"] = writers[env].episode_index

            actions = {env: plans[env].actions() for env in active}
            episode_len = {env: len(actions[env]) + settle_steps for env in active}
            hold = sim.qpos()  # idle envs hold their current pose
            idle_action = np.column_stack([hold[:, :6], (hold[:, 6] - hold[:, 7]) / 2])
            n_steps = max(episode_len.values(), default=0)

            for t in tqdm(range(n_steps), desc=f"demos {batch_start + 1}-{batch_start + len(batch)}", unit="step",
                          leave=False, dynamic_ncols=True):
                if args.record:
                    obs, qpos, poses = sim.observe(), sim.qpos(), object_poses_flat(sim, OBJECT_NAMES)
                step_actions = idle_action.copy()
                for env in active:
                    step_actions[env] = actions[env][min(t, len(actions[env]) - 1)]  # hold last target while settling
                    if args.record and t < episode_len[env]:
                        writers[env].add(obs.env(env), step_actions[env], qpos[env], poses[env])
                clipped = sim.apply_action(step_actions)
                for env in active:
                    rows[env]["clipped_steps"] += int(clipped[env] and t < len(actions[env]))
                sim.step_control()

            for env in active:  # already in episode-index order
                rows[env].update(stack_metrics(sim, env, rows[env]["source"], rows[env]["destination"], third_initial[env]))
                rows[env]["reason"] = "" if rows[env]["success"] else "stack check failed"
                rows[env]["n_frames"] = episode_len[env]
                if args.record:
                    dataset.save_episode(episode_data=writers[env].episode_buffer())

            if args.record:
                with log_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                    for env in range(len(batch)):
                        writer.writerow(rows[env])
            results += [rows[env] for env in range(len(batch))]

            progress.update(len(batch))
            progress.set_postfix(
                saved=dataset.meta.total_episodes if args.record else "-",
                success=f"{sum(r['success'] for r in results)}/{len(results)}",
            )

    except KeyboardInterrupt:
        tqdm.write("interrupted: keeping the episodes saved so far")
        if args.record:  # temp frames of episodes that were never saved
            for w in writers.values():
                if w.episode_index >= dataset.meta.total_episodes:
                    for key in w.image_paths:
                        shutil.rmtree(dataset.root / "images" / key / f"episode-{w.episode_index:06d}", ignore_errors=True)
    finally:
        progress.close()
        if args.record:
            dataset.finalize()  # without this the parquet files have no footer and the dataset is unreadable
        pool.shutdown()


    elapsed = time.time() - t_start
    per_pair = Counter((r["source"], r["destination"]) for r in results)
    ok_pair = Counter((r["source"], r["destination"]) for r in results if r["success"])
    saved = f"{dataset.meta.total_episodes} episodes saved" if args.record else "nothing recorded"
    print(f"\ndone: {saved}, {sum(ok_pair.values())}/{len(results)} successful, "
          f"{elapsed / 60:.1f} min ({3600 * len(results) / elapsed:.0f} jobs/h)")
    for pair, n in per_pair.items():
        print(f"  {pair[0]:12s} -> {pair[1]:12s}: {ok_pair[pair]}/{n}")
    for r in results:
        if not r["success"]:
            print(f"  FAILED layout {r['layout_id']} {r['task']!r}: {r['reason']}")
    if args.record:
        print(f"log: {log_path}")


if __name__ == "__main__":
    main()
