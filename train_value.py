"""Train the RECAP value function on a collect_rollouts dataset (stage C).

V(state, command) -> 201 return bins, trained by cross-entropy against the Monte-Carlo return of each episode.
The whole numeric dataset is ~170k x 33 floats, so it is loaded once into memory and trained with a random
permutation per epoch - no DataLoader, no video decode.

    source .venv/bin/activate
    python train_value.py --dataset data/lerobot/v322_data_for_recap

Writes outputs/value/<dataset>/{value.pt, values.npy, metrics.csv}. `values.npy` holds E[V] for every frame in
dataset order, so stage D (advantage labelling at a chosen N and epsilon) is arithmetic on a file and never
re-runs the model.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from axibo.backend import torch_device
from axibo.value import (FEATURE_BLOCKS, N_BINS, ReturnConfig, ValueMLP, distributional_loss,
                         expected_value, load_rollout_data)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="LeRobot dataset root from collect_rollouts.py")
    p.add_argument("--out", default=None, help="default: outputs/value/<dataset name>")
    p.add_argument("--c-fail", type=float, default=250.0,
                   help="failure penalty in frames (default: one typical successful attempt)")
    p.add_argument("--scale", type=float, default=None,
                   help="return normalizer; default max(n_frames) + c_fail, measured from the dataset")
    p.add_argument("--hidden", type=int, default=128,
                   help="128 -> ~30k params. 512 overfit badly on ~650 episode-level labels")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Gaussian jitter on standardized inputs during training, in std units (try 0.1)")
    p.add_argument("--val-every", type=int, default=100,
                   help="validate every N optimizer steps; an epoch is ~650 steps, too coarse to find the min")
    p.add_argument("--patience", type=int, default=0,
                   help="stop after this many validations without improvement (0 = run all epochs)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay; 626 episode-level labels do not support many free parameters")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--val-fraction", type=float, default=0.05, help="fraction of LAYOUTS held out")
    p.add_argument("--exclude-corrections", action="store_true",
                   help="drop correction episodes: they replay their parent's grasp but succeed, so with both "
                        "present V sees opposite returns for the same lift-step state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=torch_device())
    return p.parse_args()


def report_return_separation(data) -> None:
    """Verification 1: every failure must score below every success at t=0.

    Below that, a failure that ends early outscores a slow success - early termination acting as a reward,
    which is the reward hack the challenge spec says to rule out. This is also the evidence for the report.
    """
    by_outcome = defaultdict(list)
    for ep, info in data.trainable().items():
        f, _ = data.frame_of_episode[ep]
        by_outcome[info.outcome].append((float(data.returns[f]), info.n_frames, info.stacked_step))
    print(f"\n  returns at t=0  (c_fail={data.cfg.c_fail:g}, scale={data.cfg.scale:g})")
    print(f"    {'outcome':22} {'n':>4}  {'R0 min':>8} {'R0 max':>8}   {'frames':>11}")
    for outcome in sorted(by_outcome, key=lambda o: -max(r for r, _, _ in by_outcome[o])):
        rows = by_outcome[outcome]
        r = [x for x, _, _ in rows]
        n = [x for _, x, _ in rows]
        print(f"    {outcome:22} {len(rows):>4}  {min(r):>8.3f} {max(r):>8.3f}   {min(n):>5}-{max(n):<5}")

    succ = [r for o, rows in by_outcome.items() if o == "success" for r, _, _ in rows]
    fail = [r for o, rows in by_outcome.items() if o != "success" for r, _, _ in rows]
    if succ and fail:
        gap = min(succ) - max(fail)
        print(f"\n    worst success {min(succ):.3f}  vs  best failure {max(fail):.3f}   "
              f"gap {gap:+.3f} = {gap * (N_BINS - 1):+.1f} bins")
        if gap <= 0:
            latch = [s for o, rows in by_outcome.items() if o == "success" for _, _, s in rows if s is not None]
            flen = [n for o, rows in by_outcome.items() if o != "success" for _, n, _ in rows]
            floor = max(latch) - min(flen)
            print(f"    *** OVERLAP. Floor is max(latch) - min(failure len) = {max(latch)} - {min(flen)} "
                  f"= {floor}.")
            if data.cfg.c_fail > floor:
                # c_fail already clears the floor, so the cause is a mislabelled success, not the constant.
                print(f"    c_fail={data.cfg.c_fail:g} already clears it, so this is a labelling problem, not "
                      f"c_fail - check the latchless-success count above. ***")
            else:
                print(f"    raise --c-fail above {floor}. ***")
    if abs(float(data.returns.min())) >= 1.0:
        n_clipped = int((data.returns <= -1.0).sum())
        print(f"    note: {n_clipped} frames sit at the bottom bin (-1.0) - expected for the longest failure")


def split_by_layout(data, val_fraction: float, seed: int):
    """Hold out whole layouts. Frames inside an episode share one episode-level label and are nearly
    identical, so a frame-level split leaks; and a correction shares its parent failure's layout, so a layout
    split also keeps the duplicated prefix on one side."""
    by_layout = defaultdict(list)
    for ep, info in data.trainable().items():
        by_layout[info.layout_id].append(ep)
    layouts = sorted(by_layout)
    rng = np.random.default_rng(seed)
    n_val = max(1, round(val_fraction * len(layouts)))
    val_layouts = set(rng.choice(layouts, size=n_val, replace=False).tolist())
    train_eps = [e for l in layouts if l not in val_layouts for e in by_layout[l]]
    val_eps = [e for l in sorted(val_layouts) for e in by_layout[l]]
    print(f"\n  split: {len(layouts) - n_val} train layouts / {n_val} val layouts "
          f"-> {len(train_eps)} / {len(val_eps)} episodes")
    return train_eps, val_eps


def frame_mask(data, episodes) -> np.ndarray:
    mask = np.zeros(len(data.returns), dtype=bool)
    for ep in episodes:
        f, t = data.frame_of_episode[ep]
        mask[f:t] = True
    return mask


def auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """P(score of a success > score of a failure). Mann-Whitney U, tie-corrected."""
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(scores)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


MILESTONES = ("lifted_step", "transported_step", "release_step")


def milestone_values(data, values: np.ndarray, episodes, milestone: str):
    """E[V] at a named milestone, plus the eventual success label.

    Episodes that never reached the milestone are excluded, which is the point: at `lifted_step` the question
    is whether V can see a bad *grasp* coming, and at `transported_step` whether it can see a bad *placement*
    coming. The grasp offset only becomes visible in `src_pos - dst_pos` once the object is over the
    destination, so the two can differ a lot - and which one carries signal is a reportable finding.
    """
    v, label, outcomes = [], [], []
    for ep in episodes:
        info = data.episodes[ep]
        step = getattr(info, milestone)
        if step is None or step >= info.n_frames:
            continue
        f, _ = data.frame_of_episode[ep]
        v.append(values[f + step])
        label.append(info.outcome == "success")
        outcomes.append(info.outcome)
    return np.array(v), np.array(label, dtype=bool), outcomes


def main():
    args = parse_args()
    root = Path(args.dataset)
    out = Path(args.out) if args.out else Path("outputs/value") / root.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cfg = ReturnConfig(c_fail=args.c_fail, scale=args.scale)
    print(f"loading {root}")
    data = load_rollout_data(root, cfg, ("correction",) if args.exclude_corrections else ())
    used = int((data.episode_of_frame >= 0).sum())
    print(f"  {len(data.trainable())} trainable of {len(data.episodes)} episodes, {used} frames, {data.features.shape[1]} features "
          f"({', '.join(f'{n} {k}' for k, n in FEATURE_BLOCKS)})")
    report_return_separation(data)

    train_eps, val_eps = split_by_layout(data, args.val_fraction, args.seed)
    tr, va = frame_mask(data, train_eps), frame_mask(data, val_eps)

    device = torch.device(args.device)
    model = ValueMLP(n_in=data.features.shape[1], hidden=args.hidden, dropout=args.dropout,
                     noise_std=args.noise_std).to(device)
    model.set_normalization(data.features[tr].mean(0), data.features[tr].std(0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    X = torch.from_numpy(data.features).to(device)
    y = torch.from_numpy(data.returns).to(device)
    tr_idx = torch.from_numpy(np.flatnonzero(tr)).to(device)
    va_idx = torch.from_numpy(np.flatnonzero(va)).to(device)
    print(f"  {len(tr_idx)} train frames / {len(va_idx)} val frames, device={device}, "
          f"{sum(p.numel() for p in model.parameters())} params")

    def evaluate(idx) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for s in range(0, len(idx), 4096):
                b = idx[s:s + 4096]
                total += distributional_loss(model(X[b]), y[b]).item() * len(b)
        return total / len(idx)

    steps_per_epoch = max(1, (len(tr_idx) + args.batch_size - 1) // args.batch_size)
    print(f"\n  {'step':>7} {'epoch':>6} {'train':>9} {'val':>9}   ({steps_per_epoch} steps/epoch)")
    metrics = []
    best = {"val": float("inf"), "step": 0, "epoch": 0, "state": None}
    step, since_best, stop = 0, 0, False
    for epoch in range(1, args.epochs + 1):
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        run, seen = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            model.train()
            b = perm[s:s + args.batch_size]
            loss = distributional_loss(model(X[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * len(b)
            seen += len(b)
            step += 1
            if step % args.val_every:
                continue
            val = evaluate(va_idx)
            metrics.append({"step": step, "epoch": epoch, "train_loss": run / seen, "val_loss": val})
            improved = val < best["val"]
            if improved:
                best = {"val": val, "step": step, "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            since_best = 0 if improved else since_best + 1
            if improved or step % (args.val_every * 20) == 0:
                print(f"  {step:>7} {epoch:>6} {run / seen:>9.4f} {val:>9.4f}{'  *' if improved else ''}")
            run, seen = 0.0, 0
            if args.patience and since_best >= args.patience:
                print(f"  early stop: {since_best} validations without improvement")
                stop = True
                break
        if stop:
            break

    # Restore the lowest-val-loss weights: everything below, and the saved checkpoint, must describe one
    # model. Saving the final epoch would be the worst checkpoint of a run whose val loss rises throughout.
    print(f"\n  best val {best['val']:.4f} at step {best['step']} (epoch {best['epoch']} of {args.epochs})"
          f"   final {metrics[-1]['val_loss']:.4f}")
    model.load_state_dict(best["state"])

    # E[V] for every frame, in dataset order.
    model.eval()
    values = np.full(len(data.returns), np.nan, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(values), 8192):
            values[s:s + 8192] = expected_value(model(X[s:s + 8192])).cpu().numpy()
    # Frames of episodes that were excluded have zeroed features, so their V is meaningless: mark them NaN so
    # stage D cannot use them by accident.
    values[data.episode_of_frame < 0] = np.nan

    def fmt(x):
        return "  -  " if x is None else f"{x:.3f}"

    print("\n  AUC(E[V] at milestone -> eventual success).  val is the only honest column: `all` includes "
          "the\n  episodes the model trained on, so it measures memory, not skill.")
    print(f"    {'milestone':18} {'n_all':>6} {'AUC_all':>8}   {'n_val':>5} {'AUC_val':>8}")
    for m in MILESTONES:
        v_a, l_a, _ = milestone_values(data, values, sorted(data.trainable()), m)
        v_v, l_v, _ = milestone_values(data, values, val_eps, m)
        print(f"    {m:18} {len(v_a):>6} {fmt(auc(v_a, l_a)):>8}   {len(v_v):>5} {fmt(auc(v_v, l_v)):>8}")
    print("    above ~0.65 the advantage carries real signal. The grasp offset only shows up in the object's\n"
          "    position once it is over the destination, so transported_step may separate when lifted_step "
          "does not.")

    for m in ("lifted_step", "transported_step"):
        v_all, _, out_all = milestone_values(data, values, sorted(data.trainable()), m)
        by = defaultdict(list)
        for value, outcome in zip(v_all, out_all, strict=True):
            by[outcome].append(value)
        print(f"\n  E[V] at {m}, by eventual outcome")
        print(f"    {'outcome':22} {'n':>4} {'mean E[V]':>10}")
        for outcome in sorted(by, key=lambda o: -np.mean(by[o])):
            print(f"    {outcome:22} {len(by[outcome]):>4} {np.mean(by[outcome]):>10.3f}")

    torch.save({"state_dict": model.state_dict(), "c_fail": cfg.c_fail, "scale": cfg.scale,
                "n_bins": N_BINS, "feature_blocks": FEATURE_BLOCKS, "hidden": args.hidden,
                "dropout": args.dropout, "val_episodes": val_eps,
                "best_epoch": best["epoch"], "best_step": best["step"],
                "best_val_loss": best["val"], "noise_std": args.noise_std,
                "exclude_corrections": args.exclude_corrections}, out / "value.pt")
    np.save(out / "values.npy", values)
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(metrics)
    print(f"\nwrote {out}/value.pt, values.npy ({values.shape}), metrics.csv")


if __name__ == "__main__":
    main()
