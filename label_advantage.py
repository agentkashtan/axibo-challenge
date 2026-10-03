"""RECAP stage D, part 1: turn the trained value function into a per-frame binary improvement indicator.

    source .venv/bin/activate
    python label_advantage.py --dataset data/lerobot/v322_data_for_recap \
                              --values outputs/value/nc_h64_dropout_only

No model inference happens here: stage C already wrote E[V] for every frame, so this is arithmetic over that
array. The advantage is the paper's N-step form (section V-D):

    A_t = sum_{t'=t}^{t+N-1} r_t'  +  V(o_{t+N})  -  V(o_t)
        = V(o_{t+N}) - N/scale - V(o_t)                            inside an episode
        = R_t - V(o_t)                                             when t+N runs past the end

The second line is exact rather than an approximation: once the lookahead passes the terminal, the N-step
return bootstrapped at that terminal *is* the full Monte-Carlo return, which `episode_returns` already gives.
That removes any need to hand-code the terminal reward (0 on success, -C_fail otherwise) a second time.

N defaults to the action chunk (50). That alignment is deliberate: SmolVLA is trained on (o_t, a_t..t+49), so
the advantage then scores exactly the decision the policy makes at o_t, no more and no less.

The threshold is the paper's: "We set eps_l to the 30% percentile of values predicted by the value function for
the task l" - per task, data-driven, no magic constant. Two deliberate departures, both stated in the report:

- the percentile is taken over the **advantage** distribution, not over V. As written the threshold comes from
  the spread of V but is compared against A, and those are not on the same scale here: our V spans about
  -1.0..-0.2 while A is a difference centred near -N/scale (~ -0.068). The 30th percentile of V is around
  -0.5, so `A > eps` would be true for essentially every frame and the indicator would be constant. Taking
  the percentile over A keeps the paper's intent - fix the split rate rather than the scale.
- "task" is the ordered pair. Cylinder destinations succeed 57% vs 71% for cubes, so a single global threshold
  would mark the cylinder pairs negative far more often and the indicator would encode *which pair it is*
  rather than *which action was taken*. Per-pair percentiles remove that by construction.

Corrections are forced I_t = True (paper), so they are excluded from the percentile - otherwise the
forced-positive frames would drag the threshold they are exempt from.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from axibo.backend import torch_device
from axibo.value import ReturnConfig, ValueMLP, expected_value, load_rollout_data

PCTS = (1, 5, 10, 20, 30, 40, 50, 70, 90, 99)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="LeRobot dataset root from collect_rollouts.py")
    p.add_argument("--values", required=True, help="a stage-C run dir holding value.pt and values.npy")
    p.add_argument("--horizon", type=int, default=50, help="N; defaults to the action chunk")
    p.add_argument("--percentile", type=float, default=30.0, help="paper: 30")
    p.add_argument("--eps-from", default=None,
                   help="reuse eps from an existing thresholds.json instead of computing it. This is how the "
                        "demos get labelled: eps estimated on the rollouts is applied unchanged, which tests "
                        "whether V and the threshold transfer to a different behaviour distribution")
    p.add_argument("--out", default=None, help="default: <values>/advantage_N<horizon>_p<percentile>")
    return p.parse_args()


def advantages(data, values: np.ndarray, horizon: int, scale: float) -> np.ndarray:
    """(N,) advantage per frame, computed strictly within each episode.

    The reward sum is counted, not assumed. `r_t = -1` only *until the stack exists*; at and after the latch it
    is 0, because nothing is being lost by waiting once the task is done. Charging a flat -N/scale everywhere
    (as an earlier version did) pushed every post-completion frame about 0.068 below where it belonged - larger
    than any eps, so those frames were forced negative regardless of what V said, which amounts to telling the
    policy that holding still after a successful placement is a bad action.
    """
    adv = np.full(len(values), np.nan, dtype=np.float32)
    for ep, info in data.episodes.items():
        f, t = data.frame_of_episode[ep]
        n = t - f
        v = values[f:t]
        a = np.empty(n, dtype=np.float32)
        k = max(0, n - horizon)              # frames whose lookahead stays inside the episode
        if k:
            if info.outcome == "success":
                # Same effective latch episode_returns uses: a success whose stack was confirmed after the
                # recording ended is anchored to the last written frame.
                latch = info.stacked_step if info.stacked_step is not None else n
                costed = np.clip(latch - np.arange(k), 0, horizon)   # frames before the latch in each window
            else:
                costed = np.full(k, horizon)                          # a failure never stops accruing -1
            a[:k] = v[horizon:horizon + k] - costed / scale - v[:k]
        a[k:] = data.returns[f + k:t] - v[k:]  # lookahead passes the terminal -> the true MC return
        adv[f:t] = a
    return adv


def main():
    args = parse_args()
    vdir = Path(args.values)
    ckpt = torch.load(vdir / "value.pt", map_location="cpu", weights_only=False)
    # Reuse the checkpoint's reward constants exactly: a different c_fail or scale would silently change both
    # the boundary returns and the advantage units.
    cfg = ReturnConfig(c_fail=ckpt["c_fail"], scale=ckpt["scale"])
    print(f"value fn {vdir}  (c_fail={cfg.c_fail:g}, scale={cfg.scale:g}, hidden={ckpt['hidden']}, "
          f"trained_without_corrections={ckpt.get('exclude_corrections')})")

    data = load_rollout_data(args.dataset, cfg)

    # Score this dataset with the checkpoint rather than reading its values.npy: that file only covers the
    # dataset V was trained on, and scoring here is what lets the same V be applied to a second dataset (the
    # demos) so its labels are out-of-sample by construction.
    device = torch.device(torch_device())
    model = ValueMLP(n_in=data.features.shape[1], hidden=ckpt["hidden"], dropout=ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    X = torch.from_numpy(data.features).to(device)
    values = np.empty(len(data.returns), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(values), 8192):
            values[i:i + 8192] = expected_value(model(X[i:i + 8192])).cpu().numpy()
    values[data.episode_of_frame < 0] = np.nan
    print(f"  scored {int((~np.isnan(values)).sum())} frames   E[V] mean {np.nanmean(values):+.4f} "
          f"p10 {np.nanpercentile(values, 10):+.4f} p50 {np.nanpercentile(values, 50):+.4f} "
          f"p90 {np.nanpercentile(values, 90):+.4f}")

    adv = advantages(data, values, args.horizon, cfg.scale)
    pair_of = {ep: f"{i.source} -> {i.destination}" for ep, i in data.episodes.items()}

    # --- the distribution of A per pair, over ROLLOUT frames only (corrections are forced positive) -------
    frames = defaultdict(list)
    for ep, info in data.episodes.items():
        if info.kind == "correction":
            continue
        f, t = data.frame_of_episode[ep]
        frames[pair_of[ep]].append(adv[f:t])
    by_pair = {k: np.concatenate(v) for k, v in frames.items()}

    print(f"\n  advantage distribution per pair, N={args.horizon}, rollout frames only "
          f"(step cost = N/scale = {args.horizon / cfg.scale:.4f})")
    head = "  ".join(f"p{p}" .rjust(7) for p in PCTS)
    print(f"    {'pair':34} {'n':>7}  {head}")
    for pair in sorted(by_pair):
        a = by_pair[pair]
        row = "  ".join(f"{v:7.4f}" for v in np.percentile(a, PCTS))
        print(f"    {pair:34} {len(a):>7}  {row}")
    allf = np.concatenate(list(by_pair.values()))
    print(f"    {'ALL':34} {len(allf):>7}  " + "  ".join(f"{v:7.4f}" for v in np.percentile(allf, PCTS)))

    if args.eps_from:
        src = json.loads(Path(args.eps_from).read_text())
        eps = src["eps_per_pair"]
        missing = sorted(set(by_pair) - set(eps))
        if missing:
            raise SystemExit(f"--eps-from has no threshold for: {missing}")
        print(f"\n  eps reused from {args.eps_from} (estimated on {src.get('value_dir', '?')}'s data, "
              f"N={src['horizon']}, {src['percentile']:g}th percentile)")
    else:
        eps = {pair: float(np.percentile(a, args.percentile)) for pair, a in by_pair.items()}
        print(f"\n  eps per pair at the {args.percentile:g}th percentile of A")
    for pair in sorted(eps):
        hit = 100 * (by_pair[pair] > eps[pair]).mean()
        print(f"    {pair:34} {eps[pair]:+.4f}   -> {hit:5.1f}% positive on this dataset")

    # --- the indicator ------------------------------------------------------------------------------------
    ind = np.zeros(len(adv), dtype=bool)
    forced = 0
    for ep, info in data.episodes.items():
        f, t = data.frame_of_episode[ep]
        if info.kind == "correction":
            ind[f:t] = True                      # paper: corrections are forced positive
            forced += t - f
        else:
            ind[f:t] = adv[f:t] > eps[pair_of[ep]]

    def rate(mask) -> str:
        n = int(mask.sum())
        return f"{n:>7} {100 * ind[mask].mean():6.1f}%" if n else f"{0:>7}      -"

    valid = ~np.isnan(adv)
    print(f"\n  positive rate: {100 * ind[valid].mean():.1f}% overall "
          f"({forced} frames forced positive by {sum(1 for i in data.episodes.values() if i.kind == 'correction')} corrections)")

    print("\n  by outcome (rollouts only) - should be high for success, low for misplaced")
    per_outcome = defaultdict(lambda: np.zeros(len(adv), dtype=bool))
    for ep, info in data.episodes.items():
        if info.kind == "correction":
            continue
        f, t = data.frame_of_episode[ep]
        per_outcome[info.outcome][f:t] = True
    print(f"    {'outcome':22} {'frames':>7} {'positive':>8}")
    for outcome in sorted(per_outcome, key=lambda o: -ind[per_outcome[o]].mean()):
        print(f"    {outcome:22} {rate(per_outcome[outcome])}")

    print("\n  by pair (rollouts only)")
    print(f"    {'pair':34} {'frames':>7} {'positive':>8}")
    for pair in sorted(by_pair):
        m = np.zeros(len(adv), dtype=bool)
        for ep in data.episodes:
            if data.episodes[ep].kind != "correction" and pair_of[ep] == pair:
                f, t = data.frame_of_episode[ep]
                m[f:t] = True
        print(f"    {pair:34} {rate(m)}")

    # Per layout: the indicator must be MIXED. If some layouts are ~0% positive then asking for "positive" on
    # them at inference is an input combination the policy never saw - off-distribution exactly where it is
    # needed. This is the check that decides whether the conditioning can work at all.
    per_layout = defaultdict(list)
    for ep, info in data.episodes.items():
        if info.kind == "correction":
            continue
        f, t = data.frame_of_episode[ep]
        per_layout[info.layout_id].append(ind[f:t])
    rates = np.array([np.concatenate(v).mean() for v in per_layout.values()])
    print(f"\n  by layout ({len(rates)} layouts): positive rate p10 {np.percentile(rates, 10):.2f}  "
          f"p50 {np.median(rates):.2f}  p90 {np.percentile(rates, 90):.2f}")
    print(f"    layouts below 5% positive: {int((rates < 0.05).sum())}   above 95%: {int((rates > 0.95).sum())}"
          f"    <- both should be near zero, or the indicator encodes the layout rather than the action")

    out = Path(args.out) if args.out else vdir / f"advantage_N{args.horizon}_p{args.percentile:g}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "labels.npz", advantage=adv, indicator=ind)
    (out / "thresholds.json").write_text(json.dumps(
        {"horizon": args.horizon, "percentile": args.percentile, "c_fail": cfg.c_fail, "scale": cfg.scale,
         "value_dir": str(vdir), "eps_per_pair": eps, "eps_from": args.eps_from,
         "dataset": str(args.dataset),
         "positive_rate": float(ind[valid].mean())}, indent=2))
    print(f"\nwrote {out}/labels.npz, thresholds.json")


if __name__ == "__main__":
    main()
