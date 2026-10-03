"""Plot per-step joint traces saved by eval_async.py --save-traces.

Shows joint position and velocity against time with chunk boundaries marked and stalled spans shaded, so the
chunk-boundary stall is visible rather than only tabulated. Pass two traces to put the conditions side by side.

    source .venv/bin/activate
    python plot_trace.py outputs/eval/t3_k0_smoke/traces/trial_0000.npz
    python plot_trace.py outputs/eval/t3_k0_smoke/traces/trial_0000.npz \\
                         outputs/eval/t3_k10_smoke/traces/trial_0000.npz --out outputs/eval/stall.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HZ = 30


def spans(mask: np.ndarray):
    """Contiguous True runs of a boolean mask as (start, end) index pairs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def panel(ax_pos, ax_vel, path: Path, joints):
    d = np.load(path, allow_pickle=True)
    qpos, qvel = d["qpos"], d["qvel"]
    t = np.arange(len(qpos)) / HZ
    k = int(d["queue_threshold"]) if "queue_threshold" in d else -1
    stall = d["stalled"].mean()

    for j in joints:
        ax_pos.plot(t, qpos[:, j], lw=1.0, label=f"j{j + 1}")
        ax_vel.plot(t, qvel[:, j], lw=1.0)

    for ax in (ax_pos, ax_vel):
        for a, b in spans(d["stalled"]):
            ax.axvspan(a / HZ, b / HZ, color="tab:red", alpha=0.18, lw=0)
        for s in np.flatnonzero(d["switch"]):
            ax.axvline(s / HZ, color="tab:green", lw=0.8, alpha=0.7)

    ax_pos.set_title(f"{path.parent.parent.name}  k={k}  stall {stall:.1%}  ({str(d['outcome'])})", fontsize=9)
    ax_pos.set_ylabel("q [rad]")
    ax_vel.set_ylabel("dq/dt [rad/s]")
    ax_vel.set_xlabel("time [s]")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+", type=Path)
    p.add_argument("--joints", default="0,1,2,3,4,5", help="arm joint indices to draw")
    p.add_argument("--out", type=Path, default=Path("outputs/eval/trace.png"))
    args = p.parse_args()

    joints = [int(j) for j in args.joints.split(",")]
    n = len(args.traces)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 6), squeeze=False, sharex="col")
    for i, path in enumerate(args.traces):
        panel(axes[0][i], axes[1][i], path, joints)
    axes[0][0].legend(fontsize=7, ncol=3, loc="upper left")
    fig.suptitle("green = chunk boundary, red band = arm stalled waiting for inference", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
