"""Speed and acceleration distributions for one or more runs saved with --save-traces.

A single smoothness scalar hides the trade-off between the two conditions: a synchronous controller stands still at
every chunk boundary (mass at zero speed, big accelerations when it restarts), while an asynchronous one keeps
moving but changes plan more often (less mass at zero, more mid-size acceleration spikes). Time-averaged metrics
also flatter the stalling controller, because stalled steps contribute zero jerk. Distributions show both effects
without needing to know where the chunk boundaries are.

    source .venv/bin/activate
    python plot_motion_dist.py synctest synctestv1                    # opens a window
    python plot_motion_dist.py synctest synctestv1 --out dist.png     # save instead
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HZ = 30
ARM = slice(0, 6)
STILL = 0.02  # [rad/s] below this the arm counts as not moving


def load_run(name: str):
    """Pooled per-step speed and acceleration magnitudes over every saved trial of a run."""
    run = Path(name) if Path(name).is_dir() else Path("outputs/eval") / name
    files = sorted(glob.glob(str(run / "traces" / "*.npz")))
    if not files:
        raise SystemExit(f"no traces in {run / 'traces'} - run with --save-traces")
    speed, acc, cmd = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        qvel = d["qvel"][:, ARM]
        speed.append(np.abs(qvel).max(axis=1))
        acc.append(np.abs(np.diff(qvel, axis=0) * HZ).max(axis=1))
        # Step-to-step change of the commanded target: where a chunk switch actually shows up, since the PD
        # controller and the arm's inertia smooth it out of the measured motion.
        cmd.append(np.abs(np.diff(d["action"][:, ARM], axis=0)).max(axis=1))
    meta = {}
    summary = run / "summary.json"
    if summary.exists():
        m = json.loads(summary.read_text())
        meta = {"k": m.get("queue_threshold"), "latency": m.get("latency_mode"), "trials": len(files)}
    return np.concatenate(speed), np.concatenate(acc), np.concatenate(cmd), meta


def describe(label: str, speed: np.ndarray, acc: np.ndarray, cmd: np.ndarray, meta: dict) -> str:
    still = float((speed < STILL).mean())
    return (f"{label:14} k={meta.get('k', '?'):>3} | steps {len(speed):5d} | still {still:5.1%}\n"
            f"    speed  p50 {np.percentile(speed, 50):6.3f} p90 {np.percentile(speed, 90):6.3f} "
            f"p99 {np.percentile(speed, 99):6.3f} max {speed.max():6.3f}\n"
            f"    acc    p50 {np.percentile(acc, 50):6.2f} p90 {np.percentile(acc, 90):6.2f} "
            f"p99 {np.percentile(acc, 99):6.2f} max {acc.max():6.2f}\n"
            f"    |da|   p50 {np.percentile(cmd, 50):6.4f} p90 {np.percentile(cmd, 90):6.4f} "
            f"p99 {np.percentile(cmd, 99):6.4f} max {cmd.max():6.4f} | "
            f">0.05 rad: {float((cmd > 0.05).mean()):5.2%} of steps")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run names under outputs/eval/ (or run dirs)")
    p.add_argument("--out", type=Path, default=None, help="save instead of showing the window")
    p.add_argument("--bins", type=int, default=60)
    args = p.parse_args()

    data = [(name, *load_run(name)) for name in args.runs]

    print()
    for name, speed, acc, cmd, meta in data:
        print("  " + describe(name, speed, acc, cmd, meta))

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    speed_max = max(s.max() for _, s, _, _, _ in data)
    acc_max = max(np.percentile(a, 99.9) for _, _, a, _, _ in data)
    cmd_max = max(c.max() for _, _, _, c, _ in data)
    xs = np.linspace(0, 100, 400)

    for name, speed, acc, cmd, meta in data:
        lbl = f"{name} (k={meta.get('k', '?')})"
        axes[0][0].hist(speed, bins=args.bins, range=(0, speed_max), histtype="step", lw=1.4, label=lbl, density=True)
        axes[0][1].hist(acc, bins=args.bins, range=(0, acc_max), histtype="step", lw=1.4, label=lbl, density=True)
        axes[0][2].hist(cmd, bins=args.bins, range=(0, cmd_max), histtype="step", lw=1.4, label=lbl, density=True)
        axes[1][0].plot(xs, np.percentile(speed, xs), lw=1.4, label=lbl)
        axes[1][1].plot(xs, np.percentile(acc, xs), lw=1.4, label=lbl)
        axes[1][2].plot(xs, np.percentile(cmd, xs), lw=1.4, label=lbl)

    axes[0][0].set(title="joint speed distribution", xlabel="max |dq/dt| [rad/s]", ylabel="density", yscale="log")
    axes[0][1].set(title="joint acceleration distribution", xlabel="max |d2q/dt2| [rad/s^2]", yscale="log")
    axes[0][2].set(title="commanded step |a(t) - a(t-1)|", xlabel="rad", yscale="log")
    axes[1][0].set(title="speed percentiles", xlabel="percentile", ylabel="rad/s")
    axes[1][1].set(title="acceleration percentiles", xlabel="percentile", ylabel="rad/s^2", yscale="log")
    axes[1][2].set(title="command-step percentiles (the chunk switches)", xlabel="percentile", ylabel="rad",
                   yscale="log")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("mass near zero speed = time standing still; acceleration tail = how violently the plan changes",
                 fontsize=9)
    fig.tight_layout()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"\nwrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
