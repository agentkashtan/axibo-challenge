"""Sample N random object layouts, save them to CSV, and save a coverage plot next to it.

    source .venv/bin/activate
    python sample_layouts.py --n 50 --seed 0 --out data/layouts/pilot_50.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from axibo.scene_builder import Workspace, load_layouts_csv, sample_layout, save_layouts_csv
from axibo.sim import OBJECT_NAMES

COLORS = {"red cube": "tab:red", "red cylinder": "darkred", "blue cube": "tab:blue"}
MARKERS = {"red cube": "s", "red cylinder": "o", "blue cube": "s"}


def plot_coverage(layouts, workspace: Workspace, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in OBJECT_NAMES:
        xy = np.array([poses[name].pos[:2] for poses, _ in layouts])
        ax.scatter(xy[:, 1], xy[:, 0], c=COLORS[name], marker=MARKERS[name], label=name, s=25, alpha=0.8)
    (x0, x1), (y0, y1) = workspace.x_range, workspace.y_range
    ax.plot([y0, y1, y1, y0, y0], [x0, x0, x1, x1, x0], "k--", lw=1)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m] (away from robot)")
    ax.invert_xaxis()  # +y on the left, as seen from behind the robot
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    ax.set_title(f"{len(layouts)} layouts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/layouts/pilot_50.csv")
    args = p.parse_args()

    workspace = Workspace()
    rng = np.random.default_rng(args.seed)
    layouts = [sample_layout(rng, workspace) for _ in range(args.n)]

    out = Path(args.out)
    save_layouts_csv(out, layouts, seed=args.seed)
    plot_coverage(layouts, workspace, out.with_suffix(".png"))

    assert len(load_layouts_csv(out)) == args.n
    print(f"saved {args.n} layouts to {out} (+ {out.with_suffix('.png')})")


if __name__ == "__main__":
    main()
