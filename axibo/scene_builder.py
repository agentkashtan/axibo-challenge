"""Random object layouts for the stacking task: sampling, and saving/loading them as CSV.

A layout is one pose per object, all upright on the table inside the workspace, pairwise at least `min_dist` apart.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from axibo.sim import OBJECT_NAMES, ObjectPose, SimConfig, object_height


@dataclass
class Workspace:
    """Region for object centers. Chosen from an IK sweep: gripper-down grasp (z=0.02), stack (z=0.07) and approach
    (z<=0.12) all reachable with joints 1-5 >= 0.1 rad from their limits, and fully visible in top and side cameras."""

    x_range: tuple[float, float] = (0.18, 0.34)
    y_range: tuple[float, float] = (-0.15, 0.15)
    yaw_range_deg: tuple[float, float] = (0.0, 90.0)  # cubes only: a cube looks the same every 90 deg
    cylinder_yaw_deg: float = 0.0  # the cylinder is rotationally symmetric, so its yaw carries no information
    # Between object centers. 0.08 left only a 2.3 cm surface gap in the worst case (two cubes corner to corner,
    # half-diagonal 2.83 cm); 0.10 leaves 4.3 cm, so the fingers descend beside a neighbour with room to spare.
    # Sampling is unaffected (250/250 layouts placed, mean pair distance 15.2 -> 16.2 cm, coverage unchanged).
    # The cost is that tightly packed scenes never appear in training or eval.
    min_dist: float = 0.10


def yaw_quat(deg: float) -> tuple[float, float, float, float]:
    """Rotation about world z as a Genesis (w, x, y, z) quaternion."""
    x, y, z, w = Rotation.from_euler("z", deg, degrees=True).as_quat()
    return (float(w), float(x), float(y), float(z))


def resting_z(name: str, cfg: SimConfig) -> float:
    """Center height of an upright object resting on the table."""
    return object_height(name, cfg) / 2


def sample_layout(
    rng: np.random.Generator, workspace: Workspace = Workspace(), cfg: SimConfig = SimConfig(), max_tries: int = 1000
) -> tuple[dict[str, ObjectPose], dict[str, float]]:
    """Place objects one at a time; re-sample an object until it is >= min_dist from those already placed.

    The placement order is shuffled per layout so no object is systematically placed first (the first object is
    uniform, later ones are pushed away from it — with a fixed order that bias would be tied to object identity).
    Returns (poses, yaw_deg per object).
    """
    placed_xy: list[np.ndarray] = []
    poses, yaws = {}, {}
    for name in rng.permutation(OBJECT_NAMES):
        for _ in range(max_tries):
            xy = np.array([rng.uniform(*workspace.x_range), rng.uniform(*workspace.y_range)])
            if all(np.linalg.norm(xy - other) >= workspace.min_dist for other in placed_xy):
                break
        else:
            raise RuntimeError(f"could not place {name} after {max_tries} tries; workspace too small for min_dist")
        # Only cubes get a random yaw: the cylinder's yaw is unobservable, and randomizing it would make the
        # scripted grasp orientation ambiguous for visually identical scenes.
        yaw = workspace.cylinder_yaw_deg if "cylinder" in name else float(rng.uniform(*workspace.yaw_range_deg))
        placed_xy.append(xy)
        poses[name] = ObjectPose(pos=(float(xy[0]), float(xy[1]), resting_z(name, cfg)), quat=yaw_quat(yaw))
        yaws[name] = yaw
    return {name: poses[name] for name in OBJECT_NAMES}, {name: yaws[name] for name in OBJECT_NAMES}


def _col(name: str, field: str) -> str:
    return f"{name.replace(' ', '_')}_{field}"


def save_layouts_csv(path: str | Path, layouts: list[tuple[dict[str, ObjectPose], dict[str, float]]], seed: int):
    """One row per layout: layout_id, seed, then x, y, z, yaw_deg for each object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["layout_id", "seed"] + [_col(n, f) for n in OBJECT_NAMES for f in ("x", "y", "z", "yaw_deg")]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (poses, yaws) in enumerate(layouts):
            row = {"layout_id": i, "seed": seed}
            for n in OBJECT_NAMES:
                x, y, z = poses[n].pos
                row.update({_col(n, "x"): f"{x:.5f}", _col(n, "y"): f"{y:.5f}", _col(n, "z"): f"{z:.5f}"})
                row[_col(n, "yaw_deg")] = f"{yaws[n]:.3f}"
            writer.writerow(row)


def load_layouts_csv(path: str | Path) -> list[dict[str, ObjectPose]]:
    """Inverse of save_layouts_csv: list index = layout_id."""
    with Path(path).open(newline="") as f:
        return [
            {
                n: ObjectPose(
                    pos=(float(row[_col(n, "x")]), float(row[_col(n, "y")]), float(row[_col(n, "z")])),
                    quat=yaw_quat(float(row[_col(n, "yaw_deg")])),
                )
                for n in OBJECT_NAMES
            }
            for row in csv.DictReader(f)
        ]
