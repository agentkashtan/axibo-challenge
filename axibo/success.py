"""Stacking success check from simulator state. Shared by demo collection (log/filter) and policy evaluation."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from axibo.sim import OBJECT_NAMES, PiperStackingScene, object_height


@dataclass
class StackTolerances:
    """Placeholder tolerances (to be stated in the report). Checked after the arm retreated and the scene settled."""

    max_xy_offset: float = 0.010  # [m] source center vs destination center, horizontal
    max_dz_error: float = 0.005  # [m] source center height above destination vs resting-on-top height
    max_dst_tilt_deg: float = 10.0  # destination still upright
    max_third_displacement: float = 0.010  # [m] third object undisturbed


def tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between an object's local z axis and world z."""
    w, x, y, z = quat_wxyz
    return float(np.degrees(np.arccos(np.clip(Rotation.from_quat([x, y, z, w]).as_matrix()[2, 2], -1.0, 1.0))))


def third_object(source: str, destination: str) -> str:
    return next(n for n in OBJECT_NAMES if n not in (source, destination))


def stack_metrics(
    sim: PiperStackingScene,
    env_idx: int,
    source: str,
    destination: str,
    third_initial_pos: np.ndarray,
    tol: StackTolerances = StackTolerances(),
) -> dict:
    """Metrics (mm / deg) plus a `success` flag for one env."""
    src_pos, _ = (a[env_idx] for a in sim.object_pose(source))
    dst_pos, dst_quat = (a[env_idx] for a in sim.object_pose(destination))
    third_pos, _ = (a[env_idx] for a in sim.object_pose(third_object(source, destination)))

    expected_dz = (object_height(source, sim.cfg) + object_height(destination, sim.cfg)) / 2
    xy = float(np.linalg.norm(src_pos[:2] - dst_pos[:2]))
    dz_err = float(src_pos[2] - dst_pos[2] - expected_dz)
    tilt = tilt_deg(dst_quat)
    third = float(np.linalg.norm(third_pos - third_initial_pos))
    success = (
        xy < tol.max_xy_offset
        and abs(dz_err) < tol.max_dz_error
        and tilt < tol.max_dst_tilt_deg
        and third < tol.max_third_displacement
    )
    return {
        "success": bool(success),
        "xy_offset_mm": round(xy * 1000, 2),
        "dz_error_mm": round(dz_err * 1000, 2),
        "dst_tilt_deg": round(tilt, 2),
        "third_moved_mm": round(third * 1000, 2),
    }
