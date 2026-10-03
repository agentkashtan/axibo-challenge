"""Smoothness metrics for a rollout: what Task 3 improves, measured where it matters.

The point of asynchronous execution is that the arm never waits for the next chunk, so the metrics are split into

    - the stall itself (fraction of steps that re-sent the previous action because the queue was dry),
    - discontinuity *at chunk boundaries*, which is where a stall or a hard switch shows up, and
    - whole-episode jerk, reported alongside so a boundary improvement cannot be claimed while the rest of the
      trajectory gets worse.

Averaging jerk over a whole episode hides the effect entirely: a handful of boundary steps are diluted by hundreds
of smooth ones, which is why every metric below also has a boundary-restricted form.
"""

import numpy as np

ARM = slice(0, 6)  # arm joints only; the gripper moves in steps by design


def _arm(frames, attr: str) -> np.ndarray:
    vals = [getattr(f, attr) for f in frames]
    if any(v is None for v in vals):
        return np.empty((0, 6))
    return np.asarray(vals, dtype=float)[:, ARM]


def episode_metrics(trace, hz: int, boundary_window: int = 2) -> dict:
    """Metrics for one episode. Returns {} when the trace carries no joint data (plain eval runs)."""
    frames = trace.frames
    qpos = _arm(frames, "qpos")
    qvel = _arm(frames, "qvel")
    if len(qpos) < 4:
        return {}

    dt = 1.0 / hz
    switches = [i for i, f in enumerate(frames) if f.switch]
    stalled = np.array([bool(f.stalled) for f in frames])

    # Velocity discontinuity: |dq/dt(t) - dq/dt(t-1)|, max over joints. Uses the measured joint velocity, so it
    # reflects what the arm actually did rather than what was commanded.
    dv = np.abs(np.diff(qvel, axis=0)).max(axis=1) if len(qvel) else np.abs(np.diff(qpos, axis=0) / dt).max(axis=1)
    # Jerk: third difference of position. Units rad/s^3.
    jerk = np.abs(np.diff(qpos, n=3, axis=0) / dt**3).max(axis=1)

    def at_boundaries(series: np.ndarray, offset: int) -> np.ndarray:
        """Values of `series` within +-boundary_window of a switch (`offset` aligns the differenced index)."""
        if not switches or len(series) == 0:
            return np.empty(0)
        idx = set()
        for s in switches:
            for j in range(s - boundary_window, s + boundary_window + 1):
                k = j - offset
                if 0 <= k < len(series):
                    idx.add(k)
        return series[sorted(idx)] if idx else np.empty(0)

    dv_b = at_boundaries(dv, 1)
    jerk_b = at_boundaries(jerk, 3)
    return {
        "stall_fraction": float(stalled.mean()),
        "stalled_steps": int(stalled.sum()),
        "switches": len(switches),
        "vel_disc_mean": float(dv.mean()),
        "vel_disc_max": float(dv.max()),
        "vel_disc_boundary_mean": float(dv_b.mean()) if len(dv_b) else 0.0,
        "vel_disc_boundary_max": float(dv_b.max()) if len(dv_b) else 0.0,
        "jerk_rms": float(np.sqrt((jerk**2).mean())),
        "jerk_boundary_rms": float(np.sqrt((jerk_b**2).mean())) if len(jerk_b) else 0.0,
        "jerk_max": float(jerk.max()),
    }


def trace_arrays(trace, stats) -> dict:
    """Per-step arrays for plotting / npz dumps."""
    frames = trace.frames
    out = {
        "qpos": np.asarray([f.qpos for f in frames if f.qpos is not None], dtype=float),
        "qvel": np.asarray([f.qvel for f in frames if f.qvel is not None], dtype=float),
        "action": np.asarray([f.action for f in frames if f.action is not None], dtype=float),
        "stalled": np.asarray([f.stalled for f in frames], dtype=bool),
        "switch": np.asarray([f.switch for f in frames], dtype=bool),
        "src_pos": np.asarray([f.src_pos for f in frames], dtype=float),
        "dst_pos": np.asarray([f.dst_pos for f in frames], dtype=float),
        "gripper": np.asarray([f.gripper for f in frames], dtype=float),
        # dst_quat / third_pos / src_speed are here so a saved run can be re-scored offline when the outcome
        # taxonomy changes: the rotated footprint test needs the destination's yaw, third_ok needs the third
        # object, and the placement test needs the source's speed at release.
        "dst_quat": np.asarray([f.dst_quat for f in frames], dtype=float),
        "third_pos": np.asarray([f.third_pos for f in frames], dtype=float),
        "src_speed": np.asarray([f.src_speed for f in frames], dtype=float),
    }
    out["step"] = np.arange(len(frames))
    if stats is not None:
        out["latency_s"] = np.asarray(stats.latency_s, dtype=float)
        out["switch_steps"] = np.asarray(stats.switch_steps, dtype=int)
    return out
