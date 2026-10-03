"""Scripted pick-and-place demonstrator: plan a full trajectory with IK (no execution), then execute it separately.

Segment sequence (all at the scene's control rate):
    to_pregrasp (joint quintic) -> open -> descend (vertical) -> close -> grasp_wait -> lift (descend reversed)
    -> to_predrop (joint quintic) -> lower (vertical) -> release -> release_wait -> retreat (lower reversed)

Gripper yaw is a deterministic function of the object poses (see grasp_yaw_deg), never of the arm's configuration.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from axibo.scene_builder import load_layouts_csv
from axibo.sim import (
    ARM_DOFS,
    FINGERTIP_BELOW_PAD,
    GRASP_LOCAL_POINT,
    GRIPPER_MAX,
    OBJECT_NAMES,
    ObjectPose,
    PiperStackingScene,
    object_half_width,
    object_height,
)

# Ordered (source, destination) index pairs into OBJECT_NAMES. The held-out pair is never collected: both objects are
# red, so only shape tells source from destination; its reverse (red cube on red cylinder) stays in training.
HELD_OUT_PAIR = (OBJECT_NAMES.index("red cylinder"), OBJECT_NAMES.index("red cube"))
TRAIN_PAIRS = tuple(
    (s, d) for s in range(len(OBJECT_NAMES)) for d in range(len(OBJECT_NAMES)) if s != d and (s, d) != HELD_OUT_PAIR
)

# World-frame gripper yaw used whenever the object's orientation does not constrain the grasp (the cylinder is
# rotationally symmetric). A constant keeps the recorded wrist angle a function of what the cameras can see.
SYMMETRIC_GRASP_YAW_DEG = -90.0

# Added to a cube's own yaw; the jaws are 180-deg symmetric so the grasp is unchanged, but joint6 moves off its
# limit (see grasp_yaw_deg). The cylinder needs no offset: its angles already sit near -55..-113 deg.
CUBE_GRASP_YAW_OFFSET_DEG = 180.0


class PlanningError(RuntimeError):
    """IK failed or jumped branches; the layout/pair should be skipped."""


@dataclass
class PlannerConfig:
    pregrasp_offset: float = 0.08  # [m] above the grasp point
    predrop_offset: float = 0.05  # [m] above the drop point (8 cm would exceed the wrist-pitch comfort zone)
    fingertip_clearance: float = 0.01  # [m] fingertips above the table at grasp
    drop_clearance: float = 0.003  # [m] source bottom above destination top at release (1 cm looked like a drop)
    # Free-space transfers: nothing is near an obstacle, so they run ~17% faster than the first version to buy back
    # some of the time spent on the delicate segments.
    spline_speed: float = 0.14  # [m/s] Cartesian distance / speed -> joint-spline duration
    spline_min_duration: float = 0.85  # [s]
    # [rad/s] peak. At 1.8 (3.6 rad/s once time_scale halves the durations) the PD arm could not keep up with its
    # own target during to_pregrasp: 0.148 rad mean / 0.265 rad peak tracking error, against <= 0.04 rad on every
    # other segment. The recorded action then leads the achieved state, which is what the policy has to learn from.
    spline_max_joint_speed: float = 1.2
    # The delicate segments run slower than the rest: descend/lift/lower happen next to an object the policy has to
    # hit within a few mm, so they get more frames and smaller per-step deltas. The retreat is over empty space.
    vertical_speed: float = 0.025  # [m/s] descend and lift: 8 cm -> 3.2 s
    lower_duration: float = 3.2  # [s] lower: placing is as delicate as grasping, so it gets the same time as descend
    retreat_speed: float = 0.04  # [m/s] retreat, away from the stack
    ik_step: float = 0.005  # [m] IK waypoint spacing on vertical segments
    gripper_duration: float = 0.5  # [s] opening on the way in, where nothing is held
    grip_duration: float = 0.8  # [s] closing on the object and releasing it
    grasp_wait: float = 0.5  # [s]
    release_wait: float = 0.5  # [s]
    # [m] close target = object half-width - squeeze. At 0.005 the cylinder tipped in the grip during the lift: once
    # it leaves the table it hangs on one contact per pad, and rotation about the line joining them is unconstrained
    # (torsional friction is off). Deeper penetration gives a firmer grip and stopped the tipping.
    squeeze: float = 0.008
    ik_pos_tol: float = 2e-3  # [m]
    ik_rot_tol: float = 2e-2  # [rad]
    max_waypoint_jump: float = 0.2  # [rad] between consecutive vertical IK waypoints
    # Divides every segment duration (so 0.5 runs the whole demo twice as fast) without changing the timing profile
    # between segments. The demos are dominated by frames where the arm barely moves: at 1.0 a 16.4 s episode has
    # 31% of frames under 0.0005 rad of joint motion and 73% under 0.005, which leaves little signal per step, and a
    # policy re-planning every 10 steps re-observes an almost identical state and stalls. At 0.5, over
    # testv2_10 x 5 pairs: still 50/50 successes, xy offset 0.89 -> 1.05 mm mean (max 5.6 -> 4.8), episodes 16.9 ->
    # 8.5 s, collection wall time halved. The cost is PD tracking lag, 0.138 -> 0.272 rad on the fast free-space
    # segments: the recorded action leads the achieved state by more than it used to.
    time_scale: float = 0.5


@dataclass
class Segment:
    name: str
    q: np.ndarray  # (T, 6) arm joint targets
    gripper: np.ndarray  # (T,) gripper opening targets


@dataclass
class DemoPlan:
    layout_id: int
    source: str
    destination: str
    task: str
    object_poses: dict[str, ObjectPose]
    grasp_yaw_deg: float
    drop_yaw_deg: float
    control_hz: int
    segments: list[Segment] = field(default_factory=list)
    max_ik_pos_error: float = 0.0
    delta_xy: np.ndarray | None = None  # grasp offset compensated for; set only by plan_place_from_here

    def actions(self) -> np.ndarray:
        """(T, 7) = [6 arm joint targets, gripper opening], one row per control step."""
        return np.concatenate([np.column_stack([s.q, s.gripper]) for s in self.segments]).astype(np.float32)

    def summary(self) -> str:
        lines = [f"{self.task!r} | layout {self.layout_id} | grasp yaw {self.grasp_yaw_deg:.0f} deg, "
                 f"drop yaw {self.drop_yaw_deg:.0f} deg | max IK pos error {self.max_ik_pos_error * 1000:.2f} mm"]
        lines += [f"  {s.name:13s} {len(s.q):4d} steps  {len(s.q) / self.control_hz:5.2f} s" for s in self.segments]
        total = sum(len(s.q) for s in self.segments)
        lines.append(f"  {'total':13s} {total:4d} steps  {total / self.control_hz:5.2f} s")
        return "\n".join(lines)


def quintic(s: np.ndarray) -> np.ndarray:
    """Minimum-jerk progress 0 -> 1 with zero velocity and acceleration at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def gripper_down_quat(yaw_deg: float) -> np.ndarray:
    """Link6 z pointing down, rotated by yaw about world z. Genesis (w, x, y, z)."""
    x, y, z, w = (Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler("x", np.pi)).as_quat()
    return np.array([w, x, y, z], dtype=np.float32)


def object_yaw_deg(pose: ObjectPose) -> float:
    w, x, y, z = pose.quat
    return float(Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)[2])


def grasp_yaw_deg(name: str, pose: ObjectPose) -> float:
    """Gripper yaw for grasping `name`: the cube's own yaw (layouts sample it in [0, 90), which covers every
    appearance of a 90-degree-symmetric cube), a fixed world yaw for the rotationally symmetric cylinder.

    Deliberately single-valued: picking among the equivalent candidates by arm configuration, as an earlier version
    did, made the recorded wrist angle depend on something the policy cannot observe.

    The cube's yaw carries a +180 deg offset. The jaws are symmetric, so yaw and yaw+180 are the same grasp, but
    without the offset the resulting joint6 lands against its own limit (+-179.9 deg): measured over
    testv2_10 x 5 pairs, 13% of the grasp/drop angles sat within 10 deg of that wall. There, one degree of cube
    rotation flips joint6 by ~359 deg (layout 8: cube yaw 72 -> -179.4, yaw 73 -> +179.6) and turns a 28 deg wrist
    motion into 331 deg, because joint6 cannot wrap. Visually identical scenes then carry opposite targets, which
    is not learnable. With the offset the cube angles move to roughly [0, 70] deg, clear of both limits.
    """
    return object_yaw_deg(pose) + CUBE_GRASP_YAW_OFFSET_DEG if "cube" in name else SYMMETRIC_GRASP_YAW_DEG


def _progress_steps(duration: float, hz: int) -> np.ndarray:
    n = max(1, math.ceil(duration * hz))
    return quintic(np.arange(1, n + 1) / n)  # excludes the start point, ends exactly at 1


class _Planner:
    def __init__(self, sim: PiperStackingScene, cfg: PlannerConfig, env_idx: int):
        self.sim, self.cfg, self.hz, self.env_idx = sim, cfg, sim.cfg.control_hz, env_idx
        self.link = sim.robot.get_link("Link6")
        self.local_point = np.array(GRASP_LOCAL_POINT, dtype=np.float32)
        self.max_pos_err = 0.0

    def ik(self, pos, yaw_deg: float, seed6: np.ndarray, where: str) -> np.ndarray:
        seed = np.zeros((1, self.sim.robot.n_qs), dtype=np.float32)
        seed[0, ARM_DOFS] = seed6
        q, err = self.sim.robot.inverse_kinematics(
            link=self.link, pos=np.asarray(pos, dtype=np.float32)[None], quat=gripper_down_quat(yaw_deg)[None],
            local_point=self.local_point, init_qpos=seed, dofs_idx_local=ARM_DOFS, return_error=True,
            envs_idx=[self.env_idx],
        )
        q, err = q.cpu().numpy().reshape(-1), err.cpu().numpy().reshape(-1)
        pos_err, rot_err = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
        if pos_err > self.cfg.ik_pos_tol or rot_err > self.cfg.ik_rot_tol:
            raise PlanningError(f"{where}: IK error {pos_err * 1000:.1f} mm / {rot_err:.3f} rad at {np.round(pos, 3)}")
        self.max_pos_err = max(self.max_pos_err, pos_err)
        return q[ARM_DOFS]

    def joint_spline(self, name, q0, q1, cart_dist, gripper) -> Segment:
        # Quintic peak velocity = 1.875 * delta / duration, so the joint-speed cap sets a second lower bound.
        joint_bound = 1.875 * float(np.abs(q1 - q0).max()) / self.cfg.spline_max_joint_speed
        duration = max(cart_dist / self.cfg.spline_speed, self.cfg.spline_min_duration, joint_bound)
        s = self._steps(duration)[:, None]
        return Segment(name, q0 + (q1 - q0) * s, np.full(len(s), gripper))

    def vertical_waypoints(self, start_pos, dz, yaw, q_start, name) -> np.ndarray:
        """(n+1, 6) IK solutions every ik_step along a straight vertical line, each seeded with the previous one."""
        n = max(1, math.ceil(abs(dz) / self.cfg.ik_step))
        qs = [q_start]
        for i in range(1, n + 1):
            pos = np.asarray(start_pos) + np.array([0.0, 0.0, dz * i / n])
            q = self.ik(pos, yaw, qs[-1], f"{name} waypoint {i}/{n}")
            jump = float(np.abs(q - qs[-1]).max())
            if jump > self.cfg.max_waypoint_jump:
                raise PlanningError(f"{name}: joint jump {jump:.2f} rad at waypoint {i}/{n} (IK branch flip)")
            qs.append(q)
        return np.array(qs)

    def timed_path(self, name, waypoints: np.ndarray, length: float, gripper: float, speed: float | None = None) -> Segment:
        """Traverse the waypoint list with quintic progress at `speed`, interpolating between waypoints."""
        s = self._steps(length / (speed or self.cfg.vertical_speed)) * (len(waypoints) - 1)
        i0 = np.minimum(np.floor(s).astype(int), len(waypoints) - 2)
        frac = (s - i0)[:, None]
        q = waypoints[i0] + (waypoints[i0 + 1] - waypoints[i0]) * frac
        return Segment(name, q, np.full(len(q), gripper))

    def gripper_move(self, name, q, g0, g1, duration: float | None = None) -> Segment:
        s = self._steps(duration or self.cfg.gripper_duration)
        return Segment(name, np.tile(q, (len(s), 1)), g0 + (g1 - g0) * s)

    def _steps(self, duration: float) -> np.ndarray:
        return _progress_steps(duration * self.cfg.time_scale, self.hz)

    def hold(self, name, q, gripper, duration) -> Segment:
        n = max(1, math.ceil(duration * self.cfg.time_scale * self.hz))
        return Segment(name, np.tile(q, (n, 1)), np.full(n, gripper))


def plan_demo(
    sim: PiperStackingScene,
    layout_id: int,
    pair_ids: tuple[int, int],
    layouts_csv: str,
    cfg: PlannerConfig = PlannerConfig(),
    env_idx: int = 0,
) -> DemoPlan:
    """Plan a full pick-and-place from env `env_idx`'s current state. Nothing is executed; the robot is not moved.

    pair_ids = (source index, destination index) into OBJECT_NAMES. That env should already be reset to this layout.
    """
    src, dst = OBJECT_NAMES[pair_ids[0]], OBJECT_NAMES[pair_ids[1]]
    if src == dst:
        raise ValueError("source and destination must differ")
    poses = load_layouts_csv(layouts_csv)[layout_id]
    p = _Planner(sim, cfg, env_idx)
    sim_cfg = sim.cfg

    qpos = sim.qpos()[env_idx]
    q_start = qpos[ARM_DOFS]
    g_start = float(np.clip((qpos[6] - qpos[7]) / 2, 0.0, GRIPPER_MAX))  # measured value can be slightly negative

    # Key positions of the grasp point (finger-pad center).
    pad_z = cfg.fingertip_clearance + FINGERTIP_BELOW_PAD  # objects rest on the table: bottom z = 0
    grasp = np.array([*poses[src].pos[:2], pad_z])
    pregrasp = grasp + [0.0, 0.0, cfg.pregrasp_offset]
    drop = np.array([*poses[dst].pos[:2], object_height(dst, sim_cfg) + cfg.drop_clearance + pad_z])
    predrop = drop + [0.0, 0.0, cfg.predrop_offset]

    # Yaws. Both are functions of the object poses alone (see grasp_yaw_deg), so visually identical scenes get
    # identical wrist angles. The wrist only turns between grasp and drop when the placed cube has to line up
    # with the cube below it; for a cylinder at either end the orientation does not matter, so it is kept.
    grasp_yaw = grasp_yaw_deg(src, poses[src])
    drop_yaw = grasp_yaw_deg(dst, poses[dst]) if "cube" in src and "cube" in dst else grasp_yaw

    g_open = GRIPPER_MAX
    g_closed = max(0.0, object_half_width(src, sim_cfg) - cfg.squeeze)

    q_pregrasp = p.ik(pregrasp, grasp_yaw, q_start, "pregrasp")
    descend = p.vertical_waypoints(pregrasp, -cfg.pregrasp_offset, grasp_yaw, q_pregrasp, "descend")
    q_grasp = descend[-1]

    q_predrop = p.ik(predrop, drop_yaw, q_pregrasp, "predrop")
    lower = p.vertical_waypoints(predrop, -cfg.predrop_offset, drop_yaw, q_predrop, "lower")
    q_drop = lower[-1]

    plan = DemoPlan(
        layout_id=layout_id, source=src, destination=dst, task=f"put the {src} on the {dst}", object_poses=poses,
        grasp_yaw_deg=grasp_yaw % 360, drop_yaw_deg=drop_yaw % 360, control_hz=p.hz,
    )
    plan.segments = [
        p.joint_spline("to_pregrasp", q_start, q_pregrasp, np.linalg.norm(pregrasp - sim.grasp_point_world(env_idx)), g_start),
        # Skipped when the home pose already has the gripper open: the segment would only hold the arm still.
        *([] if abs(g_start - g_open) < 1e-4 else [p.gripper_move("open", q_pregrasp, g_start, g_open)]),
        p.timed_path("descend", descend, cfg.pregrasp_offset, g_open),
        p.gripper_move("close", q_grasp, g_open, g_closed, cfg.grip_duration),
        p.hold("grasp_wait", q_grasp, g_closed, cfg.grasp_wait),
        p.timed_path("lift", descend[::-1], cfg.pregrasp_offset, g_closed),
        p.joint_spline("to_predrop", q_pregrasp, q_predrop, np.linalg.norm(predrop - pregrasp), g_closed),
        p.timed_path("lower", lower, cfg.predrop_offset, g_closed, cfg.predrop_offset / cfg.lower_duration),
        p.gripper_move("release", q_drop, g_closed, g_open, cfg.grip_duration),
        p.hold("release_wait", q_drop, g_open, cfg.release_wait),
        p.timed_path("retreat", lower[::-1], cfg.predrop_offset, g_open, cfg.retreat_speed),
    ]
    plan.max_ik_pos_error = p.max_pos_err
    return plan


def execute_plan(sim: PiperStackingScene, plan: DemoPlan, on_step=None) -> int:
    """Run the plan open-loop in a single-env scene. Returns the number of clipped actions.

    on_step(step, action) is called after each control step. Batched collection lives in collect_demos.py.
    """
    if sim.n_envs != 1:
        raise ValueError("execute_plan runs a single-env scene; use collect_demos.py for batched execution")
    if plan.control_hz != sim.cfg.control_hz:
        raise ValueError(f"plan made for {plan.control_hz} Hz, scene runs at {sim.cfg.control_hz} Hz")
    n_clipped = 0
    for step, action in enumerate(plan.actions()):
        n_clipped += int(sim.apply_action(action[None])[0])
        sim.step_control()
        if on_step is not None:
            on_step(step, action)
    return n_clipped


def current_wrist_yaw_deg(sim: PiperStackingScene, env_idx: int = 0) -> float:
    """The yaw of Link6's current orientation, in the same "pointing down, rotated by yaw" family the planner uses.

    gripper_down_quat builds Rz(yaw) @ Rx(pi) = [[c, s, 0], [s, -c, 0], [0, 0, -1]], so yaw = atan2(R[1,0], R[0,0]).
    Any pitch/roll the policy left the wrist in is projected away, which is what we want: the correction re-aims
    straight down like the demos do.
    """
    w, x, y, z = sim.robot.get_link("Link6").get_quat().cpu().numpy().reshape(-1, 4)[env_idx]
    r = Rotation.from_quat([x, y, z, w]).as_matrix()
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def plan_place_from_here(
    sim: PiperStackingScene,
    source: str,
    destination: str,
    cfg: PlannerConfig = PlannerConfig(),
    env_idx: int = 0,
    layout_id: int = -1,
) -> DemoPlan:
    """Place an *already held* source onto the destination, from wherever the arm currently is.

    The place half of plan_demo - `to_predrop -> lower -> release -> release_wait -> retreat` - used to take over
    from the policy mid-episode once it has carried the object to the destination.

    The point of it is the grasp-offset compensation. The arm commands the TCP (finger-pad centre), and while the
    object is held `src_xy = tcp_xy + delta`, so aiming the TCP at the destination lands the object at
    `dst_xy + delta`. The policy's grasp is off-centre by 20-37 mm, which is exactly the misplacement measured at
    its releases, so the TCP is aimed at `dst_xy - delta` instead and the *object* ends over the destination.
    delta is read from the live sim, so it needs no logged field and holds whatever the grasp actually was.

    Only `drop` is compensated: `predrop` is `drop` plus a pure z offset and `lower` descends vertically from it,
    so the corrected xy carries through to the release. The current wrist yaw is kept rather than recomputed -
    the policy's grasp yaw is unknown and the object may be rotated in the jaws, while success checks only the
    source's centre, so re-deriving it would reintroduce the joint6 multimodality that grasp_yaw_deg fixed.
    """
    if sim.n_envs != 1 and env_idx >= sim.n_envs:
        raise ValueError(f"env_idx {env_idx} out of range for {sim.n_envs} envs")
    qpos = sim.qpos()[env_idx]
    q_start = qpos[ARM_DOFS]
    sim_cfg = sim.cfg
    p = _Planner(sim, cfg, env_idx)

    src_pos = sim.object_pose(source)[0][env_idx]
    dst_pos, dst_quat = (a[env_idx] for a in sim.object_pose(destination))
    tcp = sim.grasp_point_world(env_idx)
    delta_xy = src_pos[:2] - tcp[:2]  # the object's offset within the jaws; constant while held

    pad_z = cfg.fingertip_clearance + FINGERTIP_BELOW_PAD
    drop = np.array([*(dst_pos[:2] - delta_xy), object_height(destination, sim_cfg) + cfg.drop_clearance + pad_z])
    predrop = drop + [0.0, 0.0, cfg.predrop_offset]
    yaw = current_wrist_yaw_deg(sim, env_idx)

    # Re-command the demos' closed target rather than the measured opening: the fingers rest ~3 mm wider than the
    # command while squeezing an object, so echoing the measurement back would slacken the grip during the move.
    g_closed = max(0.0, object_half_width(source, sim_cfg) - cfg.squeeze)
    g_open = GRIPPER_MAX

    q_predrop = p.ik(predrop, yaw, q_start, "correction predrop")
    lower = p.vertical_waypoints(predrop, -cfg.predrop_offset, yaw, q_predrop, "correction lower")
    q_drop = lower[-1]

    plan = DemoPlan(
        layout_id=layout_id, source=source, destination=destination,
        task=f"put the {source} on the {destination}",
        object_poses={n: ObjectPose(pos=tuple(sim.object_pose(n)[0][env_idx]), quat=tuple(sim.object_pose(n)[1][env_idx]))
                      for n in OBJECT_NAMES},
        grasp_yaw_deg=yaw % 360, drop_yaw_deg=yaw % 360, control_hz=p.hz,
    )
    plan.segments = [
        p.joint_spline("to_predrop", q_start, q_predrop, float(np.linalg.norm(predrop - tcp)), g_closed),
        p.timed_path("lower", lower, cfg.predrop_offset, g_closed, cfg.predrop_offset / cfg.lower_duration),
        p.gripper_move("release", q_drop, g_closed, g_open, cfg.grip_duration),
        p.hold("release_wait", q_drop, g_open, cfg.release_wait),
        p.timed_path("retreat", lower[::-1], cfg.predrop_offset, g_open, cfg.retreat_speed),
    ]
    plan.max_ik_pos_error = p.max_pos_err
    plan.delta_xy = delta_xy
    return plan
