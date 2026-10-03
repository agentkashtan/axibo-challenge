"""Rollout outcome detection and failure taxonomy for policy evaluation.

`axibo/success.py` answers "is this final state a good stack?" for scripted demos, whose episodes end at a known
step. A policy rollout has no such end: it may stack and keep moving, stall, or never grasp. So evaluation needs

    1. a completion detector that says when a stack exists (which starts the settle window), and
    2. rules that turn the recorded trajectory into one label explaining *how* an attempt went.

Everything is computed from object poses and the gripper opening, sampled once per control step. Heights are
absolute (table at z = 0), never relative to the destination's current center, so a stack built on a destination
that has been knocked over or shoved cannot pass as success.

Conditions, all evaluated per step (see OutcomeTolerances for the constants):

    C1  over_dst        source center inside the destination footprint (square in the cube's yaw frame, disc
                        for the cylinder) - the success test
    C1a overlaps_dst    C1 with the footprint grown by transport_margin - "the object reached the destination",
                        for the TRANSPORTED milestone, since C1 ignores the source's own width
    C1b stably_on_dst   C1 with the footprint shrunk by stable_frac - "this placement was sound", for the
                        knocked_off / misplaced split, since a centre on the footprint edge has almost no
                        support beneath it
    C2  at_stack_height |z_src - (h_dst + h_src / 2)| <= dz_tol
    C3  above_dst       z_src >= h_dst + h_src / 2 + hover
    C4  lift_height     z_src >= resting + h_src
    C5  settled         source moved <= settle_disp over the last settle_window_steps (net displacement, not
                        instantaneous speed: a stack can vibrate in contact while going nowhere)
    C6  open            gripper opening >= release_open
    C7  dst_upright     destination tilt <= max_dst_tilt_deg
    C8  third_ok        third object moved <= max_third_displacement
    C9  on_table        z_src <= resting + on_table_tol
    C10 dst_unmoved     destination moved <= max_dst_displacement
    C11 near_stack_ht   |z_src - (h_dst + h_src / 2)| <= tol_place  (looser than C2: "resting on it", not
                        "within success tolerance")
    C12 set_down        C11 at the frame the jaws let go: the source was resting on the destination rather than
                        being released above it

Milestones latch the first step at which they hold: LIFTED = C4, TRANSPORTED = C1a & C3, STACKED = C1 & C2 & C5
& C6 sustained for detector_hold_steps. RELEASED is the frame the jaws let go of a lifted object, and PLACED is
the frame before it when C12 holds there.

The placement event is what separates the failure modes. A rollout that reaches the destination can fail three
distinct ways, and the old taxonomy called all of them `misplaced`. Given the episode did not end in a stack,
the release frame answers them in two tests:

    arrival  TRANSPORTED never latched   ->  dropped_in_transport (lost before the placement stage)
    release  the FIRST jaw opening after TRANSPORTED is the placement attempt; later openings are recovery
    height   |z_src - stack_z| > tol_place  ->  released_in_air     (arrived, then let go above it)
    footprint C1b (the shrunk footprint)  ->  knocked_off, else misplaced

So `knocked_off` means "set down inside the footprint and not there at the end" and `misplaced` means "set down
outside it". The first is an inference, not a measurement - the destination's own tilt/displacement is checked
first so a shifting destination is not blamed on the arm, but a tilt at release is invisible to an xy-centre
test, and the footprint test compares the source's *centre* to the destination's face, so a placement with the
centre on the edge (~half the cube overhanging) still counts as inside.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from axibo.sim import PiperStackingScene, object_half_width, object_height
from axibo.success import third_object, tilt_deg

# Terminal labels, in the order they are reported.
OUTCOMES = (
    "success",
    "stacked_dst_moved",
    "stacked_third_moved",
    "knocked_off",  # set down inside the destination footprint, but not stacked at the end
    "misplaced",  # set down outside the destination footprint: it was never going to hold
    "released_in_air",  # arrived over the destination, then let go above it instead of onto it
    "dropped_in_transport",
    "no_grasp",
    "timeout_carrying",
    "timeout",
)


@dataclass
class OutcomeTolerances:
    """Thresholds for the per-step conditions. Every reported number depends on these, so they go in the report.

    Margins against the scripted demos the policy imitates: they lift an object exactly 80 mm above its resting
    height and carry it 93-103 mm up, so they clear C4 by 30-40 mm and C3 by 33 mm. The largest artifact that
    could fake a lift is a cube tipped onto a corner (+14.6 mm), a third of the smallest C4 margin; a tipped
    cylinder's center drops, so it cannot fake one at all.
    """

    hover: float = 0.020  # [m] above stack height; the demos release only 3 mm above it, so C3 means "carried
    # into position", not "let go"
    dz_tol: float = 0.005  # [m]
    # "At rest" is net displacement over a window, not instantaneous speed. A cube resting on the cylinder was seen
    # jittering at 32 mm/s in contact while staying put, which an instantaneous test reads as "still moving" for the
    # whole episode. Over 0.5 s the deliberate motions are far larger: the demos lower at ~31 mm/s (16 mm per
    # window) and a free fall is larger still, so 3 mm separates "vibrating in place" from "going somewhere" by ~5x.
    settle_window_steps: int = 15  # 0.5 s at 30 Hz
    settle_disp: float = 0.003  # [m]
    release_open: float = 0.030  # [m]
    on_table_tol: float = 0.005  # [m]
    max_dst_tilt_deg: float = 10.0
    max_dst_displacement: float = 0.010  # [m]; separate from tilt, which misses a destination shoved flat
    max_third_displacement: float = 0.010  # [m]
    # A single step can fire on an object passing through stack height mid-bounce, which would start the settle
    # window early; requiring a few consecutive steps costs 0.1 s at 30 Hz.
    detector_hold_steps: int = 3
    # --- the placement event (C11, C12) ---
    # C11/C12: how close to stack height counts as "resting on the destination" rather than "let go above it".
    # Deliberately looser than dz_tol: the scripted demos release 3 mm above stack height, and 10 mm admits a
    # slightly early release as a placement attempt while classing a 2 cm drop as a drop.
    tol_place: float = 0.010  # [m]
    # Midpoint of the measured gripper range (open 50 mm, closed on a cube ~12 mm). Used only to recognise that
    # the jaws were closed around something, so it needs no precision.
    closed_grip: float = 0.025  # [m]
    # --- the two footprint tests (C1a, C1b) ---
    # ARRIVAL. The TRANSPORTED milestone asks "did the object reach the destination", so it tests the source's
    # centre against the destination footprint grown by this much. The bare footprint (C1) ignores the source's
    # own width and is far too strict for it: over test_cl x 50 trials, six failures came within 25-34 mm of the
    # destination and were labelled dropped_in_transport - "never got there" - when the arm had plainly brought
    # the object over and placed it badly. 20 mm also happens to equal object_half_width for both objects here
    # (cube edge 40, cylinder diameter 40), so this is equivalently "grown by the source's half-width".
    transport_margin: float = 0.020  # [m]
    # STABILITY. The knocked_off / misplaced split asks "was this placement sound", so it tests the centre
    # against the footprint *shrunk* by this factor. The bare footprint is useless for it: a cube centre 19.7 mm
    # from a 20 mm-radius cylinder axis is 0.3 mm inside the support edge with almost no area beneath its centre
    # of mass, and two such placements (19.7 and 19.8 mm) fell on their own yet were labelled knocked_off, i.e.
    # blamed on the arm. A strict geometric guarantee is not available - full containment needs
    # w_dst - w_src = 0 mm for equal-sized objects, i.e. perfect centring - so this is a chosen fraction, not a
    # derived bound: the inner half of the footprint, where the support area under the centre is substantial.
    stable_frac: float = 0.5


@dataclass
class Frame:
    """One control step of a rollout: scalars only, no images.

    The fields below `gripper` are filled in by the queue-based rollout (axibo/rollout.py) and are what the Task 3
    smoothness metrics read; the outcome classifier ignores them, so the plain eval harness can leave them None.
    """

    src_pos: np.ndarray
    src_speed: float
    dst_pos: np.ndarray
    dst_quat: np.ndarray
    third_pos: np.ndarray
    gripper: float
    qpos: np.ndarray | None = None  # (8,) joint positions
    qvel: np.ndarray | None = None  # (8,) joint velocities
    action: np.ndarray | None = None  # (7,) commanded target
    stalled: bool = False  # the queue was dry: the previous action was re-sent
    switch: bool = False  # first step executed from a newly arrived chunk


@dataclass
class Trace:
    """Per-step record of one rollout plus the geometry its conditions need."""

    source: str
    destination: str
    third: str
    h_src: float
    h_dst: float
    w_dst: float
    dst_is_cube: bool
    dst_pos0: np.ndarray
    third_pos0: np.ndarray
    tol: OutcomeTolerances = field(default_factory=OutcomeTolerances)
    frames: list[Frame] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        sim: PiperStackingScene,
        env_idx: int,
        source: str,
        destination: str,
        tol: OutcomeTolerances | None = None,
    ) -> "Trace":
        third = third_object(source, destination)
        return cls(
            source=source,
            destination=destination,
            third=third,
            h_src=object_height(source, sim.cfg),
            h_dst=object_height(destination, sim.cfg),
            w_dst=object_half_width(destination, sim.cfg),
            dst_is_cube="cube" in destination,
            dst_pos0=sim.object_pose(destination)[0][env_idx].copy(),
            third_pos0=sim.object_pose(third)[0][env_idx].copy(),
            tol=tol or OutcomeTolerances(),
        )

    def record(
        self,
        sim: PiperStackingScene,
        env_idx: int,
        gripper: float,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        action: np.ndarray | None = None,
        stalled: bool = False,
        switch: bool = False,
    ) -> Frame:
        dst_pos, dst_quat = (a[env_idx] for a in sim.object_pose(self.destination))
        frame = Frame(
            src_pos=sim.object_pose(self.source)[0][env_idx].copy(),
            src_speed=float(np.linalg.norm(sim.object_vel(self.source)[env_idx])),
            dst_pos=dst_pos.copy(),
            dst_quat=dst_quat.copy(),
            third_pos=sim.object_pose(self.third)[0][env_idx].copy(),
            gripper=float(gripper),
            qpos=None if qpos is None else np.asarray(qpos).copy(),
            qvel=None if qvel is None else np.asarray(qvel).copy(),
            action=None if action is None else np.asarray(action).copy(),
            stalled=bool(stalled),
            switch=bool(switch),
        )
        self.frames.append(frame)
        return frame

    # --- heights (absolute, table at z = 0) ---

    @property
    def stack_z(self) -> float:
        """Source center when resting on the destination: destination top + half the source."""
        return self.h_dst + self.h_src / 2

    @property
    def lift_z(self) -> float:
        """Source center height only a real lift reaches: resting height + one object height."""
        return self.h_src / 2 + self.h_src

    @property
    def rest_z(self) -> float:
        return self.h_src / 2

    # --- per-step conditions ---

    def over_dst(self, f: Frame) -> bool:  # C1 - the success test; centre inside the bare footprint
        return self._in_footprint(f, self.w_dst)

    def at_stack_height(self, f: Frame) -> bool:  # C2
        return bool(abs(f.src_pos[2] - self.stack_z) <= self.tol.dz_tol)

    def above_dst(self, f: Frame) -> bool:  # C3
        return bool(f.src_pos[2] >= self.stack_z + self.tol.hover)

    def lifted(self, f: Frame) -> bool:  # C4
        return bool(f.src_pos[2] >= self.lift_z)

    def settled(self, i: int) -> bool:  # C5
        """Net displacement of the source over the last window; False until the window is full."""
        w = self.tol.settle_window_steps
        if i < w:
            return False
        return bool(np.linalg.norm(self.frames[i].src_pos - self.frames[i - w].src_pos) <= self.tol.settle_disp)

    def released(self, f: Frame) -> bool:  # C6
        return bool(f.gripper >= self.tol.release_open)

    def dst_upright(self, f: Frame) -> bool:  # C7
        return bool(tilt_deg(f.dst_quat) <= self.tol.max_dst_tilt_deg)

    def third_ok(self, f: Frame) -> bool:  # C8
        return bool(np.linalg.norm(f.third_pos - self.third_pos0) <= self.tol.max_third_displacement)

    def on_table(self, f: Frame) -> bool:  # C9
        return bool(f.src_pos[2] <= self.rest_z + self.tol.on_table_tol)

    def dst_unmoved(self, f: Frame) -> bool:  # C10
        return bool(np.linalg.norm(f.dst_pos - self.dst_pos0) <= self.tol.max_dst_displacement)

    def _in_footprint(self, f: Frame, half_width: float) -> bool:
        """Is the source's centre inside the destination footprint at this half-width?

        Square in the destination cube's own yaw frame, disc for the rotationally symmetric cylinder. Shared by
        the three footprint tests so they cannot drift apart: C1 uses the bare half-width, C1a grows it by
        transport_margin, C1b shrinks it by stable_frac.
        """
        d = np.asarray(f.src_pos[:2] - f.dst_pos[:2], dtype=float)
        if not self.dst_is_cube:
            return bool(np.linalg.norm(d) <= half_width)
        w, x, y, z = f.dst_quat
        yaw = float(Rotation.from_quat([x, y, z, w]).as_euler("xyz")[2])
        c, s = np.cos(-yaw), np.sin(-yaw)
        local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        return bool(np.all(np.abs(local) <= half_width))

    def overlaps_dst(self, f: Frame) -> bool:  # C1a - arrival, for the TRANSPORTED milestone
        return self._in_footprint(f, self.w_dst + self.tol.transport_margin)

    def stably_on_dst(self, f: Frame) -> bool:  # C1b - stability, for knocked_off vs misplaced
        return self._in_footprint(f, self.w_dst * self.tol.stable_frac)

    def near_stack_height(self, f: Frame) -> bool:  # C11
        """Resting on the destination, within the looser placement tolerance."""
        return bool(abs(f.src_pos[2] - self.stack_z) <= self.tol.tol_place)

    def set_down(self, i: int) -> bool:  # C12
        """The source was resting on the destination when the jaws let go, rather than let go above it.

        Height only. An earlier version also required the source to be nearly stationary, but the policy
        releases while still moving (1-114 mm/s over testv2_10 x 5 pairs, all within 3 mm of stack height) and
        that cut called two of them "released in air" - one of which succeeded. Whether the placement survived
        is judged at the end of the settle window anyway, so the speed term only added a second threshold to
        justify.
        """
        return bool(self.near_stack_height(self.frames[i]))

    def opened_at(self, i: int) -> bool:
        """Did the jaws cross from closed to open at frame i?

        A crossing, not "is open": `home_qpos` starts the gripper open, so `released()` is already true at
        frame 0. Read from the *measured* opening, not the commanded one - the fingers lag the command by a few
        steps under the position controller, and the frame that matters is the one where they physically let go.
        """
        return bool(i >= 1
                    and self.frames[i].gripper >= self.tol.release_open
                    and self.frames[i - 1].gripper < self.tol.release_open)

    def release_after(self, after: int) -> int | None:
        """The **first** jaw opening after frame `after`, or None.

        Anchored to the TRANSPORTED milestone by the caller, so this is the placement attempt: the moment the
        policy decided to let go at the destination. Openings later in the episode are recovery attempts on an
        object that is already down, and describing the episode by the last one would label the retry rather
        than the failure. (Measured on test_cl trial 46: the placement released at step 246 with the
        destination untouched, then the policy scrabbled at the fallen cube from 313 and shoved the destination
        38 mm before opening again at 364. Taking the last opening blames a target the policy itself moved.)
        """
        return next((i for i in range(after + 1, len(self.frames)) if self.opened_at(i)), None)

    def release_detected_now(self) -> bool:
        """Online counterpart: did the jaws just open? The live loops gate this on TRANSPORTED themselves, so
        the two see the same event as `release_after` does offline."""
        return self.opened_at(len(self.frames) - 1)

    def arrived(self, f: Frame) -> bool:
        """The TRANSPORTED milestone's condition: the object reached the destination, still held and clear of it.

        One definition shared by `analyse` and the live loops (via `arrived_now`). They used to test it
        separately and drifted apart when the footprint was loosened: the offline milestone latched while the
        online one did not, so an episode was labelled a placement failure yet never started its settle window
        and burned the full step budget.
        """
        return self.overlaps_dst(f) and self.above_dst(f)

    def arrived_now(self) -> bool:
        return self.arrived(self.frames[-1])

    def stack_detected(self, i: int) -> bool:
        """Completion conditions at frame i; the detector also requires them to hold (see analyse)."""
        f = self.frames[i]
        return self.over_dst(f) and self.at_stack_height(f) and self.settled(i) and self.released(f)

    def stack_detected_now(self) -> bool:
        """Convenience for a live rollout: evaluate the completion conditions at the most recent frame."""
        return self.stack_detected(len(self.frames) - 1)


@dataclass
class Milestones:
    lifted_step: int | None = None
    transported_step: int | None = None
    stacked_step: int | None = None  # kept unchanged: RECAP's reward uses it as the latch where -1/step stops
    release_step: int | None = None
    placed_step: int | None = None  # release_step - 1, when the source was set down there
    third_disturbed: bool = False  # any step with not C8
    dst_toppled: bool = False  # any step with not C7

    @property
    def lifted(self) -> bool:
        return self.lifted_step is not None

    @property
    def transported(self) -> bool:
        return self.transported_step is not None

    @property
    def stacked(self) -> bool:
        return self.stacked_step is not None

    @property
    def released(self) -> bool:
        return self.release_step is not None

    @property
    def placed(self) -> bool:
        return self.placed_step is not None


@dataclass
class Assessment:
    outcome: str
    milestones: Milestones
    metrics: dict


def analyse(trace: Trace) -> Assessment:
    """Label a finished rollout. The outcome is the last milestone reached, subdivided at the end."""
    tol = trace.tol
    ms = Milestones()
    run = 0
    for i, f in enumerate(trace.frames):
        if ms.lifted_step is None and trace.lifted(f):
            ms.lifted_step = i
        if ms.transported_step is None and trace.arrived(f):
            ms.transported_step = i
        run = run + 1 if trace.stack_detected(i) else 0
        if ms.stacked_step is None and run >= tol.detector_hold_steps:
            ms.stacked_step = i - tol.detector_hold_steps + 1
        ms.third_disturbed |= not trace.third_ok(f)
        ms.dst_toppled |= not trace.dst_upright(f)

    if ms.transported:
        ms.release_step = trace.release_after(ms.transported_step)
        if ms.release_step is not None and trace.set_down(ms.release_step - 1):
            ms.placed_step = ms.release_step - 1

    last_i = len(trace.frames) - 1
    last = trace.frames[last_i]
    holds = trace.stack_detected(last_i)
    dst_ok = trace.dst_upright(last) and trace.dst_unmoved(last)
    third_ok = trace.third_ok(last)

    # The success branch is evaluated first and is unchanged (C1 & C2 & C5 & C6 at the final frame), so
    # refining the failure labels below cannot move the success rate. Everything after it attributes *how* the
    # attempt failed, keyed on the placement event rather than on the stack detector - which never fires on a
    # failed placement and so could not distinguish one failure from another.
    if not ms.lifted:
        outcome = "no_grasp"
    elif holds:
        if not dst_ok:
            outcome = "stacked_dst_moved"
        elif not third_ok:
            outcome = "stacked_third_moved"
        else:
            outcome = "success"
    elif not ms.transported:
        # The source never reached the destination, so it was lost before the placement stage - whether the
        # jaws opened somewhere else or the object escaped them. released_in_air is reserved for "arrived,
        # then let go above it".
        outcome = "dropped_in_transport" if trace.on_table(last) else "timeout_carrying"
    elif ms.release_step is None:
        # Arrived, but the jaws never opened again: still carrying it when the budget ran out.
        outcome = "timeout_carrying"
    elif not ms.placed:
        # Let go more than tol_place above (or below) stack height: a drop, not a placement.
        outcome = "released_in_air"
    elif not (trace.dst_upright(trace.frames[ms.placed_step]) and trace.dst_unmoved(trace.frames[ms.placed_step])):
        # The destination was ALREADY displaced when the object was set down, so the placement had no sound
        # target and the failure is not attributable to the placement geometry. Tested at the release frame,
        # not at the end: measured on test_cl, trial 46's destination had moved 38 mm 52 steps *before* the
        # release (correctly this label), while trial 21's was untouched at release - 0.14 mm - and only moved
        # at step 320, 76 steps *after* it, because the retreating arm clipped it. Judging at the final frame
        # blamed the target for a misplacement that had already happened.
        outcome = "stacked_dst_moved"
    else:
        # Set down on the destination but the stack is not there at the end. Whether the source's centre was
        # inside the destination footprint at that moment is what separates "should have held, something moved
        # it" from "was never going to hold".
        outcome = "knocked_off" if trace.stably_on_dst(trace.frames[ms.placed_step]) else "misplaced"

    metrics = {
        "steps": len(trace.frames),
        "lifted_step": ms.lifted_step,
        "transported_step": ms.transported_step,
        "stacked_step": ms.stacked_step,
        "release_step": ms.release_step,
        "placed_step": ms.placed_step,
        "centered_at_release": (
            None if ms.release_step is None else trace.over_dst(trace.frames[ms.release_step - 1])),
        "release_dz_mm": (
            None if ms.release_step is None
            else round(float(trace.frames[ms.release_step - 1].src_pos[2] - trace.stack_z) * 1000, 2)),
        "release_xy_offset_mm": (
            None if ms.release_step is None
            else round(float(np.linalg.norm(trace.frames[ms.release_step - 1].src_pos[:2]
                                           - trace.frames[ms.release_step - 1].dst_pos[:2])) * 1000, 2)),
        "release_speed_mm_s": (
            None if ms.release_step is None
            else round(trace.frames[ms.release_step - 1].src_speed * 1000, 1)),
        # The destination's displacement at the release frame, which is what decides whether a failed placement
        # is blamed on a shifted target or on the placement itself. dst_moved_mm is the same thing at the end.
        "release_dst_moved_mm": (
            None if ms.release_step is None
            else round(float(np.linalg.norm(trace.frames[ms.release_step - 1].dst_pos - trace.dst_pos0)) * 1000, 2)),
        "max_src_z_mm": round(float(max(f.src_pos[2] for f in trace.frames)) * 1000, 1),
        "lift_threshold_mm": round(trace.lift_z * 1000, 1),
        "stack_height_mm": round(trace.stack_z * 1000, 1),
        "final_xy_offset_mm": round(float(np.linalg.norm(last.src_pos[:2] - last.dst_pos[:2])) * 1000, 2),
        "final_dz_error_mm": round(float(last.src_pos[2] - trace.stack_z) * 1000, 2),
        "final_src_speed_mm_s": round(last.src_speed * 1000, 1),
        "final_settle_disp_mm": round(
            float(np.linalg.norm(last.src_pos - trace.frames[max(0, last_i - tol.settle_window_steps)].src_pos))
            * 1000, 2),
        "final_gripper_mm": round(last.gripper * 1000, 1),
        "dst_tilt_deg": round(tilt_deg(last.dst_quat), 2),
        "dst_moved_mm": round(float(np.linalg.norm(last.dst_pos - trace.dst_pos0)) * 1000, 2),
        "third_moved_mm": round(float(np.linalg.norm(last.third_pos - trace.third_pos0)) * 1000, 2),
        "third_disturbed": ms.third_disturbed,
        "dst_toppled": ms.dst_toppled,
    }
    return Assessment(outcome=outcome, milestones=ms, metrics=metrics)
