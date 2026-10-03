"""Queue-based closed-loop rollout with a simulated inference latency.

The simulator is not real-time: physics only advances inside `sim.step_control()`, so the wall-clock spent in
`predict_chunk` costs zero simulated time and a naive loop has no chunk-boundary stall to remove. This module puts
the latency back into the sim timeline: a chunk requested at step s becomes available at step s + L, where L is the
measured (or fixed) inference time converted to control steps. Between those steps the robot keeps executing the
queue it already had - which is exactly what an asynchronous client does - or holds its last action if the queue is
empty, which is exactly what a synchronous one does.

One knob spans both regimes. `queue_threshold` (k) is the queue length at or below which a new chunk is requested:

    k = 0        request only once the queue is empty -> the arm holds for L steps  = naive synchronous
    k >= L       the chunk always lands before the queue runs dry                   = asynchronous, no stall
    k = H        a request every control step (ACT's g = 1), unaffordable at 200 ms

Follows SmolVLA's Algorithm 1 (arXiv 2506.01844, section 3.3) with the hold branch added so the k = 0 baseline is
the same code path rather than a separate implementation.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from axibo.outcome import OutcomeTolerances, Trace
from axibo.policy import SmolVLARunner
from axibo.sim import PiperStackingScene


@dataclass
class RolloutConfig:
    queue_threshold: int = 0  # k: request a chunk when len(queue) <= k
    latency_mode: str = "fixed"  # "fixed" (reproducible) or "measured" (real per-call jitter)
    latency_ms: float = 200.0  # used when latency_mode == "fixed"
    max_steps: int = 500  # control-step budget for the episode (16.7 s at 30 Hz)
    settle_steps: int = 120  # steps of continued policy control after the object is released (4 s)
    blend: str = "none"  # overlap handling on arrival: "none" = hard switch, "linear" = ramp over the overlap
    log_switches: bool = False  # print one line per chunk switch (debugging; too noisy for a 500-trial run)


@dataclass
class RolloutStats:
    """Everything about *how* the episode was executed; the outcome itself comes from axibo.outcome.analyse."""

    steps: int = 0
    calls: int = 0  # inference calls
    stalled_steps: int = 0
    latency_s: list[float] = field(default_factory=list)  # every sample, so p99 is real
    latency_steps: list[int] = field(default_factory=list)
    switch_steps: list[int] = field(default_factory=list)
    discarded: list[int] = field(default_factory=list)  # queued actions dropped at each switch (= max(0, k - L))
    chunk_disagreement: list[float] = field(default_factory=list)  # mean |a_old - a_new| over the overlap
    clipped_steps: int = 0
    detected_step: int | None = None  # when the stack detector first fired


def run_episode(
    sim: PiperStackingScene,
    runner: SmolVLARunner,
    task: str,
    source: str,
    destination: str,
    cfg: RolloutConfig,
    tol: OutcomeTolerances | None = None,
    on_step=None,
    on_action=None,
) -> tuple[Trace, RolloutStats]:
    """Run one episode.

    `on_step(step, frame, stats)` fires after each control step, once the resulting frame exists (the live view).
    `on_action(step, action, stalled)` fires just before the action is applied, so a recorder can capture the
    observation it was issued from.
    """
    hz = sim.cfg.control_hz
    fixed_steps = max(0, math.ceil(cfg.latency_ms * hz / 1000.0))

    trace = Trace.start(sim, 0, source, destination, tol)
    stats = RolloutStats()

    queue: deque = deque()
    pending: dict | None = None
    pending_switch = False
    last_action = np.append(sim.qpos()[0][:6], (sim.qpos()[0][6] - sim.qpos()[0][7]) / 2).astype(np.float32)
    countdown: int | None = None
    transported = False
    step = 0

    trace.record(sim, 0, last_action[6], qpos=sim.qpos()[0], qvel=sim.qvel()[0], action=last_action)

    while step < cfg.max_steps or countdown is not None:
        # 1. Trigger a chunk. Consumes wall-clock but no simulated time: the observation is captured before the
        #    timer starts, and the chunk only becomes visible to the robot L steps later.
        # The budget stops the *attempt*, but the settle window is explicitly "the policy keeps acting", so chunks
        # are still requested while it runs - otherwise the arm would hold for 4 s and every such step would be
        # counted as a stall.
        if pending is None and len(queue) <= cfg.queue_threshold and (step < cfg.max_steps or countdown is not None):
            obs = sim.observe().env(0)
            t0 = perf_counter()
            chunk = runner.predict_chunk(obs, task)
            dt_wall = perf_counter() - t0
            L = fixed_steps if cfg.latency_mode == "fixed" else max(1, math.ceil(dt_wall * hz))
            pending = {"arrival_step": step + L, "chunk": chunk, "obs_step": step, "executed": 0}
            stats.calls += 1
            stats.latency_s.append(dt_wall)
            stats.latency_steps.append(L)

        # 2. Pick this step's action: from the queue, or hold the last one if the chunk has not landed yet.
        if queue:
            action, stalled = queue.popleft(), False
            if pending is not None:
                pending["executed"] += 1
        else:
            action, stalled = last_action, True
            stats.stalled_steps += 1

        # 3. Advance the world by exactly one control step. on_action fires *before* the step so a recorder sees
        #    the observation the action was issued from - pairing obs_t with action_t, as collect_demos.py does.
        #    on_step (below) fires after, when the resulting frame exists, and is what the live view uses.
        if on_action is not None:
            on_action(step, action, stalled)
        stats.clipped_steps += int(sim.apply_action(action[None])[0])
        sim.step_control()
        last_action = action
        q = sim.qpos()[0]
        frame = trace.record(
            sim, 0, (q[6] - q[7]) / 2, qpos=q, qvel=sim.qvel()[0], action=action,
            stalled=stalled, switch=pending_switch,
        )
        if pending_switch:
            stats.switch_steps.append(step)
            pending_switch = False

        # 4. Outcome bookkeeping. The release of a transported object starts a settle window during which the
        #    policy keeps acting; the label itself is decided offline by axibo.outcome.analyse.
        transported = transported or trace.arrived_now()
        if countdown is None:
            if transported and trace.release_detected_now():
                countdown = cfg.settle_steps
                stats.detected_step = step
        else:
            countdown -= 1
            # A genuine retry cancels the window: without this, a policy that drops the object and picks it back
            # up is cut off mid-recovery and scored a failure (1 of 10 local trials retried). The test is
            # above_dst, not lifted: for a cube on a cube stack_z == lift_z == 60 mm, so a correctly placed cube
            # already counts as "lifted" and any gripper twitch during the settle would cancel the window.
            # Clearing stack height by the hover margin means it really was picked back up.
            if frame.gripper < trace.tol.closed_grip and trace.above_dst(frame):
                countdown = None

        if on_step is not None:
            on_step(step, frame, stats)

        # 5. Arrival: the chunk is available from the next step on. Index alignment is by *actions executed*, not
        #    by elapsed time: the chunk predicted from the observation at obs_step continues from wherever the robot
        #    actually got to. Asynchronously it drained L actions from the old queue meanwhile, so it resumes at
        #    index L; synchronously (k=0) it stood still, the observed state is still current, and it resumes at
        #    index 0 - using L there would discard valid actions and jump the target forward. Partial overlap
        #    (0 < k < L) lands in between at k.
        if pending is not None and step + 1 >= pending["arrival_step"]:
            offset = pending["executed"]
            new_chunk = pending["chunk"]
            if offset < len(new_chunk):
                overlap = min(len(queue), len(new_chunk) - offset)
                blended = None
                if overlap:
                    old = np.asarray(list(queue)[:overlap], dtype=np.float32)
                    new = np.asarray(new_chunk[offset : offset + overlap], dtype=np.float32)
                    stats.chunk_disagreement.append(float(np.abs(old - new).mean()))
                    if cfg.blend == "linear":
                        # The arm has to cover the gap between the two plans either way; a ramp spreads it over the
                        # overlap instead of paying it in one step, cutting the peak by ~1/overlap. lam ends at 1 so
                        # the ramp finishes fully on the new chunk. The gripper (dim 6) is switched hard: averaging
                        # "open" with "closed" commands a half-closed gripper, which at the grasp drops the object.
                        lam = ((np.arange(overlap, dtype=np.float32) + 1) / overlap)[:, None]
                        blended = (1.0 - lam) * old + lam * new
                        blended[:, 6] = new[:, 6]
                stats.discarded.append(len(queue))
                if cfg.log_switches:
                    L_used = pending["arrival_step"] - pending["obs_step"]
                    print(f"      switch @{step + 1:4d}  L={L_used:2d}  pulled={offset:2d} "
                          f"held={L_used - offset:2d}  dropped={len(queue):2d}  "
                          f"resume=chunk[{offset}:] ({len(new_chunk) - offset} actions)"
                          + (f"  disagree={stats.chunk_disagreement[-1]:.4f}" if overlap else ""))
                queue.clear()
                if blended is None:
                    queue.extend(new_chunk[offset:])  # hard switch
                else:
                    queue.extend(blended)  # the ramp...
                    queue.extend(new_chunk[offset + overlap :])  # ...then the new chunk on its own
                pending_switch = True
            pending = None

        step += 1
        if countdown is not None and countdown <= 0:
            break

    stats.steps = step
    return trace, stats
