"""Synthetic traces exercising every branch of axibo.outcome.analyse(). No sim, no policy, no checkpoint.

The policy produces some of these outcomes a handful of times in 500 trials (knocked_off,
stacked_third_moved, stacked_dst_moved), so a rollout-based check would never cover the tree. Each case here is
a hand-built frame sequence with the geometry that should force exactly one label.

    PYTHONPATH=. python test_outcome.py
"""
import numpy as np
from axibo.outcome import Frame, OutcomeTolerances, Trace, analyse

H = 0.05          # both objects 5 cm tall
STACK_Z = H + H / 2     # 0.075 source center when resting on the destination
REST_Z = H / 2          # 0.025
LIFT_Z = H / 2 + H      # 0.075 ... equals STACK_Z here, which is fine
DST = np.array([0.26, 0.0, REST_Z])
THIRD = np.array([0.30, 0.12, REST_Z])
OPEN, CLOSED = 0.050, 0.012


def trace() -> Trace:
    return Trace(source="red cube", destination="blue cube", third="red cylinder",
                 h_src=H, h_dst=H, w_dst=0.025, dst_is_cube=True,
                 dst_pos0=DST.copy(), third_pos0=THIRD.copy(), tol=OutcomeTolerances())


def push(t, n, xy, z, grip, speed=0.0, dst=DST, third=THIRD, dst_quat=(1.0, 0, 0, 0)):
    for _ in range(n):
        t.frames.append(Frame(src_pos=np.array([xy[0], xy[1], z]), src_speed=speed,
                              dst_pos=np.asarray(dst, float), dst_quat=np.asarray(dst_quat, float),
                              third_pos=np.asarray(third, float), gripper=grip))


def episode(**kw):
    """Common skeleton: approach, close, lift, carry over the destination, then a per-case ending."""
    t = trace()
    push(t, 20, DST[:2] + np.array([0.08, 0.0]), REST_Z, OPEN)        # approach, object on the table
    push(t, 5, DST[:2] + np.array([0.08, 0.0]), REST_Z, CLOSED)       # close on it
    # 60 mm off: clear of the grown transport footprint (w_dst + transport_margin = 40 mm), so carrying alone
    # never latches TRANSPORTED - only the explicit over-the-destination push below does.
    push(t, 20, DST[:2] + np.array([0.06, 0.0]), 0.13, CLOSED, 0.1)   # lift and carry (above lift_z)
    if kw.get("transport", True):
        push(t, 10, DST[:2], STACK_Z + 0.03, CLOSED, 0.03)            # over the destination with clearance
    return t


def settle(t, xy, z, grip=OPEN, speed=0.0, **kw):
    push(t, 130, xy, z, grip, speed, **kw)                            # > settle window, so `settled` holds


CASES = []

# success: set down centered, stays put
t = episode(); push(t, 2, DST[:2], STACK_Z, CLOSED, 0.01); settle(t, DST[:2], STACK_Z)
CASES.append(("success", t))

# knocked_off: set down centered, then displaced onto the table
t = episode(); push(t, 2, DST[:2], STACK_Z, CLOSED, 0.01); settle(t, DST[:2] + np.array([0.05, 0]), REST_Z)
CASES.append(("knocked_off", t))

# misplaced: set down but off-center, then falls
off = DST[:2] + np.array([0.035, 0.005])   # outside the 25 mm half-width square in the dst frame
t = episode(); push(t, 2, off, STACK_Z, CLOSED, 0.01); settle(t, off + np.array([0.03, 0]), REST_Z)
CASES.append(("misplaced", t))

# misplaced, second form: set down off-center and still perched there at the end (the old overhang case that
# used to fall through to `timeout`)
t = episode(); push(t, 2, off, STACK_Z, CLOSED, 0.01); settle(t, off, STACK_Z)
CASES.append(("misplaced", t))

# released_in_air: gripper opens 30 mm above stack height - well past tol_place (10 mm), so the case is
# not sitting on the threshold
t = episode(); push(t, 2, DST[:2], STACK_Z + 0.03, CLOSED, 0.02); settle(t, DST[:2], REST_Z)
CASES.append(("released_in_air", t))

# stacked_dst_moved: success geometry but the destination has been shoved
moved = DST + np.array([0.03, 0, 0])
t = episode(); push(t, 2, moved[:2], STACK_Z, CLOSED, 0.01, dst=moved); settle(t, moved[:2], STACK_Z, dst=moved)
CASES.append(("stacked_dst_moved", t))

# stacked_third_moved: success geometry but the third object was disturbed
bumped = THIRD + np.array([0.04, 0, 0])
t = episode(); push(t, 2, DST[:2], STACK_Z, CLOSED, 0.01, third=bumped); settle(t, DST[:2], STACK_Z, third=bumped)
CASES.append(("stacked_third_moved", t))

# dropped_in_transport, second form: jaws open while carrying, but the source never reached the destination
# (this is the real trial 19: lifted to 124 mm, released 41.6 mm off the cylinder's axis and 11.6 mm below
# stack height, landed on the table)
far = DST[:2] + np.array([0.055, 0.0])   # outside the grown footprint, so TRANSPORTED never latches
t = episode(transport=False); push(t, 2, far, STACK_Z - 0.012, CLOSED, 0.39); settle(t, far, REST_Z)
CASES.append(("dropped_in_transport", t))

# dropped_in_transport: object escapes a still-closed gripper mid-carry
t = episode(transport=False); settle(t, DST[:2] + np.array([0.06, 0]), REST_Z, grip=CLOSED)
CASES.append(("dropped_in_transport", t))

# timeout_carrying: never lets go, still holding at the end
t = episode(); settle(t, DST[:2], STACK_Z + 0.03, grip=CLOSED)
CASES.append(("timeout_carrying", t))

# no_grasp: never reaches lift height
t = trace(); push(t, 40, DST[:2] + np.array([0.08, 0]), REST_Z, OPEN); settle(t, DST[:2] + np.array([0.08, 0]), REST_Z)
CASES.append(("no_grasp", t))

bad = 0
for expect, tr in CASES:
    a = analyse(tr)
    ok = a.outcome == expect
    bad += not ok
    print(f"  {'ok ' if ok else 'FAIL'}  expect {expect:22} got {a.outcome:22} "
          f"release={a.metrics['release_step']} placed={a.metrics['placed_step']} "
          f"centered={a.metrics['centered_at_release']} dz={a.metrics['release_dz_mm']}")
print("\nall pass" if not bad else f"\n{bad} FAILURES")
raise SystemExit(1 if bad else 0)
