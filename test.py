"""Smoke test: load the PiperX arm in Genesis and swing joint1 back and forth for 10 seconds.

Run from the project root (so relative asset paths and in-project caches work):
    source .venv/bin/activate
    python test.py
"""

import math

import genesis as gs

from axibo.backend import gs_backend

URDF_PATH = "assets/piper_x_description/urdf/piper_x_description.urdf"
DT = 0.01  # simulation step [s] (Genesis default)
DURATION = 10.0  # [s]
AMPLITUDE = 1.0  # joint1 swing amplitude [rad] (limit is ±2.618)
PERIOD = 4.0  # one full back-and-forth [s]

gs.init(backend=gs_backend())

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.0, 1.0, 0.8),
        camera_lookat=(0.0, 0.0, 0.25),
    ),
    show_viewer=True,
)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, fixed=True))
scene.build()

# Without PD gains/targets the joints just sag under gravity, so hold every DOF at 0.
arm_dofs = [robot.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 7)]
gripper_dofs = [robot.get_joint(f"joint{i}").dofs_idx_local[0] for i in (7, 8)]
all_dofs = arm_dofs + gripper_dofs

robot.set_dofs_kp([100.0] * 6 + [100.0] * 2, all_dofs)
robot.set_dofs_kv([10.0] * 6 + [10.0] * 2, all_dofs)

n_steps = int(DURATION / DT)
for step in range(n_steps):
    t = step * DT
    target = [0.0] * len(all_dofs)
    target[0] = AMPLITUDE * math.sin(2 * math.pi * t / PERIOD)  # joint1
    robot.control_dofs_position(target, all_dofs)
    scene.step()

    if step % 100 == 0:
        actual = robot.get_dofs_position(arm_dofs[:1]).item()
        print(f"t={t:4.1f}s  joint1 target={target[0]:+.3f}  actual={actual:+.3f} rad")
