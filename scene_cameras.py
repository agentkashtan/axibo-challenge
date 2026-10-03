"""Scene smoke test: PiperX + D435 wrist mount, a wrist camera, a top-down camera, and the 3 task objects.

The arm holds its zero pose (no motion) for 10 s while both cameras render into their own windows.

Run from the project root:
    source .venv/bin/activate
    python scene_cameras.py
"""

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs

from axibo.backend import gs_backend

URDF_PATH = "assets/urdf/piper_x_d435.urdf"  # patched copy, see assets/SOURCE.txt
DT = 0.01  # [s]
DURATION = 6 * 10.0  # [s]
RENDER_EVERY = 10  # render cameras every N sim steps (-> 10 Hz)
CAM_RES = (320, 240)

# Placeholder object geometry/placement — these are Task 1 design decisions, not final values.
CUBE_SIZE = 0.04  # [m] edge length
CYL_RADIUS = 0.02  # [m]
CYL_HEIGHT = 0.05  # [m]
RED = (0.85, 0.1, 0.1)
BLUE = (0.1, 0.2, 0.85)


def urdf_T(xyz, rpy):
    """4x4 transform from a URDF <origin> (rpy = fixed-axis roll, pitch, yaw)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def link_world_T(link):
    """4x4 world pose of a Genesis link (Genesis quats are w, x, y, z; scipy wants x, y, z, w)."""
    T = np.eye(4)
    T[:3, 3] = link.get_pos().cpu().numpy()
    T[:3, :3] = Rotation.from_quat(link.get_quat().cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
    return T


# Genesis cameras follow the OpenGL convention (look along -z, y up); ROS camera links use x forward, z up.
T_ROS_TO_GL = np.eye(4)
T_ROS_TO_GL[:3, :3] = np.array(
    [
        [0.0, 0.0, -1.0],  # GL x (right) = -ROS y, GL z (back) = -ROS x
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],  # GL y (up) = ROS z
    ]
)

# Link6 -> camera_link, copied from the d435 URDF's fixed joints, plus a small push forward past the lens plate.
T_LINK6_TO_WRIST_CAM = (
    urdf_T((0.029, -0.069, 0.022), (0.0, -1.22, 1.57))  # d435_camera_joint
    @ urdf_T((0.0106, 0.0175, 0.0125), (0.0, 0.0, 0.0))  # camera_link_joint
    @ urdf_T((0.015, 0.0, 0.0), (0.0, 0.0, 0.0))  # in front of the housing
    @ T_ROS_TO_GL
)

gs.init(backend=gs_backend())

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT),
    viewer_options=gs.options.ViewerOptions(camera_pos=(1.0, -0.8, 0.8), camera_lookat=(0.25, 0.0, 0.1)),
    show_viewer=True,
)

scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, fixed=True))

# Robot faces +x; at zero pose the gripper sits around (0.23, 0, 0.23). Objects start slightly above the table.
red_cube = scene.add_entity(
    gs.morphs.Box(size=(CUBE_SIZE,) * 3, pos=(0.30, 0.10, CUBE_SIZE / 2 + 0.005)),
    surface=gs.surfaces.Default(color=RED),
)
red_cylinder = scene.add_entity(
    gs.morphs.Cylinder(radius=CYL_RADIUS, height=CYL_HEIGHT, pos=(0.35, 0.0, CYL_HEIGHT / 2 + 0.005)),
    surface=gs.surfaces.Default(color=RED),
)
blue_cube = scene.add_entity(
    gs.morphs.Box(size=(CUBE_SIZE,) * 3, pos=(0.30, -0.10, CUBE_SIZE / 2 + 0.005)),
    surface=gs.surfaces.Default(color=BLUE),
)

wrist_cam = scene.add_camera(res=CAM_RES, fov=70, near=0.01, far=3.0, GUI=True)
top_cam = scene.add_camera(res=CAM_RES, pos=(0.30, 0.0, 1.0), lookat=(0.30, 0.0, 0.0), up=(1.0, 0.0, 0.0), fov=45, GUI=True)

scene.build()

wrist_cam.attach(robot.get_link("Link6"), T_LINK6_TO_WRIST_CAM)

# Hold every joint at its current (zero) position so the arm doesn't sag.
dofs = list(range(robot.n_dofs))
robot.control_dofs_position(np.zeros(robot.n_dofs), dofs)

link6 = robot.get_link("Link6")
link6_frame = None  # debug frame drawn at Link6 (red = x, green = y, blue = z); redrawn every step

test =  robot.get_qpos()
test[3] = np.pi / 4

for step in range(int(DURATION / DT)):
    #test[6] += 0.05 / 10.0 * DT
    #test[7] -= 0.05 / 10.0 * DT
    robot.control_dofs_position(test, [i for i in range(8)])
    scene.step()

    if link6_frame is not None:
        scene.clear_debug_object(link6_frame)
    t = np.eye(4)
    t[:3, 3] = [0.0, 0.0, 0.175]
    link6_frame = scene.draw_debug_frame(link_world_T(link6) @ t , axis_length=0.08, origin_size=0.006, axis_radius=0.002)

    if step % RENDER_EVERY == 0:
        wrist_cam.move_to_attach()
        wrist_cam.render()
        top_cam.render()

"""
  ee  = robot.get_link("Link6")
  arm = list(range(6))                                  # IK moves joints 1–6 only, never the fingers
  TIP = np.array([0.0, 0.0, 0.175])                     # point between the fingertips, in Link6 frame (my guess; verify with 
  'L' in the viewer)

  pos  = np.array([[0.25, 0.0, 0.10], ...])             # (N, 3) world targets for the fingertip point
  quat = np.tile([0.0, 1.0, 0.0, 0.0], (N, 1))          # (w,x,y,z) = 180° about x → gripper points straight down

  q, err = robot.inverse_kinematics(
      link=ee, pos=pos, quat=quat,
      local_point=TIP,              # solve for the fingertips, not Link6's origin
      dofs_idx_local=arm,
      return_error=True,            # err: (N, 6) = position [m] + rotation error per env
  )                                  # q: (N, 8) full joint vector

"""