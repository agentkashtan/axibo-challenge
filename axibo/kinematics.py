"""Numpy forward kinematics for the PiperX arm: joint angles -> gripper pose.

The dataset records joint angles but not the gripper pose, and the quantity that decides whether a placement
succeeds is where the object sits *in the jaws* - `src_pos - tcp`, constant while the object is held. A value
function given only joint angles would have to learn this chain itself, which it cannot do from a few hundred
episode-level labels. So it is derived here instead, at load time, from the `sim.qpos` already in the dataset.

Pure numpy and vectorised over frames on purpose: `axibo/sim.py` could answer the same question, but only by
building a Genesis scene, which would couple training to the simulator and cost a scene build per run.

Chain read from `assets/urdf/piper_x_d435.urdf` - six revolute joints, each rotating about its own local z:

    T_Link6 = prod_i [ Translate(xyz_i) . R_rpy(rpy_i) . Rz(q_i) ]
    tcp     = T_Link6 . GRASP_LOCAL_POINT
"""

import numpy as np
from scipy.spatial.transform import Rotation

# (xyz, rpy) of joint1..joint6 from the URDF. Every axis is (0, 0, 1), so only Rz(q) varies.
JOINT_ORIGINS = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.13586, 3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, 2.8377)),
    ((0.27264, 0.0, 0.0), (0.0, 0.0, 0.075129)),
    ((0.076857, 0.00062395, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.035, 0.0, 0.0), (1.5708, 0.0, 1.5708)),
)
GRASP_LOCAL_POINT = (0.0, 0.0, 0.125)  # finger-pad centre in the Link6 frame; mirrors axibo/sim.py


def _static(xyz, rpy) -> np.ndarray:
    """Translate(xyz) . R_rpy(rpy) as a 4x4. URDF rpy is fixed-axis roll-pitch-yaw = scipy extrinsic "xyz"."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


_A = np.stack([_static(xyz, rpy) for xyz, rpy in JOINT_ORIGINS])  # (6, 4, 4)


def ee_pose(q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, 6) joint angles -> (tcp (N,3), link6_pos (N,3), link6_quat (N,4) as wxyz)."""
    q = np.asarray(q_arm, dtype=np.float64).reshape(-1, 6)
    n = len(q)
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    c, s = np.cos(q), np.sin(q)
    for i in range(6):
        rz = np.zeros((n, 4, 4))
        rz[:, 0, 0] = c[:, i]; rz[:, 0, 1] = -s[:, i]
        rz[:, 1, 0] = s[:, i]; rz[:, 1, 1] = c[:, i]
        rz[:, 2, 2] = 1.0; rz[:, 3, 3] = 1.0
        T = T @ _A[i] @ rz
    link6_pos = T[:, :3, 3]
    link6_quat = Rotation.from_matrix(T[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    tcp = (T @ np.append(GRASP_LOCAL_POINT, 1.0))[:, :3]
    return tcp.astype(np.float32), link6_pos.astype(np.float32), link6_quat.astype(np.float32)
