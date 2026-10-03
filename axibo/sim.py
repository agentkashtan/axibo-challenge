"""Genesis scene for the PiperX stacking task: robot, three objects, three cameras, and a 7-dim robot interface.

State / action convention (shared by data collection, policy and eval):
    index 0-5 : arm joints joint1..joint6, absolute position [rad]
    index 6   : gripper opening w [m] in [0, 0.05]; fingers are commanded as joint7 = +w, joint8 = -w
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs

URDF_PATH = "assets/urdf/piper_x_d435.urdf"  # patched copy, see assets/SOURCE.txt

OBJECT_NAMES = ("red cube", "red cylinder", "blue cube")
ARM_DOFS = [0, 1, 2, 3, 4, 5]
FINGER_DOFS = [6, 7]
GRIPPER_MAX = 0.05  # [m], joint7 upper limit
GRASP_LOCAL_POINT = (0.0, 0.0, 0.125)  # finger-pad center in Link6 frame; fingers open along Link6 x
FINGERTIP_BELOW_PAD = 0.015  # fingertips sit at Link6 z = 0.140, i.e. 1.5 cm past the pad center
STATE_DIM = ACTION_DIM = 7
CAMERA_NAMES = ("top", "wrist", "side")

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

# Link6 -> wrist camera optical frame, from the d435 URDF's fixed joints plus a small push past the lens plate.
T_LINK6_TO_WRIST_CAM = (
    urdf_T((0.029, -0.069, 0.022), (0.0, -1.22, 1.57))  # d435_camera_joint
    @ urdf_T((0.0106, 0.0175, 0.0125), (0.0, 0.0, 0.0))  # camera_link_joint
    @ urdf_T((0.015, 0.0, 0.0), (0.0, 0.0, 0.0))  # in front of the housing
    @ T_ROS_TO_GL
)


@dataclass
class FixedCamera:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class ObjectPose:
    """World pose of an object's center. quat is (w, x, y, z), Genesis convention."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class SimConfig:
    dt: float = 1.0 / 90.0  # physics step [s]
    control_hz: int = 30  # policy action rate; SmolVLA was pretrained on 30 fps data
    cam_res: tuple[int, int] = (512, 512)  # SmolVLA resizes to 512x512 internally
    wrist_fov: float = 70.0
    # Genesis defaults ambient_light to (0.1, 0.1, 0.1), which leaves the table in near-shadow and puts a hard arm
    # shadow across the objects. SmolVLA's frozen SigLIP encoder was pretrained on normally exposed web images.
    ambient_light: tuple[float, float, float] = (0.45, 0.45, 0.45)
    # Framed on the workspace centre (x 0.18-0.34, y +-0.15) instead of the wider earlier views, where the objects
    # covered ~15% of the frame. A true top-down camera was tried and rejected: it reads the layout perfectly at the
    # start but the arm, which approaches from above, fills the whole frame exactly during the grasp.
    # The fovs are the smallest (to the nearest 2 deg, +2 for margin) that keep every object fully inside both
    # frames anywhere in the workspace, on the table or stacked on a cube, checked by projection in /tmp/vis_check.
    top_cam: FixedCamera = field(default_factory=lambda: FixedCamera(pos=(0.55, 0.0, 0.42), lookat=(0.26, 0.0, 0.04), fov=56.0))
    # Low and near-horizontal, so stack height (what the success check measures) is directly visible, and the view
    # stays useful while the arm occludes the top camera.
    side_cam: FixedCamera = field(default_factory=lambda: FixedCamera(pos=(0.26, -0.45, 0.16), lookat=(0.26, 0.0, 0.05), fov=48.0))

    arm_kp: float = 2000.0  # kp=100 sagged 10-15 mm at the fingertip under gravity
    arm_kv: float = 200.0
    # Held objects slid down in the grip during lift/transfer (cylinder 19 mm, cube 8 mm at finger_kp=100). Stiffer
    # fingers alone didn't fix it (10 mm at kp=2000); Genesis' noslip post-solve did (0 mm). kp=1000 on top gives more
    # precise placement (cylinder 0.9 vs 3.2 mm). Cost: ~1.4x physics wall time.
    finger_kp: float = 1000.0
    finger_kv: float = 30.0
    noslip_iterations: int = 5  # Genesis recommends ~5 for manipulation
    # Arm joints at zero, gripper open: the demos need an open gripper before they descend, so starting closed cost
    # every episode a segment that holds the arm still while the fingers move (15 frames at time_scale 1.0). An
    # episode now begins and ends in the same gripper state, since it also ends open after the release.
    home_qpos: tuple[float, ...] = (0.0,) * 6 + (GRIPPER_MAX, -GRIPPER_MAX)  # placeholder arm pose (Task 1 decision)
    # [rad] uniform +-jitter added to the six arm joints at reset(). Without it every episode starts from the exact
    # same configuration, so the opening motion can be reproduced from memory instead of from the scene, and any
    # deviation at eval leaves the training distribution. The gripper is not jittered: it starts open by design.
    start_jitter_rad: float = 0.05

    # Genesis defaults to 1.0 and uses the max of the two geoms in contact, so setting it on the objects also covers
    # object-finger (and object-table, object-object) contacts. The cylinder touches the flat pads along a line, so it
    # is the friction-sensitive case. Placeholder value (Task 1 decision); range is [1e-2, 5.0].
    object_friction: float = 1.0

    # Off by default in Genesis (point-contact idealization: only sliding is resisted). With it off, a side-grasped
    # cylinder is a free hinge about the pad-normal axis: the two pad contacts sit on that axis, so tangential
    # friction there produces no torque about it, and the cylinder tips as soon as the table stops supporting it
    # (layout 3 of testv2_10: 0.5 -> 58.8 deg during the lift, landing upside down). Real pads resist this through
    # their contact patch. Enabling it kept the tilt at 0.2 deg and all three pairs on that layout succeeded.
    # Caveat: Genesis reuses the tangential coefficient, so the torsional resistance is coarse, and the option is
    # global (object-table and object-object contacts get it too). More contact_pruning_tolerance did not help.
    enable_torsional_friction: bool = True

    # Placeholder object geometry (Task 1 decision).
    cube_size: float = 0.04
    cyl_radius: float = 0.02
    cyl_height: float = 0.05

    @property
    def steps_per_action(self) -> int:
        return max(1, round(1.0 / (self.control_hz * self.dt)))


def object_height(name: str, cfg: SimConfig) -> float:
    return cfg.cyl_height if "cylinder" in name else cfg.cube_size


def object_half_width(name: str, cfg: SimConfig) -> float:
    """Half the distance between the grasped faces (cube: half edge, cylinder: radius)."""
    return cfg.cyl_radius if "cylinder" in name else cfg.cube_size / 2


@dataclass
class Observation:
    """Batched observation: images[name] is uint8 (N, H, W, 3), state is float32 (N, 7)."""

    images: dict[str, np.ndarray]
    state: np.ndarray

    def env(self, i: int) -> "Observation":
        """Single-env view: images (H, W, 3), state (7,)."""
        return Observation(images={k: v[i] for k, v in self.images.items()}, state=self.state[i])


LayoutArg = dict[str, ObjectPose] | list[dict[str, ObjectPose]]


class PiperStackingScene:
    """Batched Genesis scene (n_envs independent copies) exposing observe() / apply_action() / step_control()."""

    def __init__(
        self,
        object_poses: LayoutArg,
        n_envs: int = 1,
        cfg: SimConfig | None = None,
        show_viewer: bool = False,
        seed: int = 0,
    ):
        """object_poses: one layout for all envs, or a list with one layout per env."""
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)  # start-pose jitter only; seed it to reproduce a rollout
        self.cfg = cfg = cfg or SimConfig()
        self.object_poses = self._per_env_layouts(object_poses, list(range(n_envs)))

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt),
            rigid_options=gs.options.RigidOptions(
                noslip_iterations=cfg.noslip_iterations,
                enable_torsional_friction=cfg.enable_torsional_friction,
            ),
            vis_options=gs.options.VisOptions(split_envs=True, ambient_light=cfg.ambient_light),  # one image per env
            viewer_options=gs.options.ViewerOptions(camera_pos=(1.0, -0.8, 0.8), camera_lookat=(0.25, 0.0, 0.1)),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=URDF_PATH, fixed=True))
        self.objects = self._add_objects()
        self.cameras = self._add_cameras()
        self.scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))  # spacing only affects the viewer

        self.cameras["wrist"].attach(self.robot.get_link("Link6"), T_LINK6_TO_WRIST_CAM)
        self.robot.set_dofs_kp([cfg.arm_kp] * len(ARM_DOFS), ARM_DOFS)
        self.robot.set_dofs_kv([cfg.arm_kv] * len(ARM_DOFS), ARM_DOFS)
        self.robot.set_dofs_kp([cfg.finger_kp] * len(FINGER_DOFS), FINGER_DOFS)
        self.robot.set_dofs_kv([cfg.finger_kv] * len(FINGER_DOFS), FINGER_DOFS)

        lower, upper = (t.cpu().numpy().reshape(-1) for t in self.robot.get_dofs_limit())
        self._qpos_low, self._qpos_high = lower.astype(np.float32), upper.astype(np.float32)
        self.action_low = np.append(lower[ARM_DOFS], 0.0).astype(np.float32)
        self.action_high = np.append(upper[ARM_DOFS], GRIPPER_MAX).astype(np.float32)

    @staticmethod
    def _check_poses(object_poses: dict[str, ObjectPose]) -> dict[str, ObjectPose]:
        if set(object_poses) != set(OBJECT_NAMES):
            raise ValueError(f"object_poses must have exactly the keys {OBJECT_NAMES}, got {tuple(object_poses)}")
        return object_poses

    def _per_env_layouts(self, object_poses: LayoutArg, envs_idx: list[int]) -> list[dict[str, ObjectPose]]:
        layouts = [object_poses] * len(envs_idx) if isinstance(object_poses, dict) else list(object_poses)
        if len(layouts) != len(envs_idx):
            raise ValueError(f"got {len(layouts)} layouts for {len(envs_idx)} envs")
        return [self._check_poses(layout) for layout in layouts]

    def _add_objects(self) -> dict:
        cfg = self.cfg
        cube = dict(size=(cfg.cube_size,) * 3)
        cylinder = dict(radius=cfg.cyl_radius, height=cfg.cyl_height)
        specs = {
            "red cube": (gs.morphs.Box, cube, RED),
            "red cylinder": (gs.morphs.Cylinder, cylinder, RED),
            "blue cube": (gs.morphs.Box, cube, BLUE),
        }
        first = self.object_poses[0]  # spawn pose; reset() sets the real per-env poses
        return {
            name: self.scene.add_entity(
                morph(**geometry, pos=first[name].pos, quat=first[name].quat),
                material=gs.materials.Rigid(friction=cfg.object_friction),
                surface=gs.surfaces.Default(color=color),
            )
            for name, (morph, geometry, color) in specs.items()
        }

    def _add_cameras(self) -> dict:
        cfg = self.cfg
        fixed = {
            name: self.scene.add_camera(res=cfg.cam_res, pos=c.pos, lookat=c.lookat, up=c.up, fov=c.fov)
            for name, c in (("top", cfg.top_cam), ("side", cfg.side_cam))
        }
        wrist = self.scene.add_camera(res=cfg.cam_res, fov=cfg.wrist_fov, near=0.01, far=3.0)
        return {"top": fixed["top"], "wrist": wrist, "side": fixed["side"]}

    def reset(self, object_poses: LayoutArg | None = None, envs_idx: list[int] | None = None):
        """Robot to home pose, objects to their poses (velocities zeroed), for the selected envs (default: all).

        object_poses: optional new layout(s) for those envs (one dict for all, or one per env); stored, so later
        reset() calls reuse them. Swapping layouts this way avoids rebuilding the Genesis scene.
        """
        envs_idx = list(range(self.n_envs)) if envs_idx is None else list(envs_idx)
        if object_poses is not None:
            for env, layout in zip(envs_idx, self._per_env_layouts(object_poses, envs_idx)):
                self.object_poses[env] = layout
        home = np.tile(np.asarray(self.cfg.home_qpos, dtype=np.float32), (len(envs_idx), 1))
        if self.cfg.start_jitter_rad:
            j = self.cfg.start_jitter_rad
            home[:, ARM_DOFS] += self.rng.uniform(-j, j, size=(len(envs_idx), len(ARM_DOFS))).astype(np.float32)
            home = np.clip(home, self._qpos_low, self._qpos_high)
        self.robot.set_qpos(home, envs_idx=envs_idx)
        self.robot.control_dofs_position(home, envs_idx=envs_idx)
        for name, entity in self.objects.items():
            entity.set_pos(np.array([self.object_poses[e][name].pos for e in envs_idx], dtype=np.float32), envs_idx=envs_idx)
            entity.set_quat(np.array([self.object_poses[e][name].quat for e in envs_idx], dtype=np.float32), envs_idx=envs_idx)
        self.scene.step()

    def _batched(self, arr) -> np.ndarray:
        arr = arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)
        return arr.reshape(self.n_envs, *arr.shape[-1:]) if arr.ndim <= 2 else arr

    def qpos(self) -> np.ndarray:
        """(N, 8) raw joint positions (both fingers separately)."""
        return self._batched(self.robot.get_qpos())

    def qvel(self) -> np.ndarray:
        """(N, 8) raw joint velocities, in the same order as qpos()."""
        return self._batched(self.robot.get_dofs_velocity())

    def observe(self) -> Observation:
        self.cameras["wrist"].move_to_attach()
        res_w, res_h = self.cfg.cam_res
        # render() returns flipped views (negative strides) and drops the env axis for a single env.
        images = {
            name: np.ascontiguousarray(np.asarray(self.cameras[name].render()[0]).reshape(self.n_envs, res_h, res_w, 3))
            for name in CAMERA_NAMES
        }
        q = self.qpos()
        gripper = (q[:, 6] - q[:, 7]) / 2.0  # joint7 = +w, joint8 = -w
        state = np.column_stack([q[:, ARM_DOFS], gripper]).astype(np.float32)
        return Observation(images=images, state=state)

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) positions and (N, 4) w,x,y,z quaternions of an object."""
        entity = self.objects[name]
        return self._batched(entity.get_pos()), self._batched(entity.get_quat())

    def object_vel(self, name: str) -> np.ndarray:
        """(N, 3) linear velocity of an object's center."""
        return self._batched(self.objects[name].get_vel())

    def reseed(self, seed: int):
        """Restart the start-pose jitter stream, so a trial can be reproduced from its seed alone."""
        self.rng = np.random.default_rng(seed)

    def grasp_point_world(self, env_idx: int = 0) -> np.ndarray:
        """Current world position (3,) of the finger-pad center in one env."""
        link = self.robot.get_link("Link6")
        T = np.eye(4)
        T[:3, 3] = self._batched(link.get_pos())[env_idx]
        w, x, y, z = self._batched(link.get_quat())[env_idx]
        T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        return (T @ np.append(GRASP_LOCAL_POINT, 1.0))[:3]

    def apply_action(self, actions: np.ndarray) -> np.ndarray:
        """Set PD targets from (N, 7) actions. Returns a (N,) bool mask of envs whose action was clipped to limits."""
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n_envs, 7)
        clipped = np.clip(actions, self.action_low, self.action_high)
        self.robot.control_dofs_position(clipped[:, :6], ARM_DOFS)
        self.robot.control_dofs_position(np.column_stack([clipped[:, 6], -clipped[:, 6]]), FINGER_DOFS)
        # Ignore sub-mrad overshoot: at the zero home pose joints 2/3 rest on their limit (0) and gravity pushes
        # joint3 ~1e-4 rad past it, which isn't a policy problem.
        return ~np.all(np.isclose(actions, clipped, rtol=0.0, atol=1e-3), axis=1)

    def step_control(self):
        """Advance physics by one control period (1 / control_hz)."""
        for _ in range(self.cfg.steps_per_action):
            self.scene.step()
