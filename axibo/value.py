"""Distributional value function for RECAP (arXiv 2511.14759 section V-C).

V maps a state and the language command to a distribution over B=201 discretised return bins, trained by
cross-entropy against the Monte-Carlo return of the episode it came from. Returns are what the paper's reward
definition implies (verbatim: "r_t = 0 if t=T and success, -C_fail if t=T and failure, -1 otherwise"):

    R_t = -(t_stacked - t)          success: literally "frames until the stack exists", 0 at and after it
          -(T - t) - C_fail         failure

Nothing is stored per frame: R_t is this closed form over the frame index plus three per-episode scalars from
`rollout_log.csv`, so C_fail and the normalisation can be swept without re-collecting.

**This V reads simulator state, not images** - a deliberate deviation. RECAP's contribution is the mechanism
(value -> binarised advantage -> conditioning token -> high-advantage at inference), which is identical
whichever way V perceives the world, and V runs *only offline to label*, so the policy still sees nothing but
one bit. The images are available (`collect_rollouts.py` records all three cameras) but a frozen-VLM feature
pass over ~170k frames costs 30-50 min and carries the risk that pooled features cannot resolve a ~15 mm grasp
offset - with no way to tell an encoder problem from a reward-definition problem afterwards. The privileged
version needs no precompute because the object poses are already a dataset column (`axibo/recording.py`,
`sim.object_poses`, deliberately without the `observation.` prefix so they are never a policy input). This is
the standard asymmetric-critic trick, and it must be reported as a deviation: it would not transfer to a real
robot, where that state does not exist.
"""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from axibo.kinematics import ee_pose
from axibo.sim import OBJECT_NAMES

N_BINS = 201  # paper: B = 201

# The improvement indicator, verbatim from the paper (section V-D): "inputting 'Advantage: positive' when
# I_t=True, and 'Advantage: negative' otherwise". Appended to the task string, so training and inference must
# use byte-identical text - hence one definition, imported by train_recap.py and eval_async.py.
ADVANTAGE_POSITIVE = "Advantage: positive"
ADVANTAGE_NEGATIVE = "Advantage: negative"

# Feature layout, 33 dims. Poses are ordered by ROLE (source, destination, third), not by object identity:
# the command is always "put the {source} on the {destination}", so its entire information content is which
# object is which, and this ordering encodes that bijectively. That makes V language-conditioned without text
# tokens and without a task one-hot - which would split the data into five near-separate functions instead of
# sharing every frame. Lossless for this instruction template; not a general language interface.
FEATURE_BLOCKS = (
    ("source pose", 7),        # x y z qw qx qy qz
    ("destination pose", 7),
    ("third pose", 7),
    ("src_pos - dst_pos", 3),  # the placement error: what decides P(fail) once the object is over the target
    ("gripper pos", 3),        # TCP, derived by FK from the recorded joints (axibo/kinematics.py)
    ("gripper quat", 4),       # Link6 orientation
    ("src_pos - tcp", 3),      # the GRASP offset: constant while held, and the cause of the misplacements.
                               # Without it V would have to learn forward kinematics from a few hundred
                               # episode labels to tell a good grasp from a bad one, which it cannot.
    ("robot state", 7),        # joint1-6 + gripper opening, so V can tell "held" from "released"
    ("shape flags", 2),        # src_is_cube, dst_is_cube: cylinder destinations are 57% vs 71% for cubes
)
FEATURE_DIM = sum(n for _, n in FEATURE_BLOCKS)


@dataclass
class ReturnConfig:
    """Reward shape and the mapping from raw returns onto (-1, 0).

    `c_fail` is the paper's "large constant ... chosen so as to ensure that failed episodes have low values".
    It is given no numeric value there, so ours is justified instead: 250 frames is the duration of a typical
    success, i.e. the cost-to-go of failing, since after a failure the whole task still has to be done. The
    hard floor is `max(success latch) - min(failure length)` (20 frames on the smoke data) - below it a fast
    failure outscores a slow success, which is early termination acting as a reward.

    `scale` plays the role of the paper's "normalize the values per task based on the maximum episode length".
    None means derive it from the dataset as `max(n_frames) + c_fail`, which is the largest magnitude any raw
    return can take, so nothing ever clips and it tracks `c_fail` automatically.
    """

    c_fail: float = 250.0
    scale: float | None = None


def episode_returns(n_frames: int, outcome: str, stacked_step: int | None, cfg: ReturnConfig) -> np.ndarray:
    """(n_frames,) normalised MC returns in [-1, 0].

    A success whose latch falls outside the recorded frames latches at `n_frames` instead - the stack exists by
    the end of the recording. This is the corrections: recording stops 30 frames after the release, `settled`
    needs ~15 frames of rest, and a scripted place's release_wait + retreat can run past the cut, so the stack
    never latches inside the written frames. Applying `c_fail` to them instead (as an earlier version did) is
    wrong by ~0.34 normalised and hands V targets that contradict the clean successes they look identical to -
    which it can only fit by memorising the layout. Assuming the latch at the last written frame is optimistic
    by the ~15 frames until `settled` completes, i.e. ~0.02 normalised.
    """
    if cfg.scale is None:
        raise ValueError("ReturnConfig.scale must be resolved first (load_rollout_data does this)")
    t = np.arange(n_frames, dtype=np.float32)
    if outcome == "success":
        latch = float(stacked_step if stacked_step is not None else n_frames)
        raw = -(latch - t)
    else:
        raw = -(float(n_frames) - t) - cfg.c_fail
    return np.clip(raw / cfg.scale, -1.0, 0.0).astype(np.float32)


def two_hot(values: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """(..., n_bins) two-hot targets for values in [-1, 0]. Bin 0 is -1.0, bin n_bins-1 is 0.0.

    Two-hot rather than one-hot: the return is continuous and almost never lands on a bin centre, so splitting
    the mass between the two neighbours by distance keeps the sub-bin information a hard assignment would throw
    away, and makes `expected_value` round-trip exactly.
    """
    x = (values.clamp(-1.0, 0.0) + 1.0) * (n_bins - 1)  # -> [0, n_bins-1]
    lo = x.floor().clamp(0, n_bins - 2).long()
    frac = (x - lo.float()).unsqueeze(-1)
    out = torch.zeros(*values.shape, n_bins, device=values.device, dtype=torch.float32)
    out.scatter_(-1, lo.unsqueeze(-1), 1.0 - frac)
    out.scatter_(-1, (lo + 1).unsqueeze(-1), frac)
    return out


def bin_centers(n_bins: int = N_BINS, device=None) -> torch.Tensor:
    return torch.linspace(-1.0, 0.0, n_bins, device=device)


def expected_value(logits: torch.Tensor) -> torch.Tensor:
    """The distribution's mean, which is what the advantage uses."""
    return (logits.softmax(-1) * bin_centers(logits.shape[-1], logits.device)).sum(-1)


def distributional_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-likelihood over the 201 bins: cross-entropy against two-hot targets (paper: section V-C)."""
    return -(two_hot(targets, logits.shape[-1]) * logits.log_softmax(-1)).sum(-1).mean()


# --------------------------------------------------------------------------------------------------- episodes


@dataclass
class EpisodeInfo:
    episode_index: int
    n_frames: int
    outcome: str
    stacked_step: int | None
    lifted_step: int | None
    transported_step: int | None
    release_step: int | None
    kind: str
    layout_id: int
    task: str
    source: str
    destination: str


def _opt_int(row: dict, key: str) -> int | None:
    v = row.get(key)
    return int(v) if v not in (None, "") else None


def read_log(path: str | Path) -> dict[int, EpisodeInfo]:
    """Episode-level facts from a collection log, keyed by episode_index.

    Handles `rollout_log.csv` (collect_rollouts.py: outcome + written-frame milestones) and
    `collection_log.csv` (collect_demos.py: success / n_frames only). Rows without an episode_index are
    corrections that were attempted but never entered the dataset, and are skipped.
    """
    out: dict[int, EpisodeInfo] = {}
    for r in csv.DictReader(Path(path).open()):
        if not r.get("episode_index"):
            continue
        ep = int(r["episode_index"])
        if "outcome" in r:  # rollout log
            outcome, n = r["outcome"], int(r["steps"])
            stacked, lifted = _opt_int(r, "stacked_step"), _opt_int(r, "lifted_step")
            transported, release = _opt_int(r, "transported_step"), _opt_int(r, "release_step")
        else:  # demo log: every kept episode is a success, but it carries no milestones
            outcome = "success" if str(r.get("success", "")).lower() == "true" else "failure"
            n, stacked, lifted, transported, release = int(r["n_frames"]), None, None, None, None
        out[ep] = EpisodeInfo(ep, n, outcome, stacked, lifted, transported, release, r.get("kind", "demo"),
                              int(r.get("layout_id", -1)), r.get("task", ""),
                              r.get("source", ""), r.get("destination", ""))
    return out


def build_features(poses: np.ndarray, state: np.ndarray, tcp: np.ndarray, grip_quat: np.ndarray,
                   src_i: int, dst_i: int, third_i: int,
                   src_is_cube: bool, dst_is_cube: bool) -> np.ndarray:
    """(N, FEATURE_DIM) features for one episode's frames.

    `poses` is the raw `sim.object_poses` column, (N, 21) = 3 objects x [x,y,z,qw,qx,qy,qz] in the fixed
    OBJECT_NAMES order, so role-ordering is just an index permutation. `tcp` / `grip_quat` come from
    `ee_pose` on the recorded joints.
    """
    obj = lambda i: poses[:, 7 * i:7 * i + 7]  # noqa: E731
    src, dst, third = obj(src_i), obj(dst_i), obj(third_i)
    flags = np.tile(np.array([float(src_is_cube), float(dst_is_cube)], dtype=np.float32), (len(poses), 1))
    return np.concatenate([src, dst, third, src[:, :3] - dst[:, :3],
                           tcp, grip_quat, src[:, :3] - tcp, state, flags], axis=1).astype(np.float32)


@dataclass
class RolloutData:
    features: np.ndarray             # (N, FEATURE_DIM)
    returns: np.ndarray              # (N,) normalised MC returns in [-1, 0]
    episode_of_frame: np.ndarray     # (N,) episode index per frame
    frame_of_episode: dict[int, tuple[int, int]]   # episode index -> [from, to) into the arrays above
    episodes: dict[int, EpisodeInfo]
    excluded: set[int]               # episode indices kept out of training but still scored
    cfg: ReturnConfig

    def trainable(self) -> dict[int, EpisodeInfo]:
        return {ep: i for ep, i in self.episodes.items() if ep not in self.excluded}


def load_rollout_data(root: str | Path, cfg: ReturnConfig, exclude_kinds: tuple[str, ...] = ()) -> RolloutData:
    """Read a collect_rollouts dataset into flat arrays, resolving `cfg.scale` from the episode lengths.

    `exclude_kinds` holds whole episodes out of *training* by their `kind`, but their features and returns are
    still built: V has to be evaluable on them, because the advantage of a correction's action is exactly what
    stage D needs and it is measured against the uncorrected policy's value.

    Excluding "correction" matters: a correction replays its parent's grasp and then succeeds, so the two
    episodes present near-identical states at the lift step with opposite returns. V can only fit that by
    averaging, which is why it cannot tell a bad grasp from a good one when both are present (measured:
    lift-step AUC 0.424 with corrections, 0.610 without). Keeping them is faithful to the paper - V of the
    data mixture, where a bad grasp is recoverable - while dropping them gives V of the uncorrected policy,
    which is the baseline a correction's advantage should be measured against.

    Uses `LeRobotDataset` for the metadata but bulk-reads `ds.hf_dataset` rather than `ds[i]`: the numeric
    columns come back in one shot (~1 s per 7.5k frames) while `__getitem__` decodes three videos per sample
    (~90 ms), which over 170k frames would be hours per epoch for pixels this V never looks at.
    """
    from lerobot.datasets import LeRobotDataset  # imported lazily: value.py is also used without a dataset

    root = Path(root)
    ds = LeRobotDataset("local/value", root=root, video_backend="pyav")
    hf = ds.hf_dataset
    # hf[col] is a datasets Column of per-row tensors; np.stack consumes it directly.
    poses = np.stack(hf["sim.object_poses"]).astype(np.float32)
    state = np.stack(hf["observation.state"]).astype(np.float32)
    qpos = np.stack(hf["sim.qpos"]).astype(np.float32)
    index = np.asarray(hf["index"])
    # The gripper pose is not recorded, so derive it from the joints rather than re-collecting. Verified
    # against Genesis to 1e-4 mm over the full joint ranges.
    tcp, _, grip_quat = ee_pose(qpos[:, :6])
    # Row i must be global frame i, because the episode ranges below index into this same order.
    assert (index == np.arange(len(index))).all(), "hf_dataset is not in global frame order"

    meta = ds.meta.episodes
    ranges = {int(e): (int(f), int(t)) for e, f, t in
              zip(meta["episode_index"], meta["dataset_from_index"], meta["dataset_to_index"], strict=True)}

    log_path = next((root / n for n in ("rollout_log.csv", "collection_log.csv") if (root / n).exists()), None)
    if log_path is None:
        raise SystemExit(f"{root} has neither rollout_log.csv nor collection_log.csv")
    episodes = read_log(log_path)
    # Held out of training, but still featurised and still scored by V (see the docstring).
    excluded = {ep for ep, i in episodes.items() if i.kind in exclude_kinds}
    if excluded:
        print(f"  {len(excluded)} episodes of kind {exclude_kinds} held out of training (still scored by V)")
    if cfg.scale is None:
        cfg.scale = float(max(i.n_frames for i in episodes.values())) + cfg.c_fail

    features = np.zeros((len(poses), FEATURE_DIM), dtype=np.float32)
    returns = np.zeros(len(poses), dtype=np.float32)
    episode_of_frame = np.full(len(poses), -1, dtype=np.int64)
    missing_latch = []

    for ep, info in sorted(episodes.items()):
        f, t = ranges[ep]
        # The written-frame bookkeeping is the one thing that would silently corrupt every target, so the
        # CSV's `steps` and the dataset's own episode length have to agree exactly.
        assert t - f == info.n_frames, \
            f"ep {ep}: dataset has {t - f} frames, {log_path.name} says {info.n_frames}"
        # The outcome in the log is authoritative - it is what analyse() decided, and corrections are only
        # kept when it says "success". stacked_step only says WHEN the -1/frame stops accruing, and it can be
        # absent (rollouts: written_index returns None past the recording cut) or out of range (corrections:
        # written untranslated, so it can exceed `steps`). Both mean the same thing - the stack was confirmed
        # after the recording ended - so both are normalised to None here rather than asserted on.
        stacked = info.stacked_step
        if stacked is not None and stacked >= info.n_frames:
            stacked = None
        if info.outcome == "success" and stacked is None:
            missing_latch.append(ep)

        src_i, dst_i = OBJECT_NAMES.index(info.source), OBJECT_NAMES.index(info.destination)
        third_i = next(i for i in range(len(OBJECT_NAMES)) if i not in (src_i, dst_i))
        features[f:t] = build_features(poses[f:t], state[f:t], tcp[f:t], grip_quat[f:t],
                                       src_i, dst_i, third_i,
                                       "cube" in info.source, "cube" in info.destination)
        returns[f:t] = episode_returns(info.n_frames, info.outcome, stacked, cfg)
        episode_of_frame[f:t] = ep

    if missing_latch:
        # Expected for corrections (see episode_returns): the latch falls past the recording cut.
        kinds = Counter(episodes[e].kind for e in missing_latch)
        print(f"  {len(missing_latch)} successes have no stacked_step -> latch assumed at the last written "
              f"frame  ({', '.join(f'{v} {k}' for k, v in kinds.most_common())})")
    unassigned = int((episode_of_frame < 0).sum())
    if unassigned:
        # Either excluded by `exclude_kinds` or missing from the log. Either way these frames are never
        # trained on (the splits are built from `episodes`), but they must not be read downstream either -
        # train_value.py writes NaN into values.npy wherever episode_of_frame is -1.
        print(f"  {unassigned} of {len(poses)} frames belong to episodes absent from rollout_log.csv "
              f"(correction attempts that never entered the dataset) and are not used")

    return RolloutData(features, returns, episode_of_frame,
                       {ep: ranges[ep] for ep in episodes}, episodes, excluded, cfg)


# ----------------------------------------------------------------------------------------------------- model


class ValueMLP(nn.Module):
    """V from simulator state -> 201 return-bin logits. ~380k params at hidden=512.

    Input standardisation is a buffer pair rather than a LayerNorm: LayerNorm normalises per sample *across*
    features, which would mix metres, quaternion components and radians into one meaningless mean and variance.
    The stats are fitted on the training split only and travel with the checkpoint, so stage D's labelling
    uses the identical transform.
    """

    def __init__(self, n_in: int = FEATURE_DIM, hidden: int = 512, dropout: float = 0.1,
                 n_bins: int = N_BINS, noise_std: float = 0.0):
        super().__init__()
        self.noise_std = noise_std
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_bins),
        )

    def set_normalization(self, mean: np.ndarray, std: np.ndarray, floor: float = 1e-4) -> None:
        """Fit from the training split. The floor keeps constant features (joint5 is constant in the demos,
        and the shape flags are constant within a pair) from exploding into huge standardised values."""
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(np.maximum(std, floor), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
        return self.net(z)
