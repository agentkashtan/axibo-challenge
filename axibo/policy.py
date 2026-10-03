"""SmolVLA wrapper: our Observation (3 cameras + 7-dim state) + task string -> (chunk, 7) joint-space actions."""

import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from axibo.backend import torch_device
from axibo.sim import ACTION_DIM, STATE_DIM, Observation

# SmolVLA's pretraining convention (paper): camera1 = top, camera2 = wrist, camera3 = side.
CAMERA_KEYS = {
    "top": "observation.images.camera1",
    "wrist": "observation.images.camera2",
    "side": "observation.images.camera3",
}
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def make_piperx_config(
    checkpoint: str, device: str, cam_res: tuple[int, int], state_action_norm: NormalizationMode
) -> PreTrainedConfig:
    """Checkpoint config with PiperX features: 7-dim state/action, three cameras in SmolVLA's camera1/2/3 slots."""
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    width, height = cam_res
    cfg.input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width)) for key in CAMERA_KEYS.values()},
    }
    cfg.output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": state_action_norm,
        "ACTION": state_action_norm,
    }
    return cfg


def saved_processors(checkpoint: str, device: str):
    """(preprocessor, postprocessor) saved inside a local checkpoint dir by train_smolvla.py, or None.

    The saved config pins the training device, so it is overridden with the one we run on.
    """
    path = Path(checkpoint)
    if not (path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").is_file():
        return None
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post


class SmolVLARunner:
    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        checkpoint: str = "lerobot/smolvla_base",
        device: str | None = None,
        cam_res: tuple[int, int] = (512, 512),
    ):
        """
        A fine-tuned checkpoint carries the processors train_smolvla.py saved next to it (MEAN_STD stats from the
        dataset); they are loaded so inference normalizes exactly as training did. Without them (e.g. smolvla_base)
        we fall back to MIN_MAX over the joint limits given by action_low / action_high — a plumbing placeholder,
        since the checkpoint's own stats are 6-dim SO100 and don't apply.
        """
        device = device or torch_device()  # resolved once: it is also written into the config and the processors
        self.device = torch.device(device)
        saved = saved_processors(checkpoint, device)

        self.cfg = cfg = make_piperx_config(
            checkpoint, device, cam_res, NormalizationMode.MEAN_STD if saved else NormalizationMode.MIN_MAX
        )

        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg)
        self.policy.eval()

        if saved:
            self.preprocessor, self.postprocessor = saved
        else:
            limits = {
                "min": torch.as_tensor(action_low, dtype=torch.float32),
                "max": torch.as_tensor(action_high, dtype=torch.float32),
            }
            self.preprocessor, self.postprocessor = make_smolvla_pre_post_processors(
                cfg, dataset_stats={STATE_KEY: limits, ACTION_KEY: limits}
            )
        self.latencies_s: list[float] = []

    @property
    def chunk_size(self) -> int:
        return self.cfg.chunk_size

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def predict_chunk(self, obs: Observation, task: str) -> np.ndarray:
        """obs: single-env Observation (obs.env(i)). Returns (chunk_size, 7) absolute joint targets; latency is logged."""
        t0 = time.perf_counter()
        raw = {STATE_KEY: obs.state.astype(np.float32)}
        raw.update({key: obs.images[name] for name, key in CAMERA_KEYS.items()})

        batch = prepare_observation_for_inference(raw, self.device, task=task)
        batch = self.preprocessor(batch)
        actions = self.policy.predict_action_chunk(batch)  # (1, chunk, 7), normalized
        actions = self.postprocessor(actions)  # unnormalized, on CPU (implicit device sync)

        self.latencies_s.append(time.perf_counter() - t0)
        return actions[0].numpy()
