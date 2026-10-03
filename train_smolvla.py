"""Fine-tune SmolVLA on a collected PiperX LeRobot dataset with a plain step-based PyTorch training loop.

Per batch: DataLoader batch (raw frames + 50-step action chunk) -> rename cameras to SmolVLA's slots
-> preprocessor (tokenize task, move to device, normalize) -> policy.forward -> flow-matching loss -> AdamW step.
The DataLoader is cycled (reshuffled each pass) for --steps steps, like lerobot-train. Every --save-every steps a
checkpoint is written and a validation loss on held-out layouts is logged.

    source .venv/bin/activate
    python train_smolvla.py --dataset data/lerobot/pilot_50 --steps 20000 --save-every 2000
"""

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs import NormalizationMode
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors

from axibo.policy import ACTION_KEY, CAMERA_KEYS, STATE_KEY, make_piperx_config
from axibo.recording import IMAGE_KEYS
from axibo.sim import SimConfig
from axibo.backend import torch_device

# SmolVLA paper (arXiv 2506.01844): AdamW (b1=0.9, b2=0.95), 100-step warmup, cosine decay from 1e-4 to 2.5e-6,
# only the action expert trained (VLM frozen).
PAPER_LR = 1e-4
PAPER_DECAY_LR = 2.5e-6
PAPER_WARMUP_STEPS = 100
PAPER_BETAS = (0.9, 0.95)

# Dataset camera names -> SmolVLA's pretrained camera slots.
RENAME = {IMAGE_KEYS[name]: CAMERA_KEYS[name] for name in CAMERA_KEYS}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="LeRobot dataset root (from collect_demos.py)")
    p.add_argument("--repo-id", default="local/piperx_stack")
    p.add_argument("--checkpoint", default="lerobot/smolvla_base")
    p.add_argument("--out", default=None, help="default: outputs/train/<dataset name>")
    p.add_argument("--steps", type=int, default=20000, help="total optimizer steps (1 step = 1 batch)")
    p.add_argument("--save-every", type=int, default=2000, help="checkpoint + validation every N steps (and at the end)")
    p.add_argument("--val-batches", type=int, default=50, help="validation batches per checkpoint (0 = full val set)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=PAPER_LR, help="peak learning rate (SmolVLA paper: 1e-4)")
    p.add_argument("--val-fraction", type=float, default=0.1, help="fraction of layouts held out for validation")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=torch_device())
    p.add_argument("--std-floor", type=float, default=0.01,
                   help="minimum std for state/action normalization (joint5 is constant in the demos: std=0)")
    p.add_argument("--include-failed", action="store_true", help="also use episodes with success=False")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def split_episodes(dataset_root: Path, val_fraction: float, include_failed: bool, seed: int):
    """Split episodes by layout so validation scenes never appear in training. Falls back to an episode split if
    the dataset has a single layout."""
    by_layout = defaultdict(list)
    with (dataset_root / "collection_log.csv").open() as f:
        for r in csv.DictReader(f):
            if r["episode_index"] != "-1" and (include_failed or r["success"] == "True"):
                by_layout[int(r["layout_id"])].append(int(r["episode_index"]))
    layouts = sorted(by_layout)
    rng = np.random.default_rng(seed)
    if len(layouts) >= 2:
        n_val = max(1, math.ceil(val_fraction * len(layouts)))
        val_layouts = set(rng.choice(layouts, size=n_val, replace=False).tolist())
        train = [e for l in layouts if l not in val_layouts for e in by_layout[l]]
        val = [e for l in sorted(val_layouts) for e in by_layout[l]]
    else:
        episodes = by_layout[layouts[0]]
        n_val = max(1, math.ceil(val_fraction * len(episodes)))
        tqdm.write(f"only one layout: validating on {n_val} held-out episodes of the same scene (not a scene split)")
        val, train = episodes[-n_val:], episodes[:-n_val]
    return sorted(train), sorted(val)


def normalization_stats(meta: LeRobotDatasetMetadata, std_floor: float) -> dict[str, dict[str, torch.Tensor]]:
    """State/action stats from the dataset, with std clamped so constant dimensions don't blow up."""
    stats = {}
    for key in (STATE_KEY, ACTION_KEY):
        s = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in meta.stats[key].items()}
        low = s["std"] < std_floor
        if low.any():
            print(f"{key}: std floored to {std_floor} for dims {low.nonzero().flatten().tolist()}")
        s["std"] = s["std"].clamp_min(std_floor)
        stats[key] = s
    return stats


def save_checkpoint(path: Path, policy, preprocessor, postprocessor):
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)


@torch.no_grad()
def validate(policy, preprocessor, val_loader, n_batches: int, cfg, device) -> float:
    """Mean flow-matching loss on validation batches. Noise and noise levels come from a fixed seed (noise levels from
    the training distribution, Beta(1.5, 1) * 0.999 + 0.001) so values are comparable between checkpoints."""
    policy.eval()
    total, gen = 0.0, torch.Generator().manual_seed(1234)
    for i, batch in enumerate(tqdm(val_loader, total=n_batches, desc="val", leave=False)):
        if i == n_batches:
            break
        batch = preprocessor({RENAME.get(k, k): v for k, v in batch.items()})
        bsize = batch[ACTION_KEY].shape[0]
        noise = torch.randn((bsize, cfg.chunk_size, cfg.max_action_dim), generator=gen).to(device)
        time_level = (torch.rand(bsize, generator=gen) ** (1 / 1.5) * 0.999 + 0.001).to(device)  # inverse CDF
        loss, _ = policy.forward(batch, noise=noise, time=time_level)
        total += loss.item()
    return total / max(1, n_batches)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    root = Path(args.dataset)
    out_dir = Path(args.out or f"outputs/train/{root.name}")
    device = torch.device(args.device)

    # Policy config with PiperX features; pretrained SmolVLA uses MEAN_STD for state/action.
    cfg = make_piperx_config(args.checkpoint, args.device, SimConfig().cam_res, NormalizationMode.MEAN_STD)

    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    train_eps, val_eps = split_episodes(root, args.val_fraction, args.include_failed, args.seed)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)  # 50-step action chunk per sample
    make_ds = lambda eps: LeRobotDataset(  # noqa: E731
        args.repo_id, root=root, episodes=eps, delta_timestamps=delta_timestamps, video_backend="pyav"
    )
    train_ds, val_ds = make_ds(train_eps), make_ds(val_eps)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kw)
    val_batches = min(len(val_loader), args.val_batches or len(val_loader))
    passes = lambda step: step * args.batch_size / len(train_ds)  # noqa: E731  (how many times the data was seen)

    preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, normalization_stats(meta, args.std_floor))
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=cfg)

    params = [p for p in policy.parameters() if p.requires_grad]  # vision encoder / VLM frozen by default
    cfg.optimizer_lr, cfg.optimizer_betas = args.lr, PAPER_BETAS
    cfg.scheduler_warmup_steps = min(PAPER_WARMUP_STEPS, max(1, args.steps // 10))
    cfg.scheduler_decay_steps, cfg.scheduler_decay_lr = args.steps, PAPER_DECAY_LR
    optim_cfg = cfg.get_optimizer_preset()
    optimizer = optim_cfg.build(params)
    scheduler = cfg.get_scheduler_preset().build(optimizer, num_training_steps=args.steps)
    steps_per_pass = len(train_ds) // args.batch_size
    print(
        f"\n  data points  : {len(train_ds) + len(val_ds)} total = {len(train_ds)} train ({len(train_eps)} episodes) "
        f"+ {len(val_ds)} val ({len(val_eps)} episodes), each = 1 frame + {cfg.chunk_size}-step action chunk"
        f"\n  steps        : {args.steps} total, batch {args.batch_size} -> {steps_per_pass} steps per pass over train "
        f"data = {passes(args.steps):.2f} passes"
        f"\n  checkpoints  : every {args.save_every} steps (+ final) -> {out_dir / 'checkpoints'}, "
        f"val on {val_batches} batches each"
        f"\n  model        : {sum(p.numel() for p in params) / 1e6:.1f}M / {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M "
        f"params trainable | lr {optim_cfg.lr}, betas {optim_cfg.betas}, warmup {cfg.scheduler_warmup_steps} "
        f"-> cosine to {PAPER_DECAY_LR} | device {args.device}\n"
    )

    batches = itertools.chain.from_iterable(itertools.repeat(train_loader))  # reshuffles on every pass
    policy.train()
    window_loss = []
    bar = tqdm(range(1, args.steps + 1), desc="train", unit="step", dynamic_ncols=True)
    for step in bar:
        batch = preprocessor({RENAME.get(k, k): v for k, v in next(batches).items()})
        loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, optim_cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        window_loss.append(loss.item())
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}", passes=f"{passes(step):.2f}")

        if step % args.save_every == 0 or step == args.steps:
            val_loss = validate(policy, preprocessor, val_loader, val_batches, cfg, device)
            path = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
            save_checkpoint(path, policy, preprocessor, postprocessor)
            tqdm.write(f"step {step} | train loss {np.mean(window_loss):.4f} | val loss {val_loss:.4f} | "
                       f"lr {scheduler.get_last_lr()[0]:.1e} | passes {passes(step):.2f} | saved {path}")
            window_loss.clear()
            policy.train()


if __name__ == "__main__":
    main()
