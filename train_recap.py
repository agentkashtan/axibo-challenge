"""RECAP stage D: advantage-conditioned fine-tune (arXiv 2511.14759 section V-D).

A literal copy of `train_smolvla.py` with one addition: before the preprocessor tokenises the task, each
sample's language string gets the paper's improvement indicator appended -

    "put the red cube on the blue cube Advantage: positive"      I_t = True
    "put the red cube on the blue cube Advantage: negative"      I_t = False

verbatim from the paper ("inputting 'Advantage: positive' when I_t=True, and 'Advantage: negative' otherwise"),
placed after the task text and before action generation. 8 -> 12 tokens against SmolVLA's limit of 48, so
nothing truncates, and the architecture is untouched - the task string is already a tokenised input.

`I_t` comes from `label_advantage.py` and is looked up per frame by the batch's **global** `index` column
(verified to stay global when LeRobotDataset is subset by episode, which is what makes the lookup safe).

`--cond-dropout` defaults to 0: inference always asks for "Advantage: positive", so the unconditional model is
not needed. Raising it is the paper's alternative to tuning a loss multiplier - it leaves a usable
unconditional model behind so that classifier-free guidance at beta > 1 becomes possible without retraining.
Validation never drops the indicator, so val loss stays comparable across runs.

`--no-indicator` is the attribution control the challenge spec requires: identical data, identical schedule,
indicator stripped. Without that arm an improvement is not attributable to RECAP - it could just be extra
training on self-collected data.

    source .venv/bin/activate
    python train_recap.py --dataset data/lerobot/v322_data_for_recap \
        --checkpoint outputs/train/main_250v3/checkpoints/045000/pretrained_model \
        --labels outputs/value/nc_h64_dropout_only/advantage_N50_p30 \
        --steps 10000 --save-every 2000 --batch-size 32 --num-workers 8
"""

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs import NormalizationMode
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors

from axibo.policy import ACTION_KEY, CAMERA_KEYS, STATE_KEY, make_piperx_config
from axibo.recording import IMAGE_KEYS
from axibo.sim import SimConfig
from axibo.backend import torch_device
from axibo.value import ADVANTAGE_NEGATIVE as NEGATIVE
from axibo.value import ADVANTAGE_POSITIVE as POSITIVE
from axibo.value import read_log

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
    p.add_argument("--dataset", required=True,
                   help="the advantage-labelled dataset root (from collect_rollouts.py)")
    p.add_argument("--positive-dataset", default=None,
                   help="a second dataset trained alongside with every frame forced 'Advantage: positive' - "
                        "the scripted demos. Must sit beside --dataset under the same parent directory")
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
    p.add_argument("--labels", required=True,
                   help="a label_advantage.py output dir (labels.npz + thresholds.json)")
    p.add_argument("--cond-dropout", type=float, default=0.0,
                   help="probability of omitting the indicator. 0 = always condition, which is what inference "
                        "does (always 'Advantage: positive'). Raise it only to leave an unconditional model "
                        "behind for classifier-free guidance at beta > 1")
    p.add_argument("--no-indicator", action="store_true",
                   help="attribution control: same data and schedule, no conditioning text at all")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def split_episodes(dataset_root: Path, val_fraction: float, seed: int):
    """Split by layout so validation scenes never appear in training.

    Reads `rollout_log.csv` rather than `collection_log.csv`, and keeps **every** episode: the failures carry
    the negative-advantage frames that the conditioning is built on, so there is no `--include-failed` here.
    A correction shares its parent failure's layout, so a layout split also keeps that pair on one side.
    """
    log = next((dataset_root / n for n in ("rollout_log.csv", "collection_log.csv")
                if (dataset_root / n).exists()), None)
    if log is None:
        raise SystemExit(f"{dataset_root} has neither rollout_log.csv nor collection_log.csv")
    episodes = read_log(log)
    by_layout = defaultdict(list)
    for ep, info in episodes.items():
        by_layout[info.layout_id].append(ep)
    layouts = sorted(by_layout)
    rng = np.random.default_rng(seed)
    n_val = max(1, math.ceil(val_fraction * len(layouts)))
    val_layouts = set(rng.choice(layouts, size=n_val, replace=False).tolist())
    train = [e for l in layouts if l not in val_layouts for e in by_layout[l]]
    val = [e for l in sorted(val_layouts) for e in by_layout[l]]
    kinds = Counter(i.kind for i in episodes.values())
    print(f"  {dataset_root.name}: {len(episodes)} episodes "
          f"({', '.join(f'{v} {k}' for k, v in kinds.most_common())}), "
          f"{len(layouts) - n_val} train / {n_val} val layouts")
    return sorted(train), sorted(val)


def condition(batch: dict, indicators: list, dropout: float, rng, enabled: bool) -> dict:
    """Append the improvement indicator to each sample's task string, in place.

    `indicators[d]` is the per-frame array for dataset `d`, or None meaning "forced positive" (the scripted
    demos). A sample is located by (`dataset_index`, `index`): each sub-dataset numbers its own frames from 0,
    so one flat lookup would read demo frames against rollout labels. With `dropout > 0` some samples keep the
    bare task, which is the paper's unconditional loss term.
    """
    if not enabled:
        return batch
    tasks, idx = list(batch["task"]), batch["index"].numpy()
    dsi = batch["dataset_index"].numpy() if "dataset_index" in batch else np.zeros(len(tasks), dtype=int)
    keep = np.ones(len(tasks), dtype=bool) if dropout <= 0 else rng.random(len(tasks)) >= dropout
    out = []
    for task, i, d, k in zip(tasks, idx, dsi, keep, strict=True):
        if not k:
            out.append(task)
            continue
        arr = indicators[d]
        out.append(f"{task} {POSITIVE if arr is None or arr[i] else NEGATIVE}")
    batch["task"] = out
    return batch


def normalization_stats(all_stats: dict, std_floor: float) -> dict[str, dict[str, torch.Tensor]]:
    """State/action stats, with std clamped so constant dimensions don't blow up."""
    stats = {}
    for key in (STATE_KEY, ACTION_KEY):
        s = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in all_stats[key].items()}
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
def validate(policy, preprocessor, val_loader, n_batches: int, cfg, device, indicators, enabled) -> float:
    """Mean flow-matching loss on validation batches. Noise and noise levels come from a fixed seed (noise levels from
    the training distribution, Beta(1.5, 1) * 0.999 + 0.001) so values are comparable between checkpoints."""
    policy.eval()
    total, gen = 0.0, torch.Generator().manual_seed(1234)
    for i, batch in enumerate(tqdm(val_loader, total=n_batches, desc="val", leave=False)):
        if i == n_batches:
            break
        # dropout 0: validation always sees the true indicator, so the loss is comparable across runs
        batch = condition(batch, indicators, 0.0, None, enabled)
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

    labels = Path(args.labels)
    thresholds = json.loads((labels / "thresholds.json").read_text())
    use_indicator = not args.no_indicator
    print(f"labels {labels}  N={thresholds['horizon']} percentile={thresholds['percentile']} "
          f"positive_rate={100 * thresholds['positive_rate']:.1f}%  from {thresholds['value_dir']}")
    print(f"conditioning: {'OFF (attribution control)' if args.no_indicator else f'ON, dropout {args.cond_dropout}'}")
    cond_rng = np.random.default_rng(args.seed)

    # Dataset 0 is the advantage-labelled one; dataset 1, if given, is forced positive (indicator None).
    # MultiLeRobotDataset takes a parent directory plus directory names, so both must sit side by side.
    roots = [root]
    if args.positive_dataset:
        pos_root = Path(args.positive_dataset)
        if pos_root.parent != root.parent:
            raise SystemExit(f"--positive-dataset must sit beside --dataset ({root.parent})")
        roots.append(pos_root)
    repo_ids = [r.name for r in roots]
    indicators = [np.load(labels / "labels.npz")["indicator"]] + [None] * (len(roots) - 1)

    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    assert len(indicators[0]) == meta.total_frames, \
        f"labels cover {len(indicators[0])} frames but {root.name} has {meta.total_frames}"
    delta_timestamps = resolve_delta_timestamps(cfg, meta)  # 50-step action chunk per sample

    splits = {r.name: split_episodes(r, args.val_fraction, args.seed) for r in roots}
    make_ds = lambda which: MultiLeRobotDataset(  # noqa: E731
        repo_ids, root=root.parent,
        episodes={name: splits[name][which] for name in repo_ids},
        delta_timestamps=delta_timestamps, video_backend="pyav",
    )
    train_ds, val_ds = make_ds(0), make_ds(1)
    if train_ds.disabled_features:
        raise SystemExit(f"datasets disagree on features: {train_ds.disabled_features}")
    for name, r in zip(repo_ids, roots, strict=True):
        tag = "forced positive" if r != root else f"labelled ({labels.name})"
        print(f"  {name}: {tag}")
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kw)
    val_batches = min(len(val_loader), args.val_batches or len(val_loader))
    passes = lambda step: step * args.batch_size / len(train_ds)  # noqa: E731  (how many times the data was seen)

    # Aggregated across both datasets: the policy trains from the pre-trained checkpoint on the mixture, so
    # the normalization has to describe everything it sees, not just the labelled half.
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        cfg, normalization_stats(train_ds.stats, args.std_floor))
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
        f"\n  data points  : {len(train_ds) + len(val_ds)} total = {len(train_ds)} train "
        f"({train_ds.num_episodes} episodes) + {len(val_ds)} val ({val_ds.num_episodes} episodes), "
        f"each = 1 frame + {cfg.chunk_size}-step action chunk"
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
        raw = condition(next(batches), indicators, args.cond_dropout, cond_rng, use_indicator)
        batch = preprocessor({RENAME.get(k, k): v for k, v in raw.items()})
        loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, optim_cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        window_loss.append(loss.item())
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}", passes=f"{passes(step):.2f}")

        if step % args.save_every == 0 or step == args.steps:
            val_loss = validate(policy, preprocessor, val_loader, val_batches, cfg, device,
                                indicators, use_indicator)
            path = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
            save_checkpoint(path, policy, preprocessor, postprocessor)
            tqdm.write(f"step {step} | train loss {np.mean(window_loss):.4f} | val loss {val_loss:.4f} | "
                       f"lr {scheduler.get_last_lr()[0]:.1e} | passes {passes(step):.2f} | saved {path}")
            window_loss.clear()
            policy.train()


if __name__ == "__main__":
    main()
