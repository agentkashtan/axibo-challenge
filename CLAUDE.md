# CLAUDE.md

## What this repo is

My submission for the **AXIBO VLA Challenge** (full spec in `README.md` — read it before helping). In short:
build a Vision-Language-Action pipeline in **Genesis** on the **AgileX PiperX** arm (6-DOF + gripper) for
language-conditioned stacking ("put the {source} on the {destination}") with a red cube, red cylinder, and blue cube.

| Task | What I must deliver |
|---|---|
| 1. Data engine | Scripted (IK-based), batched, randomized demo collection over 5 of 6 ordered pairs (1 held out), exported to LeRobot or a justified format. Every design decision documented with reasoning. |
| 2. Fine-tune + eval | Fine-tune an open VLA checkpoint. Per-pair success with stated trial count, seed list, failure taxonomy. Separate reversal test. Held-out-pair result. |
| 3. Smooth inference | Remove chunk-boundary stalls (async / temporal ensembling / RTC / compile / quantization). p50/p99 latency before/after, a jerk or velocity-discontinuity metric, proof success didn't drop. |
| 4. RECAP (optional, recommended) | Rollouts → scripted corrections on failure → value function → advantage-conditioned fine-tune. Honest result, attributable, reward-hacking checks. |

Submission: private GitHub repo (code, weights or pointer, technical report, videos incl. back-to-back reversal test).
Timeline: ~1 week for Tasks 1–3, +1 week for Task 4.

**Current progress, decisions, artifacts and next steps: see `STATUS.md` (handoff snapshot).**

## How to work with me — READ THIS FIRST

**This is my project and my learning. Your role is mentor / pair-programming reviewer, not the author.**
The evaluators grade *my* reasoning (action-space choice, randomization, eval protocol, reading RECAP into code),
so the thinking has to be mine.

### Do
- **Explain concepts** when I ask (Genesis APIs, IK, action chunking, flow matching, RTC, advantage conditioning, etc.), with pointers to docs, source code, or papers so I can verify.
- **Ask me questions** that expose gaps before I commit to a design ("what happens to your gripper action at the chunk boundary?", "how would that reward score a knocked-over destination?").
- **Review my code and designs** critically: bugs, wrong assumptions, eval leaks, reward hacks, unreproducible numbers. Be direct; don't flatter.
- **Help debug**: help me form hypotheses and find the root cause; explain *why* something breaks.
- **Point me to the right place** — which Genesis example, which LeRobot module, which section of a paper — rather than pasting a finished solution.
- **Offer options with trade-offs** when I'm choosing (e.g. joint-space vs. EE-delta actions, which checkpoint), then let me decide.
- Small illustrative snippets (a few lines showing an API call or pattern) are fine.
- Boilerplate / tedious glue (plotting, arg parsing, file I/O, env setup scripts) is fine to write if I ask explicitly.

### Don't
- Don't write whole modules, pipelines, training loops, or the eval harness unless I explicitly say "write it".
- Don't make design decisions for me silently (held-out pair, action space, tolerances, randomization ranges, reward definition). Surface the choice and ask.
- Don't write the technical report's reasoning sections for me. You can review drafts and point out what's missing or unsupported.
- Don't invent results, numbers, or claims about what an API does — check the source or say you're unsure.

If I ask for something that crosses the line, it's fine to do it — I'm allowed to change my mind — but briefly note it so it's my deliberate choice.

## Key constraints from the spec (keep me honest)
- Collection **and** evaluation must be native to Genesis on the provided PiperX asset.
- Held-out pair must be **entirely** absent from training data (check for leaks, including in Task 4 rollouts/corrections — decide and state whether the held-out pair is excluded there too).
- Success = source resting on destination, stable **2 s after the arm retreats**, destination upright, third object undisturbed. Tolerances must be stated explicitly.
- Every reported number needs: trial count, seed list, failure taxonomy. Aggregates alone are not enough.
- Reversal test: same fixed scene, two opposite instructions, reported separately.
- Task 3 improvements must not break the policy — success rate after ≥ before (under the same protocol).
- Task 4: reward must not be satisfiable by knocking the destination over or leaving the source where it was. A negative result with a clear explanation is acceptable; an unattributed improvement is not.

## Things worth prodding me about
- Genesis version / API drift vs. examples (e.g. `batched_IK.py`); scene build with `n_envs` for parallel collection.
- PiperX URDF quirks: gripper joints (often two prismatic fingers, possibly mimic), joint limits, EE link choice for IK.
- Grasping cylinders vs. cubes, yaw symmetry of cubes, stacking a cube on a cylinder (and vice versa) — physical feasibility of each of the 6 pairs.
- Scripted-policy failures silently entering the dataset (filter by success check?).
- Observation setup: camera count/placement, resolution, proprio; matching the chosen checkpoint's expected inputs.
- Action space vs. what the checkpoint was pretrained on; normalization stats; control frequency vs. chunk size.
- Eval determinism: seeding Genesis + policy sampling; separate eval seeds from training seeds.
- Latency measurement methodology (warmup, GPU sync, what's included in p50/p99).
- Smoothness metric computed specifically at chunk boundaries, not just averaged over the whole trajectory.
- RECAP: what the "load-bearing mechanism" is (value function → binarized advantage → conditioning token/text, high-advantage at inference, classifier-free-guidance-like behavior).
- Time tracking: the report must say how time was spent and what's next if time ran out.

## Local context
- Related prior work in sibling dirs (reference only, don't modify): `../lerobot` (LeRobot source checkout), `../imitation-learing` (my earlier record/train/run-policy imitation learning project).
- Repo is currently empty apart from `README.md` (the challenge spec). Update this file's structure/commands section as the project takes shape.

## Project structure & commands
**Rule: everything is installed inside this directory** — no global pip installs, no caches in `~`.

Environment (macOS, Apple M4, arm64):
- Python 3.13.5 venv at `.venv/` (created from `~/miniconda3/bin/python3.13`; Homebrew's 3.14 is unsupported by Genesis, which needs `>=3.10,<3.14`).
- `torch` 2.11.0 (MPS; pinned <2.12 by lerobot 0.6.1), `genesis-world` 1.4.1 — use `gs.init(backend=gs.metal)`, `lerobot[smolvla]` 0.6.1 (transformers 5.5). Pins in `requirements.txt`.
- `.venv/bin/activate` has extra exports appended so caches stay in the project (gitignored `.cache/`):
  `GS_CACHE_FILE_PATH=$ROOT/.cache/genesis`, `QD_OFFLINE_CACHE_FILE_PATH=$ROOT/.cache/quadrants`, `HF_HOME=$ROOT/.cache/huggingface` (SmolVLA weights, ~2.7 GB).
  Always `source .venv/bin/activate` first; calling `.venv/bin/python` directly skips these and writes to `~/.cache`.

Recreate from scratch:
```bash
/Users/agentkashtan/miniconda3/bin/python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
# then re-append the three cache exports above to .venv/bin/activate
```

Robot asset: `assets/piper_x_description/` (unmodified copy, provenance in `assets/SOURCE.txt`).
Main file `urdf/piper_x_description.urdf` — 8 DOFs: `joint1–6` revolute, `joint7` prismatic [0, 0.05], `joint8` prismatic [-0.05, 0] (two independent fingers, **no `<mimic>`**). Links `base_link, Link1…Link8`; no dedicated TCP/EE link. Known quirks seen when loading in Genesis 1.4.1 (open decisions for me, don't silently fix):
- `package://` mesh paths + `base_Link.dae` vs. file `base_link.dae` (case) → primary parser fails, Genesis falls back to legacy URDF parser. Case mismatch only works on case-insensitive filesystems (macOS), breaks on Linux.
- `Link6` mass 0.006 kg and `Link7` COM copied from `Link8` (outside the finger) in the vendor files — **fixed only in `assets/urdf/piper_x_d435.urdf`** (Link6 → 0.20 kg, Link7 COM mirrored; reasoning in `assets/SOURCE.txt`). The plain vendor URDF still has them (Link7 there is massless). Non-watertight meshes; one self-collision pair filtered at qpos0.
- On macOS the viewer runs in the main thread (no `run_in_thread`); it's only responsive while `scene.step()` is being called.

Wrist-camera variant: `assets/urdf/piper_x_d435.urdf` — patched copy of `piper_x_description_d435.urdf` (package:// → relative paths, base_Link case fix; sed command in `assets/SOURCE.txt`). Needs `assets/realsense2_description/meshes/d435.dae` and `assets/piper_description/meshes/dae/realsense_mid_stand.dae` (copied from same upstream commit). Camera + stand are fixed links merged into `Link6`. The URDF mesh itself doesn't render images — use `scene.add_camera` + `cam.attach(Link6, offset_T)` + `cam.move_to_attach()` each render. Genesis cameras are OpenGL-convention (look −z, y up); ROS camera links are x-forward/z-up, so the offset includes that rotation (see `scene_cameras.py`). Default `near=0.1` clips a wrist view — use ~0.01.

Frames at zero qpos: robot faces +x, base at origin, gripper ≈ (0.23, 0, 0.23) pointing +x.

Also `Link8` uses `assets/urdf/meshes/J8_clean.obj` — vendor `J8.dae` had 2 stray triangles that inflated its collision hull to the floor (see `assets/SOURCE.txt`).

Scripts (backend/device come from `axibo/backend.py`: CUDA if torch sees it, else Metal on Apple arm64, else CPU;
override with `AXIBO_BACKEND` / `AXIBO_DEVICE`. Nothing hardcodes metal/mps any more, so files copy between machines
unchanged):
- `axibo/backend.py` — `gs_backend()`, `torch_device()`.
- `axibo/sim.py` — `PiperStackingScene` (+ `SimConfig`): scene, 3 cameras (top/wrist/side, 512×512), `observe()`,
  `apply_action(a7)`, `step_control()` (dt=1/90, 30 Hz), `qpos()`, `qvel()`, `object_pose/vel()`, `reseed(seed)`.
  **7-dim state/action**: joint1–6 [rad] + gripper opening w [m], fingers = [+w, −w]. Key settings and why:
  `enable_torsional_friction=True` (without it a side-grasped cylinder is a free hinge and tips during the lift),
  `noslip_iterations=5`, `finger_kp=1000`, `ambient_light=0.45`, top/side cameras reframed on the workspace
  (fovs are the smallest that keep every object in frame anywhere, verified by projection), gripper open at
  `home_qpos`, `start_jitter_rad=0.05`.
- `axibo/scene_builder.py` + `sample_layouts.py` — `Workspace` (x∈[0.18,0.34], y∈[−0.15,0.15], **≥10 cm apart**,
  cube yaw [0,90), cylinder yaw fixed). `load_layouts_csv`; `PiperStackingScene.reset(object_poses=…)`.
- `axibo/scripted.py` — `plan_demo(...) -> DemoPlan`, `execute_plan(...)`. Yaw is a function of the object poses
  only, with `CUBE_GRASP_YAW_OFFSET_DEG=180` / `SYMMETRIC_GRASP_YAW_DEG=-90` keeping joint6 inside [−38°, +118°],
  away from its ±179.9° limit (at the limit one degree of cube rotation flipped joint6 by 359°).
  `time_scale=0.5` → ~8.4 s demos; `spline_max_joint_speed=1.2` so the arm tracks its own target.
- `axibo/success.py` — final-state stack check used by collection (placeholder tolerances).
- `axibo/outcome.py` — rollout outcome detection for evaluation: per-step conditions C1–C12, five latched
  milestones (lifted / transported / stacked / released / placed) and 10 terminal labels. `settled` is net
  displacement over 0.5 s. Success is unchanged (footprint + stack height ±5 mm + settled + open, at the final
  frame); the **release of a transported object** is what starts the 4 s settle window, and the geometry at that
  frame attributes a failure in three tests: never `transported` (C1a, footprint grown 20 mm) →
  `dropped_in_transport`; height off stack_z
  > 10 mm → `released_in_air`; else source centre inside the footprint *shrunk* by `stable_frac` (C1b, 10 mm)
  → `knocked_off`, outside → `misplaced`. The three footprint tests share one `_in_footprint(f, half_width)`
  helper, so they cannot drift apart. `test_outcome.py` forces every label from a synthetic trace.
- `axibo/rollout.py` — queue-based closed-loop rollout with simulated inference latency (Task 3).
  `queue_threshold` k: 0 = naive sync (the arm holds while computing), k ≥ L = async. Chunks are consumed from
  index L (time alignment).
- `axibo/smoothness.py` — stall fraction, velocity discontinuity, jerk (whole-episode and boundary-windowed).
- `axibo/report.py` — shared summary tables (per pair, per layout, outcomes, flags, latency).
- `axibo/policy.py` — `SmolVLARunner`; loads the **checkpoint's own saved processors** (MEAN_STD dataset stats)
  when present, falling back to MIN_MAX joint limits only for the base model.
- `collect_demos.py` — batched collection → LeRobot dataset + `collection_log.csv`.
  `--layouts CSV --n-envs N [--num-layouts K] [--out DIR] [--viewer] [--no-record]`.
  **Held-out pair: "put the red cylinder on the red cube"**.
- `train_smolvla.py` — step-based fine-tuning; `--steps --save-every --batch-size --num-workers --val-fraction`.
- `eval_policy.py` — Task 2 eval: `--checkpoint --layouts --name --pairs train|held-out|all|reversal --seeds
  --max-chunks --exec-steps --settle-s [--live-cameras]` → `outputs/eval/<name>/{eval_log.csv,summary.json}`.
- `eval_async.py` — Task 3 eval: same interface plus `--queue-threshold k --latency fixed|measured --latency-ms
  --max-steps --save-traces [N]`.
- `eval_report.py <run> [<run> …] [--failures]` — re-print/compare finished runs by name.
- `replay_trace.py <run> [--trial N] [--speed S] [--live-cameras] [--save-video]` — re-execute a saved episode.
- `plot_trace.py`, `plot_motion_dist.py` — joint traces with stall bands; speed/acceleration/command-step
  distributions for one or more runs.
- `test_outcome.py` — synthetic-trace check that every outcome label is reachable and correct. No sim, no
  checkpoint: `PYTHONPATH=. python test_outcome.py`.
- `run_scripted_demo.py` (`--live-cameras` shows what the policy sees), `run_smolvla.py` (`--layouts --layout-id
  --source --destination`), `test_scene.py`, `test.py`, `scene_cameras.py` (scratch).

Measured (M4/MPS unless noted): collection 149/200/230 jobs/h at 1/5/10 envs; SmolVLA inference ~650 ms/chunk p50
locally, ~150–200 ms on the 4090; training ~5 s/step at batch 8 on MPS. Read datasets with `video_backend="pyav"`.
**joint5 is always 0 in these demos (std 0)** → `--std-floor` for MEAN_STD normalization.

_TBD — collection, training, eval, inference commands as code is added._
