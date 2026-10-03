# STATUS — handoff notes (last updated 2026-09-30)

Read `CLAUDE.md` first (collaboration rules, environment, file-by-file notes). This file is the *where we are / what's next* snapshot.

## Collaboration notes
- User writes the design; asks the agent to build pieces explicitly ("write a script…"). When the user **asks a question, answer it — don't run scripts/benchmarks** to answer.
- Don't launch long production runs (collection, training, full evals) unless asked — give the command instead. Agent once started the full 1250-episode collection when only a smoke test was wanted.
- Concise answers; explain trade-offs, then let the user decide.

## Progress vs. README tasks
| Task | State |
|---|---|
| 1. Data engine | **Done.** `main_250v3` = 1250 episodes, 1243 successful. Report write-up not started. |
| 2. Fine-tune + eval | **Done end to end.** 45k-step run + eval harness with outcome taxonomy; 65% on unseen layouts vs 84% on training layouts. |
| 3. Smooth inference | **Done on the remote GPU.** 500-trial paired runs: dead time 7.9% → 1.0% at success parity (67.4% → 67%). `--blend linear` added. One analysis open (k=10 smoothness columns, `no_grasp` mechanism). |
| 4. RECAP | **Stages A and C done.** 656 rollout+correction episodes collected on `eval_100_v322`; privileged 201-bin distributional V trained and swept. Remaining: advantage labelling, conditioned fine-tune, eval, attribution control. |

## Remote machine
`ssh axibo` (10.203.44.245, key `~/.ssh/id_ed25519_axibo`, VPN required). **RTX 5090 32 GB** (this file said
4090 24 GB until 2026-09-30; `nvidia-smi` reports 5090 / 32607 MiB. Every Task 3 latency number was measured on
this machine, so the figures stand but the GPU label in the report must say 5090), Ubuntu 24.04, Python 3.12,
32 cores, 31 GB RAM. Project at `~/axibo-project`, venv built from `requirements.txt`.
History: the box corrupted memory/filesystem during the first setup (SIGSEGV/SIGBUS, truncated files, `unknown opcode` in frozen stdlib, read-only remount). It was repaired; if those symptoms return it is hardware, not the install.

**Backend is now auto-detected** (`axibo/backend.py`): CUDA if torch sees it, else Metal on Apple arm64, else CPU. `AXIBO_BACKEND` / `AXIBO_DEVICE` override. **No more sed after copying files.**

## Pipeline
1. `sample_layouts.py --n N --seed S --out data/layouts/<name>.csv`
2. `collect_demos.py --layouts … --n-envs 16 --out data/lerobot/<name>` → LeRobot dataset + `collection_log.csv`
3. `train_smolvla.py --dataset data/lerobot/<name> --steps N --save-every M --batch-size 32 --num-workers 8`
4. `eval_policy.py --checkpoint … --layouts … --name <run>` → `outputs/eval/<run>/{eval_log.csv,summary.json}`
5. `eval_async.py …` same, plus the queue/latency model (Task 3)
6. `eval_report.py <run> [<run> …] [--failures]`, `plot_motion_dist.py <run> <run>`, `plot_trace.py`, `replay_trace.py <run> --trial N`

Task 4 adds two stages after 4:
- `collect_rollouts.py --checkpoint … --layouts … --out data/lerobot/<name>` → rollouts + scripted corrections
  as one LeRobot dataset + `rollout_log.csv`
- `train_value.py --dataset data/lerobot/<name> [--exclude-corrections] [--hidden …]` →
  `outputs/value/<run>/{value.pt, values.npy, metrics.csv}`

## Task 1 decisions made this session (all in the code with comments)
- **Torsional friction ON** (`SimConfig.enable_torsional_friction`). With Genesis' default point-contact model a side-grasped cylinder is a free hinge about the pad-normal axis: it tipped from 0.5° to 58.8° during the lift and landed upside down. `contact_pruning_tolerance` did not help; torsional friction kept the tilt at 0.2°.
- **Yaw convention.** `CUBE_GRASP_YAW_OFFSET_DEG = 180` and `SYMMETRIC_GRASP_YAW_DEG = -90`. Without the offset joint6 sat against its ±179.9° limit: 13% of grasp/drop angles were within 10° of the wall, and one degree of cube rotation flipped joint6 by 359°, turning a 28° wrist motion into 331°. Visually identical scenes carried opposite targets — unlearnable. With the offsets the whole workspace maps into a contiguous [−38°, +118°] band, clear of both limits.
- **2× faster demos** (`PlannerConfig.time_scale = 0.5`) → ~8.4 s episodes. 50/50 success kept, xy precision unchanged (0.89 → 1.05 mm mean), collection time halved. 73% of frames used to have <0.005 rad of joint motion, which is why a policy re-planning every 10 steps stalled.
- **Free-space speed cap** `spline_max_joint_speed` 1.8 → 1.2 rad/s: at 1.8 the arm lagged its own target by 0.265 rad during `to_pregrasp` versus ≤0.04 rad everywhere else.
- **Gripper starts open** (`home_qpos`), and `plan_demo` drops the `open` segment, removing 15 static frames per episode.
- **Start-pose jitter** `start_jitter_rad = 0.05` (±2.9°/joint ≈ 13 mm at the gripper), seeded via `PiperStackingScene(seed=)` / `reseed()`.
- **Cameras and lighting**: `ambient_light` 0.45 (Genesis defaults to 0.1 — the table sat in near-shadow); top cam (0.55, 0, 0.42) fov 56; side cam (0.26, −0.45, 0.16) fov 48, low and near-horizontal so stack height is visible while the arm occludes the top view. A true top-down camera was tried and **rejected**: perfect at episode start, fully blocked by the arm during the grasp. The fovs are the smallest (+2° margin) that keep every object fully in frame anywhere in the workspace, on the table or stacked — verified by projection, not by eye.
- **`min_dist` 0.08 → 0.10 m** between object centres (2.3 cm → 4.3 cm worst-case surface gap). Sampling unaffected (250/250).

## Data / models on disk (remote unless noted)
- Layouts: `main_250v3.csv` (seed 3, training), `eval_100_v228.csv` (seed 228), `eval_100_v322.csv` (seed 322), `testv2_10.csv` (local smoke).
- `data/lerobot/main_250v3` — 1250 episodes, **1243 successful**, ~8.4 s each.
- `data/lerobot/v322_data_for_recap` — 656 episodes (500 rollouts + 156 corrections), 172,999 frames.
  Task 4 stage A.
- `outputs/value/nc_h64_dropout_only` — the chosen value function (val 2.4735, 20,041 params);
  `values.npy` is E[V] for all 172,999 frames. Sweep siblings: `nc_small`, `nc_reg`, `nc_reg2`, `nc_h32`,
  `v322_nocorr`, `v322_data_for_recap` (the last two predate the capacity sweep).
- `outputs/train/main_250v3/checkpoints/{005000…045000}` — 45k steps, batch 32, 4 passes. Val loss plateaus at ~30k (0.0130) and wobbles to 45k (0.0089–0.0140) while train keeps falling → mild overfitting, more steps would not help.
- Local: checkpoint 045000 copied to `outputs/train/main_250v3/checkpoints/045000`.

## Task 2 results (checkpoint 045000, 500 trials each, seed 0)
| run | layouts | success |
|---|---|---|
| `eval_45k_train` | first 100 training layouts | **84.2%** |
| `v3_45k_eval228` | unseen (seed 228) | **65.2%** |
| `v3_45k_eval322` | unseen (seed 322) | 62.4% |
| `v3_30k_eval228` | unseen, checkpoint 30k | 58.2% |
| `eval_45k_v228_25steps` | unseen, re-plan every 25 steps | 66.6% |

- **19-point generalization gap** (84% train vs 65% unseen) — the headline number is 65%.
- Failure mix on unseen: `misplaced` 14.2%, `dropped_in_transport` 12.8%, `no_grasp` 2.8%, `timeout_carrying` 2.8%, `stack_collapsed` 1.8%.
- **Cylinder destinations are much harder**: 71% (cube dst) vs 57% (cylinder dst). Its top is a 12.6 cm² disc vs the cube's 16 cm² square.
- **~95% of failures used the whole chunk budget**; successes averaged 7.8 of 10 chunks. A larger budget is worth testing.
- Re-planning twice as often (25 vs 50 actions per chunk) changed success by +1.4 pts (noise) but shifted the mix: fewer drops, more `no_grasp`/`timeout`.
- **Held-out pair finding**: on `red cylinder → red cube` the policy built a clean stack on the **blue cube** — 16 mm from the wrong object, 218 mm from the right one. It follows shape and ignores the colour word. Currently labelled `timeout`; deserves its own `wrong_destination` class.

### The dominant failure looks like an off-centre *grasp*, not a bad *placement* (2026-09-30, unverified)
Operator observation from replaying the `test_cl` failures: in almost every case the arm brings the gripper to
roughly the right spot above the destination and releases there — but the object sits off-centre **in the jaws**,
so it lands offset by that same amount. Had the object been grasped through its middle, the same trajectory
would have succeeded.

Circumstantial support: after re-scoring, **10 of 17 failures are `misplaced`** with release offsets clustered
at 19.7–37.3 mm, which is the right order of magnitude for a partial grasp on a 40 mm object. Two earlier
findings point the same way — `no_grasp` failures displace the source 4.3 mm (p50) before the jaws close versus
1.1 mm in successes, and `settled`/grasp-centring problems dominated Task 1 debugging.

**How to test it** (one new logged field): put the TCP in `Frame` via `sim.grasp_point_world()`, which already
exists. While the object is held, `|src_xy − tcp_xy|` is the grasp offset and is constant; compare it to
`|src_xy − dst_xy|` at release. If they match, the arm aimed correctly and the object rode offset the whole way.
If the release offset is much larger, the arm also mis-aimed. The same field answers the open Task 3 question
(TCP-to-object error at the grasp closure), so it closes both.

**Why it matters for Task 4.** If confirmed, the failure originates ~250 steps before it becomes visible, which
is exactly the credit-assignment problem a value function exists for: `V` can learn that an off-centre grasp
predicts failure long before anything looks wrong, and advantage conditioning would push the policy toward
centred grasps. It also decides the correction design — a correction that only fixes the *placement* treats the
symptom, while one that re-grasps treats the cause. **Decide this before building the correction pipeline.**

## Eval harness (Task 2 infrastructure)
`axibo/outcome.py` — conditions C1–C12 (three of them footprint tests at different widths) with explicit constants, five latched milestones, and 10 terminal labels: `success`, `stacked_dst_moved`, `stacked_third_moved`, `knocked_off`, `misplaced`, `released_in_air`, `dropped_in_transport`, `no_grasp`, `timeout_carrying`, `timeout`. `test_outcome.py` forces each one from a hand-built trace (no sim, no checkpoint), since several occur only a handful of times per 500 trials.
- Heights are absolute from the table, never relative to the destination's current centre, so a stack built on a toppled destination cannot pass.
- `lifted` = source centre above `resting + object_height` (cube 0.060, cylinder 0.075 m); demos lift exactly 80 mm, the worst tipping artifact is +14.6 mm.
- **Three footprint tests, at three widths** — all sharing one `_in_footprint` helper (square in the
  destination cube's yaw frame, disc for the cylinder), so they cannot drift apart:
  - **C1 `over_dst`**, the bare half-width (20 mm) — the success test. Centre-inside-support is the static
    stability condition, so this one is physically derived and stays.
  - **C1a `overlaps_dst`**, grown by `transport_margin` = 20 mm → 40 mm — the `transported` milestone. The bare
    footprint ignores the source's own width and was far too strict: over `test_cl` (50 trials) six failures
    came within 25–34 mm and were labelled `dropped_in_transport` ("never got there") when the arm had plainly
    brought the object over and placed it badly. 20 mm also equals `object_half_width` for both objects here
    (cube edge 40, cylinder ⌀40), so this is equivalently "grown by the source's half-width".
  - **C1b `stably_on_dst`**, shrunk by `stable_frac` = 0.5 → 10 mm — the `knocked_off` / `misplaced` split. A
    cube centre 19.7 mm from a 20 mm cylinder axis is 0.3 mm inside the support edge with almost no area under
    its centre of mass; two such placements (19.7, 19.8 mm) fell on their own yet were blamed on the arm. **A
    strict geometric guarantee is not available** — full containment needs `w_dst − w_src` = 0 mm for
    equal-sized objects — so this is a chosen fraction, stated as such, not a derived bound.
- `transported` = **C1a** and ≥ 2 cm above stack height (demos hover 33 mm above it).
- **Success criterion (unchanged)**: footprint + stack height ±5 mm + settled + gripper open, at the final frame.
- **Episode termination (changed 2026-09-30)**: the **release of a transported object** starts the 4 s window
  during which the policy keeps acting (so a policy that knocks its own stack over fails), then the final state
  is judged. Previously the window started when the stack *detector* latched — which never fires on a failed
  placement, so every failed episode burned the full 500-step budget. A re-grasp during the window cancels it,
  so a genuine retry is not cut off.
- **The placement event is what attributes failures**, in three tests. Arrival: `transported` never latched →
  `dropped_in_transport` — the object was lost before the placement stage, whether the jaws opened or it
  escaped them. Then, at the release frame, height: `|src_z − stack_z| > tol_place` (10 mm) → `released_in_air`
  — arrived over the destination, then let go above it instead of onto it. Otherwise footprint: source centre
  footprint **shrunk to 10 mm** (C1b) → `knocked_off`, else `misplaced`.
- **Verified by re-scoring `test_cl` (50 trials) offline from its traces**: 8 of 50 relabelled — the 6 bogus
  `dropped_in_transport` became 4 `misplaced` + 2 `released_in_air`, and both `knocked_off` became `misplaced`.
  `knocked_off` and `dropped_in_transport` both went to **0**, and **success stayed at 33/50**, confirming the
  changes touch only failure attribution. Inspection of the replays confirmed the arm was not knocking stacks
  off, matching the measured zero. `stack_collapsed` is subsumed by `knocked_off`, and the old overhang case
  that fell through to `timeout` is now `misplaced`. `knocked_off` is an **inference** ("set down inside the
  footprint and not there at the end"); the destination's own tilt/displacement is checked first, since that is
  a fact rather than a judgement. Note the footprint test compares the source's *centre* to the destination's
  face, so a centre on the edge — roughly half the cube overhanging — still counts as inside.
- `set_down` is **height only**. An earlier version also required the source to be nearly stationary, but the
  policy releases while still moving (1–114 mm/s over 5 trials, all within 3 mm of stack height) and that cut
  called two of them `released_in_air`, one of which succeeded. Whether the placement survived is judged at the
  end of the settle window anyway. `release_speed_mm_s` is still logged as a diagnostic.
- **Success is tested before the placement branch**, so a trial that succeeds despite a sloppy release is still
  `success` — the placement geometry only attributes *failures*.
- **`settled` is net displacement over 0.5 s ≤ 3 mm**, not instantaneous speed: a stack resting on the cylinder jittered at 32 mm/s in contact and an instantaneous test called it "moving" forever.
- Validated: 20/20 `success` on scripted demos, and each failure class forced deliberately and confirmed.

## Task 3 (async inference) — done on the remote GPU, one analysis open
`axibo/rollout.py` runs a queue-based controller following SmolVLA's Algorithm 1 (arXiv 2506.01844 §3.3) plus a hold branch, so **one knob covers both conditions**: `k = --queue-threshold`, the queue length at which a new chunk is requested. `k = 0` = naive synchronous (the arm holds while computing); `k ≥ L` = asynchronous.
- The sim is not real-time, so **latency is modelled**: a chunk requested at step s arrives at s + L, L = measured or fixed inference time in control steps. Between those steps the robot executes what it already had.
- **Index alignment is by actions executed, not elapsed time.** A chunk resumes from wherever the robot actually
  got to: async drained L actions from the old queue while computing, so it resumes at index L; sync (k=0) stood
  still, so the observed state is still current and it resumes at index **0**. Partial overlap (0 < k < L) resumes
  at k. Usable chunk = 50 − offset.
- **Stall fraction floors at ~1%, not 0**: the first chunk of an episode has nothing to overlap with, so the arm
  holds L steps at the start in every condition. k = L already removes the steady-state stall; higher k is only
  jitter margin.
- Practical range: `L ≤ k < 50 − L`; above that inference runs continuously.

### Remote results, RTX 5090, measured latency 126–128 ms → L = 4, 500 trials, `eval_100_v228`
Paired (same layouts / pairs / seeds); baseline for comparison is `v3_45k_eval228` (`exec_steps=50`, 65.2%).

| | k=0 sync | k=6 async | k=10 async | k=10 + blend |
|---|---|---|---|---|
| success | 67.4% | 63.2% | 66% | 67% |
| stall_fraction | 7.9% | **1.0%** | 1.0% | 1.0% |
| stalled steps / episode | 33.3 | **4.0** | 4.0 | 4.0 |
| vel_disc_boundary_max | **0.380** | 0.681 | — | — |
| jerk_boundary_rms | **119.5** | 186.3 | — | — |
| chunk_disagreement [rad] | 0.0 | 0.0089 | — | — |
| calls / episode | **8.35** | 10.03 | ~10.4 | ~10.4 |

(the k=10 smoothness columns still need pulling off the remote with
`eval_report.py eval_45k_v228_async_queuesize0 eval_45k_v228_async_queuesize6 eval_45k_v228_async_k10_noblend eval_45k_v228_async_k10_blend`)

**What async actually buys: dead time, not smoothness.** It removes 7.9% → 1.0% of held steps (33.3 → 4.0 per
episode, the residual being the unavoidable first chunk). Success is statistically unchanged (k=0 67.4% vs k=10
67%), so the spec's "success after ≥ before" guard holds.

**Async is *jerkier* than sync, and the reason is structural.** Sync's queue is empty when a chunk lands, so the
chunk starts at index 0 from exactly the state it was computed from — the new plan is anchored to where the arm
really is. Async splices `new[L]` over what `old[...]` was about to do, ~10× per episode, and `new[L]` presumes
the arm spent those L steps executing `new[0:L]` when it actually executed the old chunk's tail. That mismatch is
the fundamental approximation in offset alignment, and it grows with L.

Measured command/velocity steps (200 trials, max over arm joints), showing the excess is real but tail-concentrated:

| | k=0 switch | k=0 elsewhere | k=6 switch | k=6 elsewhere |
|---|---|---|---|---|
| \|Δa\| p50 [rad] | 0.0104 | 0.0059 | 0.0132 | 0.0064 |
| \|Δa\| p90 | 0.0306 | 0.0251 | **0.0595** | 0.0258 |
| \|Δv\| p90 [rad/s] | 0.2462 | 0.1217 | **0.5122** | 0.1272 |

Sync's seam costs only 0.0055 rad above ambient at p90; async's costs 0.0337 — 6× more. Converting the splice to
velocity units (0.0089 rad × 30 Hz = 0.267 rad/s) accounts for ~89% of async's excess boundary discontinuity
(0.681 − 0.380 = 0.301), which is what motivated blending.

**k must exceed L with margin, and the margin is the blend window.** `overlap = k − L`, and the replan period is
`50 − k` (independent of L). k=6 gave overlap 2 — nowhere to put the jump — and cost 4 points of success. k=10
(overlap 6) recovered it. A linear ramp over the overlap cuts the peak command step by ~1/overlap (verified:
0.066 → 0.0135 at overlap 8) and was worth ~1 point of success, i.e. within noise.

**`--blend linear`**: ramps `lam = 1/m … 1` across the overlap so it ends fully on the new chunk. Dim 6 (gripper)
is switched hard — averaging "open" with "closed" commands a half-closed gripper, which at the grasp drops the
object. Queue length is unchanged (`overlap + (50 − offset − overlap)`), so blend-on vs blend-off at the same k
differ only in action values.

**Grasp failures are an async artifact, mechanism unresolved.** k=6 turned 36 successes into `no_grasp` and cured
0 (14 → 60 overall), while *fixing* `misplaced` (43 flips its way) — two opposite effects that nearly cancel.
Two candidate mechanisms were tested and refuted: arm speed entering the grasp closure is identical across all
groups (0.049–0.061 rad/s), and switches land near the closure in 59.5% of successes but only 11.7% of
`no_grasp` failures. What `no_grasp` *is*: an approach-accuracy failure — source object displaced 4.3 mm (p50)
before the gripper closes vs 1.1 mm in successes, with the same severity in both conditions but 4× the frequency
in async. Untested hypothesis: accumulated splice drift. The decisive measurement would be TCP-to-object xy error
at the closure step (needs FK, not in the logged arrays).

**Policy timing sensitivity — carry this caveat into every number.** 159 of 500 paired trials (32%) change outcome
between k=0 and k=6, i.e. from a 4-step timing shift. McNemar χ²=2.52 (p≈0.11) on the success difference.

### Superseded Mac numbers
The earlier 5-trial Mac table (L=20) was produced before the index-alignment fix, which made sync consume every
chunk from index L — discarding 20 of 50 valid actions and jumping the target forward although the arm had not
moved. Its apparent smoothness advantage for async was that bug, and its conclusion is the reverse of the
corrected remote result. Not reproduced; kept only as a record that the bug existed.
| total command discontinuity per episode | 6.00 rad | 6.64 rad |

**Reading:** async removes the dead time and halves the violence of each transition, at the cost of ~2× compute, ~2× as many transitions, and no gain in task time. Whole-episode jerk/smoothness averages showed **nothing** (241 → 226) because 72% of sync's discontinuity is concentrated in 10% of its steps — and because stalled steps contribute zero jerk, flattering the stalling condition. Distributions of speed/acceleration overlap wherever the arm is moving; only the mass at zero differs.

Metric notes for the report:
- Report **tails and ratios**, not means: p99 of command change, and switch-window vs baseline acceleration.
- Window length does not matter (peak identical at 3, 5, 8 steps) — the PD response lands within 100 ms.
- Submovement counting (speed peaks, motion arrests) **does not work** here: it cannot distinguish an intentional pause from a stall, and ranks async worse.
- `max` acceleration (~43 rad/s², ~1.5 rad command jump) is the initial move from rest in **both** conditions — not a switch artifact.

## Task 4 (RECAP) — stages A and C done

### Stage A: rollout + correction collection (`collect_rollouts.py`)
`data/lerobot/v322_data_for_recap` on the remote: **656 episodes** (500 policy rollouts on `eval_100_v322`
x 5 train pairs, plus 156 accepted scripted corrections), 172,999 frames, 4611 s of collection.
Two passes per rollout: pass 1 runs the policy and records it; pass 2 replays pass 1's recorded actions up to
the transport moment, then hands to IK (`plan_place_from_here`), aiming the TCP at `dst_xy - delta` so the
*object* lands centred. Stall frames are skipped at write time, so all milestones in `rollout_log.csv` are
**written-frame** indices. Held-out pair excluded, as in Task 1.

### Stage C: the value function (`axibo/value.py`, `train_value.py`)
201 bins, cross-entropy on two-hot targets, returns closed-form from `rollout_log.csv`:
`R_t = -(stacked_step - t)` for a success, `-(T - t) - c_fail` otherwise, normalised by
`scale = max(n_frames) + c_fail` (nothing clips; bin 0 = -1.0, bin 200 = 0.0).

**`c_fail = 250`** — the paper gives no number, only *"a large constant ... chosen so as to ensure that failed
episodes have low values"*. 250 frames is one typical successful attempt, i.e. the cost-to-go of failing. The
hard floor is `max(success latch) - min(failure length)` = 80 on the real data; below it a fast failure
outscores a slow success (early termination acting as a reward). Measured separation at 250: worst success
-0.396, best failure -0.626, **gap 45.9 bins**, no overlap.

**V is privileged — it reads simulator state, not images.** Declared deviation. RECAP's contribution is the
mechanism (value -> binarised advantage -> conditioning -> high-advantage at inference), which is unchanged by
how V perceives; and V runs only offline to label, so the policy still sees one bit. The faithful image version
would need a 30-50 min frozen-VLM feature pass over 173k frames and would leave no way to separate an encoder
problem from a reward problem. **Report line:** *V is trained on simulator state rather than images; it runs
only offline to label advantages, so the policy sees only the binary indicator — this would not transfer to a
real robot.*

**Features, 43 dims.** Poses ordered by **role** (source, destination, third) rather than object identity —
the command is always "put the {source} on the {destination}", so role-ordering encodes its entire information
content bijectively. No text tokens, no task one-hot (which would split the data into five near-separate
functions). Plus `src-dst`, the gripper pose, `src-tcp` and two shape flags.
*Report line:* *V receives the command as a structured encoding rather than as tokens — lossless for this
instruction template, but not a general language interface.*

**`axibo/kinematics.py` (new)** derives the gripper pose from the recorded `sim.qpos` by numpy FK, at load
time — no dataset change, no Genesis in the training script. **Verified against Genesis to 0.0001 mm** across
8 random configurations spanning the full joint ranges. This closes the open "log the TCP" item: the derived
grasp offset `|src_xy - tcp_xy|` is 0.2-24.2 mm (p50 11.4), matching the 10-25 mm measured live during
collection, and it separates outcomes (successes 0.2-2.0 mm, misplacements 11-17.5 mm on the smoke set).

#### Two bugs found and fixed (both would have silently corrupted every target)
1. **Latchless successes were scored as failures.** `stacked_step` is absent when the stack is confirmed after
   recording stops (`written_index` returns None past the cut) and can exceed `steps` for corrections (written
   untranslated). The first version treated both as failures and charged the full `c_fail`, putting ~150
   corrections 0.34 below where they belonged and producing an apparent success/failure overlap that looked
   like a `c_fail` problem. Fixed: **the `outcome` column is authoritative**; `stacked_step` only says *when*
   the -1/frame stops, and a missing one anchors to the last written frame.
2. **Excluded episodes were unscored.** `--exclude-corrections` originally skipped building their features, so
   `values.npy` held NaN for every correction frame — exactly the frames stage D needs. Exclusion now affects
   training only, not inference.

#### The correction episodes destroy the grasp signal (the main stage-C finding)
A correction replays its parent's grasp and then succeeds, so the same lift-step state carries opposite
returns and V can only fit the average. Held-out ranking of eventual success at the grasp:
**0.424 with corrections in training, 0.610 without** (mean E[V] gap success-vs-misplaced 0.061 -> 0.157).
Keeping them is faithful to the paper (V of the data mixture, where a bad grasp is recoverable); dropping them
gives V of the *uncorrected* policy — which is the baseline a correction's advantage should be measured
against. **Decision: train V with `--exclude-corrections`, score every frame.**

#### Capacity sweep — overfitting was capacity, not missing regularization
All runs: `--exclude-corrections --patience 20`, 124,144 train / 6,401 val frames, split by layout.

| config | params | best val | final - best |
|---|---|---|---|
| h128 (first attempt) | 48k | 2.6031 | +0.10 |
| h128 + noise 0.1 + wd 1e-3 | 48k | 2.6049 | +0.16 |
| h128 + noise 0.2 + wd 1e-2 | 48k | 2.6380 | +0.15 |
| h64 + dropout 0.3 + noise + wd | 20k | 2.4990 | +0.02 |
| **h64 + dropout 0.3, nothing else** | **20k** | **2.4735** | **+0.018** |
| h32 + dropout 0.3 + noise + wd | 9k | 2.5262 | +0.006 |

- **Input noise and weight decay do not help** — on h64 they made it worse, on h128 they did nothing.
- **h32 underfits**, so 20k params is a real optimum for ~490 episode-level labels, not just "smaller is better".
- The train/val gap collapsed from +0.10 to +0.018, so early stopping is now almost unnecessary.
- The binding constraint is **490 episode-level labels**, not frames: 124k frames are ~250-frame runs of a
  near-identical state sharing one label. No regularizer manufactures more.

**Chosen V: `outputs/value/nc_h64_dropout_only/`** (val 2.4735, 20,041 params). Reproduce with
`train_value.py --dataset data/lerobot/v322_data_for_recap --exclude-corrections --hidden 64 --dropout 0.3
--patience 20`. Mean E[V] at the grasp orders every outcome correctly over 490 episodes: success -0.221,
`knocked_off` -0.304, `misplaced` -0.367, `released_in_air` -0.433, `dropped_in_transport` -0.521.

#### Honest caveats for the report
- **Held-out discrimination is weak and under-measured.** The val split is 5% = 25 episodes, giving a ranking
  statistic a standard error around ±0.15, so 0.66 at the grasp is not separable from chance at that sample
  size. The per-outcome ordering over all 490 episodes is the stronger evidence. **k-fold over layouts (~10
  min) is the cheap fix** and would also give out-of-fold V for every frame, which is strictly the right input
  for advantage labelling.
- **In-sample ranking is still ~0.90 against ~0.66 held out**, so memorisation has been reduced, not removed.
- **Release-step discrimination (~0.70-0.86) is partly tautological**: `axibo/outcome.py` derives the label
  from the geometry at the release frame, and V has that same vector as an input. It is reading the
  measurement the labeller used, not predicting.
- **`src - tcp` is the closest thing to a hand-engineered feature in the critic.** Justified as a coordinate
  change (the relation is visible in the policy's own wrist view) and because the privileged V is already a
  declared deviation. We deliberately stopped there: feeding V a *calibrated* grasp-badness score would make
  the whole RECAP result attributable to our own failure detector rather than to a learned critic, which is
  worse than a weak result. Rule held: **a feature may be a coordinate change, never a judgment.**
- Absolute object `xy` was *not* stripped despite letting V memorise layouts, because the workspace edges are
  where reachability genuinely bites — regularisation reduces memorisation while leaving information
  available; deleting features does not.

### Stage D: what we do differently from the paper, and why

The paper refits **both** V and the policy from the pre-trained checkpoint on *all* data collected so far, every
iteration, and estimates eps_l as a percentile over that full dataset:

> "Both the value function and policy are finetuned from the pre-trained checkpoint, rather than the policy and
> value function from the last iteration. We found this to be useful for avoiding drift over multiple
> iterations."
> "the dataset D_piref consists of all of the data collected so far, including all demonstrations and
> autonomous task attempts"
> "We set eps_l to the 30% percentile of values predicted by the value function for the task l."

We keep the from-pre-trained initialisation (policy trained from `lerobot/smolvla_base`, not from `045000`) and
the full data mixture (demos + rollouts + corrections, failures included - *"effectively utilize both good
(near-optimal) and bad (suboptimal) data"*). We deviate on **what V and eps were estimated from**, and this is a
deliberate test rather than a shortcut:

**Hypothesis.** If V has learned a state-based value rather than memorised layouts, it should transfer - to
unseen layouts, and to a different behaviour distribution (scripted demos rather than policy rollouts). And
eps, being a threshold on a smooth quantity, should carry over with it. So V is fit on the **v322 rollouts
only** and then *applied* to `main_250v3`, with the same five per-pair eps values reused unchanged.

**Test.** The positive rate V assigns to the 1250 demos. They are 1243/1250 successful and expertly executed,
so transfer predicts an overwhelmingly positive rate. Near-chance would falsify the hypothesis, and that is
itself the result.

**This is also methodologically cleaner than the paper on one axis.** The paper fits V on all the data and then
labels that same data, so its labels are in-sample - partly recall rather than judgement, which matters because
our own V ranks held-out episodes at ~0.66 against ~0.90 in-sample. Labelling the demos with a V that never saw
them is out-of-sample by construction.

**Caveat to state plainly:** that same 0.66-vs-0.90 gap means transfer is partial, so the hypothesis is being
tested, not assumed.

Not a deviation: refitting *between iterations* does not apply - we run one iteration, so there is nothing to
refit between.

So the three stage-C/D deviations and their reasons are:
1. V reads simulator state, not images - time, and V is offline-only so nothing leaks to the policy.
2. V fit on rollouts only, corrections excluded - measured: corrections replay their parent's grasp and then
   succeed, so including them cancels the grasp signal (held-out ranking 0.424 -> 0.610).
3. eps estimated on the rollouts and reused for the demos - the generalisation test above. Consequence: the
   positive rate on the combined training set is ~89% rather than 70%.

### Stage D (next) — not started
`A_t = V(o_{t+N}) - N/scale - V(o_t)`; binarise with `I_t = 1(A_t > eps)`; corrections forced `I_t = True`
(paper); indicator appended to the task string (`"... [good]"` / `"[bad]"`) since SmolVLA already tokenises
the task, so the architecture is untouched; conditioning dropout for the CFG-like behaviour; always request
`[good]` at inference. Open decisions: **N** (suggest 50, matching the action chunk, so the advantage scores
exactly the decision the policy makes) and **eps** (pick from the measured advantage distribution, not a
priori — `-N/scale` biases A downward so eps=0 would mark almost everything negative). Then eval on
`eval_100_v228` plus the attribution control: identical data, indicator stripped.

## Known issues / open items
- **Task 3: the k=10 smoothness columns are still unread** (remote was unreachable). Run
  `eval_report.py eval_45k_v228_async_queuesize0 eval_45k_v228_async_queuesize6 eval_45k_v228_async_k10_noblend eval_45k_v228_async_k10_blend`
  and fill in the table above. The open question is whether blending pulled `jerk_boundary_rms` from 186 back
  toward sync's 119, and whether `no_grasp` fell from 60 toward 14.
- **All headline Task 3 numbers use `--latency measured`**, not fixed. Reproducible reruns would want
  `--latency fixed --latency-ms 130`; measured is defensible here only because p50/p99 are 126/128 ms.
- **`no_grasp` mechanism unresolved** — see Task 3. Needed TCP-to-object xy error at grasp closure (FK);
  `axibo/kinematics.py` now provides it (verified against Genesis to 1e-4 mm), so this is answerable from the
  existing datasets without re-collecting.
- `wrong_destination` outcome class not implemented (held-out pair currently lands in `timeout`).
- Success tolerances still described as placeholders in `axibo/success.py`; `axibo/outcome.py` has the considered ones.
- Latency aggregation in `eval_policy.py` is a p50-of-per-trial-p50s; `eval_async.py` keeps every sample. Fix before quoting p99.
- `--blend` accepts only `none`. Chunk overlap disagreement is ~0.0106 rad, but the *splice* discontinuity is ~10× that, so blending may still be worth trying.
- Async's residual 19% "still" fraction is mostly the 4 s settle window; exclude it for a clean dead-time number.
- Docs: `CLAUDE.md`'s Scripts section still describes the pre-2026-09-28 code in places.
- Nothing is committed to git (repo has no commits). **Submission needs a private repo with
  `@RVSagar` and `@ax-anoop` added.**
- Report not started. Reversal test, held-out-pair eval and the results video are all still missing.

## Suggested next steps
0. **Stage D** (see Task 4 above) — advantage labelling, conditioned fine-tune, eval on `eval_100_v228`,
   attribution control. The fine-tune is the long pole.
   *Done 2026-09-30:* the TCP / grasp-centring hypothesis — `axibo/kinematics.py` derives it by FK from the
   recorded joints; the offset is 0.2-24.2 mm and separates successes from misplacements.
1. Remote Task 3 comparison at the GPU's real latency, ≥125 trials per condition (the success guard).
2. Add `wrong_destination`; re-score existing held-out runs.
3. Checkpoint sweep on eval layouts (30k vs 40k vs 45k) — the success-vs-steps figure.
4. Reversal test (`--pairs reversal`) and the held-out-pair result, reported separately.
5. Start the report; every decision above has its reasoning recorded in code comments.
6. Task 4 stage C polish, if time: **k-fold over layouts** (~10 min) to replace the 25-episode val split and
   produce out-of-fold V for every frame.
