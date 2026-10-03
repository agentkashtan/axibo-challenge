# Stacking with a VLA in Genesis — what I built, what broke, and what I learned

## 1. Setup

I chose SmolVLA as the model to fine-tune: small enough to train on a consumer GPU, but still a capable VLA.

That fixed the scene format, since SmolVLA expects a robot state, three camera images and a text instruction.

**Robot state**, 7 numbers: 6 joint angles plus one gripper parameter. I predict joint angles rather than
end-effector deltas so the model does not have to learn the robot's kinematics, which is a lot to ask of this
much training data.

**Cameras**: wrist, top and side. I tried several placements before settling. The top camera is angled rather
than straight down, because a straight-down view is completely filled by the arm during the grasp — the arm
approaches from above too. The side camera sits low and nearly horizontal so stack height is visible, which is
what the success check measures and what the top view cannot see.

I also found Genesis's default ambient light (0.1) far too dark and raised it to 0.45.

## 2. The data engine

### 2.1 Layouts

I needed a reproducible way to get varied object layouts, so I wrote `sample_layouts.py`
(`--n`, `--seed`, `--out`), which generates random scenes and writes them to a CSV indexed by `layout_id`.

For each layout it takes a random permutation of the three objects and places them one at a time. The first
gets a random xy. For each next object it draws a random xy and checks it is at least 10 cm from everything
already placed, redrawing until it fits. Permuting the order matters: without it the first object is always
uniform over the workspace and the later ones are always pushed away from it, so the bias would attach to a
specific object.

Objects are always upright, z pointing up. For yaw I started with random values and changed both:

- **Cylinder**: rotationally symmetric, so its yaw is unobservable. Fixing it costs nothing in scene variety
  and makes the grasp consistent across trials.
- **Cube**: symmetric every 90 deg, so I sample yaw in [0, 90) instead of [0, 360). That still covers every
  physically distinct layout while keeping the scenes more uniform.

With that I have a script that builds a Genesis scene from a layout id, which is what collection runs on.

### 2.2 The scripted demo

Ten segments:

```
to_pregrasp -> open -> descend -> close -> grasp_wait -> lift -> to_predrop -> lower -> release -> release_wait -> retreat
```

Free-space moves (`to_pregrasp`, `to_predrop`) are quintic splines in joint space between two IK solutions -
the path through the air does not matter. The four vertical moves (`descend`, `lift`, `lower`, `retreat`) are
IK-solved every 5 mm along a straight line, each solve seeded with the previous one. Those have to be straight:
the gripper descends into a few centimetres of clearance beside another object, and a joint-space
interpolation between the same endpoints bows sideways.

To watch one demo with the three camera feeds the policy will see:

```bash
python run_scripted_demo.py --layouts data/layouts/main_250v3.csv --layout-id 0 \
    --source "red cube" --destination "blue cube" --live-cameras
```

### 2.3 Two changes I made to the scripted demo after looking at the recorded data and the first training attempts

**Grasp yaw, so that similar scenes give similar grasps.** My first version picked the grasp yaw that
minimised joint movement from the arm's current pose. After the first training run I realised it hurt: the
same scene produced different grasps depending on where the arm happened to be, which the cameras cannot see.
I made the grasp depend only on the object pose - a fixed gripper yaw in the world frame for the cylinder, and
a fixed transform from the cube frame to the gripper for cubes.

One correction on that transform: taking the cube's yaw directly puts `joint6` on its limit. 13% of grasp and
drop angles landed within 10 deg of the +-179.9 deg wall, where one degree of cube rotation flips `joint6` by
~359 deg because it cannot wrap. Adding 180 deg is the same grasp, since the jaws are symmetric, and moves the
angles clear of both limits.

**Episode duration, because most frames had nothing happening in them.** At the original speed **73% of
recorded frames had under 0.005 rad of joint motion**. The demos were successful and nearly useless: a policy
re-planning every 10 steps learns that the right action is usually to stay still. I added a `time_scale`
factor and set it to 0.5, halving every segment duration (~8.4 s per demo). Success stayed 50/50 and
placement precision was unchanged.

From the same pass over the data I also dropped the free-space speed cap from 1.8 to 1.2 rad/s. At 1.8 the arm
lagged its own commanded target by 0.265 rad during `to_pregrasp` against 0.04 elsewhere, so the recorded
action and the recorded state disagreed.

### 2.4 Which pair I held out

I held out **`put the red cylinder on the red cube`**. Two constraints picked it.

With three objects there are six ordered pairs. Two are cube-on-cube, four involve the cylinder:

| | red cube | red cylinder | blue cube |
|---|---|---|---|
| **red cube** -> | - | cube->cyl | cube->cube |
| **red cylinder** -> | **held out** | - | cyl->cube |
| **blue cube** -> | cube->cube | cube->cyl | - |

First, I did not want to lose cube-on-cube coverage. If I held out one of the two cube-on-cube pairs, only one
would be left in the five training pairs. Holding out a cylinder-cube pair leaves 2 cube-cube and 3 involving
the cylinder.

Second, the blue cube is the only blue object in the scene, so every pair involving it is also the only data
the policy gets about that colour. Removing one would leave too little blue. That rules out everything except
the two red-only pairs, and I took the cylinder-on-cube direction.

Its reverse, `red cube on red cylinder`, stays in training — and turns out to be the hardest of the five.

### 2.5 Collecting the data

`collect_demos.py` takes a layouts CSV and runs the planner over every layout x pair, recording as it goes. It
batches across parallel Genesis environments (`--n-envs`), and writes a LeRobot dataset plus a
`collection_log.csv` with one row per episode so failures can be filtered before training.

```bash
python collect_demos.py --layouts data/layouts/main_250v3.csv --n-envs 10 --out data/lerobot/main_250v3
```

I used LeRobot's format rather than rolling my own. It was good enough for this and it saved me writing a
loader, a video writer and a chunked-storage layout.

250 layouts x 5 training pairs = **1250 episodes**, of which **1243 succeeded**. ~8.4 s each, **398 694
recorded frames** at 30 Hz. Throughput was 149 / 200 / 230 episodes per hour at 1 / 5 / 10 parallel
environments.

## 3. Training

Nothing special here. LeRobot ships SmolVLA, so I started from the open `lerobot/smolvla_base` checkpoint and
fed it all the demos.

```bash
python train_smolvla.py --dataset data/lerobot/main_250v3 --steps 45000 --save-every 5000 \
    --batch-size 32 --num-workers 8
```

45 000 steps at batch 32, about 4.6 passes over the 1250 episodes. 99.9M of 450M parameters train - the vision
encoder stays frozen. AdamW, betas (0.9, 0.95), 100 warmup steps then cosine from 1e-4 down to 2.5e-6. 5 h
11 m on the GPU.

Two things I had to set rather than accept:

- `--std-floor 0.01`. State and action use MEAN_STD normalisation, and `joint5` never moves in these demos
  (std exactly 0), so without a floor its normalised value blows up.
- The validation split is by layout, not by episode, so validation scenes never appear in training.

## 4. How I evaluate

`eval_policy.py` takes a checkpoint and a layouts CSV and runs the policy against every layout x pair. The
first version was the naive loop: read the observation, predict a chunk, execute it, repeat.

Then I needed async inference, so I wrote `eval_async.py`, which does both regimes through one parameter.

### 4.1 Why the latency has to be simulated

The simulator is not real-time. Physics only advances inside `sim.step_control()`, so the wall clock spent in
`predict_chunk` costs **zero simulated time** - a naive loop in sim has no chunk-boundary stall at all, even
though the real robot would have one. So I put the latency back into the sim timeline by hand.

Two quantities matter throughout. **`L`** is the inference time converted to control steps - 4 at 125 ms and
30 Hz. **`k`** is the queue threshold: the number of actions left in the queue at which the next chunk is
requested.

A chunk requested at step `s` becomes available at step `s + L`. During those `L` steps the robot keeps
executing the queue it already has. If the queue is empty it holds its last action, which is exactly what a
synchronous client does.

```
k = 0  (sync)                        k >= L  (async)
step  queue   action                 step  queue   action
 48   [a49]   a49                     38   [a39..a50]  a39   <- request fires at len(queue) = k
 49   []      request, HOLD           39   [a40..a50]  a40
 50   []      HOLD      | L steps     ..                     | chunk computes
 51   []      HOLD                    41   [a42..a50]  a42
 52   [b5..]  b5  <- arrives,         42   [b5..b50]   b5    <- arrives, spliced in with no gap
              consumed from index L
```

The arriving chunk is consumed from index `L`, not index 0, because its first `L` actions describe a moment
that has already passed.

### 4.2 The one knob

`--queue-threshold k` is the queue length at or below which a new chunk is requested.

- `k = 0` requests only once the queue is empty, so the arm holds for `L` steps. This is the naive
  synchronous baseline.
- `k >= L` means the chunk always lands before the queue runs dry, so there is no stall.

Both regimes are the same code path, which is the point - the sync baseline is not a separate implementation
that could differ in some other way.

## 5. Classifying the outcome

`axibo/outcome.py` splits the work into **online** and **offline** parts. Online conditions are evaluated
while the episode runs, because they decide when it ends. Everything else is inferred from the recorded trace
afterwards, so I can re-score a finished run without touching the simulator.

### 5.1 Online: two events

During the rollout I watch for exactly two things:

- **transport** - the source's footprint overlaps the destination's and it is above it
- **release** - the first gripper opening after transport

The episode ends 4 s (120 control steps) after the release, or when the 500-step budget runs out. The 4 s
window exists because success has to hold after the arm retreats, not at the moment the object is let go.

A re-grasp cancels the window: if the gripper closes again above the destination, the countdown resets, so a
genuine second attempt is not cut off mid-way.

### 5.2 Offline: the decision tree

The order of the questions is the whole design. It attributes a failure to **where the episode broke**, not to
what the final frame looks like.

```
lifted?
 |-- no --------------------------------------------> no_grasp
 |-- yes
     stacked at the final frame?
      |-- yes --> destination moved ----------------> stacked_dst_moved
      |           third object disturbed -----------> stacked_third_moved
      |           otherwise -----------------------> success
      |-- no
          transported?
           |-- no --> on the table -----------------> dropped_in_transport
           |          still holding it -------------> timeout_carrying
           |-- yes
               released?
                |-- no ---------------------------- > timeout_carrying
                |-- yes
                    placed at stack height?
                     |-- no -------------------------> released_in_air
                     |-- yes
                         destination already displaced?
                          |-- yes --------------------> stacked_dst_moved
                          |-- no
                              inside the shrunken footprint?
                               |-- yes ---------------> knocked_off
                               |-- no ----------------> misplaced
```

"Transported" is the same condition detected online in 5.1.

### 5.3 The outcomes and how each is decided

| outcome | what happened | how it is evaluated |
|---|---|---|
| `success` | source on the destination and still there after the arm retreats | at the final frame: source centre inside the destination footprint (half-width 20 mm, rotated into the destination frame for cubes), `\|z_src − stack_z\| ≤ 5 mm`, net displacement over 0.5 s ≤ 3 mm, jaws open |
| `stacked_dst_moved` | stacked, destination shifted or toppled | destination tilt > 10° or displacement > 10 mm from its initial pose |
| `stacked_third_moved` | stacked, third object disturbed | third object displaced > 10 mm |
| `knocked_off` | set down soundly, not stacked at the end | at the release frame, source centre inside the destination's footprint **shrunk by half** |
| `misplaced` | set down too far off to hold | at the release frame, source centre outside the destination's halved footprint |
| `released_in_air` | got the object over the destination, then opened the jaws too high on the way down | requires the transport milestone, so the source was already above the destination; then at the release frame `\|z_src − stack_z\| > 10 mm` |
| `dropped_in_transport` | lost before reaching the destination | never transported, and at the end `\|z_src − resting_z\| ≤ 5 mm` |
| `no_grasp` | never picked it up | `z_src` never reached resting height + one object height (80 mm for a 50 mm object) |
| `timeout_carrying` | lifted it and never let go | lifted, no release, and not on the table at the end |
| `timeout` | none of the above | — |

### 5.4 Classified results, 45k checkpoint on unseen layouts

`eval_mainmodel_45k_on_v228_queuesize0_final_report` - 500 trials (100 unseen layouts x 5 training pairs),
seed 0, k=0, fixed 130 ms latency. This is the only 500-trial run scored with the current classifier, so it is
the one I quote.

| pair | success | failures |
|---|---|---|
| red cylinder -> blue cube | **91%** | misplaced 8, released_in_air 1 |
| red cube -> blue cube | 69% | misplaced 21, no_grasp 6, released_in_air 3, knocked_off 1 |
| blue cube -> red cube | 65% | misplaced 25, released_in_air 5, dropped_in_transport 3, no_grasp 2 |
| blue cube -> red cylinder | 57% | misplaced 33, released_in_air 6, dropped_in_transport 3, no_grasp 1 |
| red cube -> red cylinder | 53% | misplaced 30, no_grasp 6, released_in_air 5, dropped_in_transport 5, knocked_off 1 |
| **total** | **67.0%** | misplaced 117, released_in_air 20, no_grasp 15, dropped_in_transport 11, knocked_off 2 |

Also flagged: `dst_toppled` on 14 trials, `third_disturbed` on 2.

Two things stand out.

**Destination geometry dominates.** The three cube destinations average 75%, the two cylinder destinations
55%. The cylinder's top is a 12.6 cm² disc against the cube's 16 cm² square, and that 20-point spread is
larger than any other effect I measured.

**Almost all failures are placement failures.** 137 of the 165 failures (83%) are `misplaced` or
`released_in_air` - the object reached the destination and was put down badly. Only 11 were lost in transport
and 15 never grasped.

Inspecting the replays of the `misplaced` episodes told me why, and it is not what the label suggests. The
gripper itself was more or less in the right place over the destination - roughly where it should be if it
were about to release an object held in the middle of the jaws. The problem is that it had not grasped the
source in the middle. It picked the object up off-centre and then placed it as though the grasp had been
perfect, so the object landed off by whatever the grasp offset was.

Put another way: the policy learned to put the **end-effector** above the destination, regardless of how it
was holding the source.

I think that is a consequence of my own data. The scripted planner always grasps the object at its centre, so
in every single training example "gripper over the destination" and "object over the destination" are the same
thing. The policy has no example where those two differ, so it learns the easier one. At test time the grasp
is not centred, the two come apart, and the object lands off by the offset.

**As a next step this is what I would have done:** vary the grasp offset in the scripted demos so the gripper
is sometimes off-centre while the planner still places the *object* correctly. The two objectives then stop
being the same thing in the training data, and the policy has to learn to align source to destination instead
of gripper to destination. The fix belongs in the data engine rather than in the policy, and I did not get to
test it.

Either way it is what Task 4 is built on.

### 5.5 The held-out pair

`put the red cylinder on the red cube`, left out of training entirely. I ran the policy on 100 layouts from
the training set and 100 from the unseen seed-228 set. **Both gave 0% success.**

Then I ran the classifier on the v228 result:

| outcome | n |
|---|---|
| `timeout_carrying` | 56 |
| `dropped_in_transport` | 28 |
| `misplaced` | 16 |

Three different stories, so I inspected the recordings. In all of them the policy picks up the correct red
cylinder and tries to place it on the **blue cube**.

That tells me my classifier was good enough for the five training pairs because there the failures are
placement failures, which it was built to attribute. It does not cover the failure modes that appear here:
there is no label for using the wrong object, so these 100 episodes got scattered across three buckets
according to incidental geometry.

But since every episode attempted the same wrong task, I could use the classifier to measure how well it
performed that wrong task. It runs offline on traces I had already recorded, so I just made it think the
destination was the blue cube — rebuild the trace with the destination and third object swapped, run
`analyse()` unchanged. It is clean here because both are same-sized cubes, so no tolerance shifts.

| label | destination = red cube | destination = blue cube |
|---|---|---|
| `success` | 0 | **68** |
| `misplaced` | 16 | 30 |
| `timeout_carrying` | 56 | 0 |
| `dropped_in_transport` | 28 | 0 |
| `released_in_air` / `stacked_dst_moved` | 0 | 1 / 1 |

So the policy attempted to stack the red cylinder on the blue cube in all 100 episodes and succeeded in 68% of
them, the same rate it gets on its training pairs. The stacking itself is fine, and it picks the right source
every time. What it has not learned is to use the colour to choose the *destination* — it goes to the cube it
has always put the cylinder on.

I attribute this to having too few pairs, not to the missing pair itself. The problem is not that the policy
never saw the red cylinder go on the red cube — it is that it *always* saw the red cylinder go on the blue
cube. With only one destination ever shown for that source, the policy simply learned that the red cylinder
goes on the blue cube, whatever the instruction says.

With a third colour it would be different. If training contained both `red cylinder -> blue cube` and
`red cylinder -> green cube`, the policy would see the destination for that source change between episodes,
and the only thing that tells it which one is the instruction. Having learned to read it there, it could then
generalise to the unseen `red cylinder -> red cube`. Dropping a pair out of only five removes that
opportunity: the remaining destination for that source is 100% predictable, so the language never has to be
used.

0% made me go back and question whether I had picked the right pair to hold out. I do not think it would have
mattered: any choice leaves its source with a single destination. Had I held out `blue cube -> red cylinder`
instead, the policy would have seen the blue cube always go on the red cube, and the same thing would have
happened in the other direction.

### 5.6 The reversal test

**474 of the 500 v228 trials** carry one of four outcomes: `success` (335), `misplaced` (117),
`released_in_air` (20) or `knocked_off` (2).

Each of those requires the transport milestone — the source was above the destination named in the
instruction. On its own that guarantees nothing, since the arm might have been passing over that point on its
way elsewhere (5.5 has exactly such a case).

For `success` and `knocked_off` the definitions themselves guarantee the object was put on the named
destination: both require the source within 10 mm of the height it would sit at resting on that object, and
its centre inside the destination's footprint at the release frame.

For `misplaced` and `released_in_air` the definitions do not guarantee it — all we have from them is the
transport milestone, and that can fire on a swipe past. So for those two I measured it instead. Distance from
the source's centre to the destination's centre at the release frame:

| | n | mean | max |
|---|---|---|---|
| misplaced | 117 | 27.5 mm | 44.9 mm |
| released_in_air | 20 | 44.3 mm | 84.8 mm |
| both | 137 | 30.0 mm | 84.8 mm |

Object centres are at least **10 cm** apart in our layouts, and the worst of these released 84.8 mm from the
destination it was asked for. So in every one of these trials the named destination was the closest object to
where the policy let go — it was trying to stack onto the right one.

So in 474 of 500 trials the policy picked up the object it was asked for and tried to put it on the object it
was asked to put it on. It learned the reversal semantics; what it did not learn is to do it accurately.

<!-- TODO: link the back-to-back reversal video here — same layout, red cylinder -> blue cube then
     blue cube -> red cylinder, same seed. -->

## 6. Async inference

I went with asynchronous inference. The mechanism is already described in 4.1 and 4.2, so this section is just
running the eval script with a non-zero queue threshold.

```bash
python eval_async.py --checkpoint outputs/train/main_250v3/checkpoints/045000/pretrained_model \
    --layouts data/layouts/eval_100_v228.csv --num-layouts 100 --pairs train --seeds 0 \
    --queue-threshold 10 --latency fixed --latency-ms 125 \
    --name eval_mainmodel_45k_on_v228_queuesize10_final_report
```

Inference on the 5090 is 120-150 ms, which at 30 Hz is 4-5 control steps, so k = 5 is in principle enough to
keep the queue from ever running dry. I used k = 10 to leave margin.

Latency is held fixed at 125 ms rather than measured per call, so the run is reproducible.

### 6.1 Results

Dead time is the fraction of control steps where the robot has no command to execute and holds its last one
while waiting for inference to finish. At k = 0 that happens at every chunk boundary.

Async does not take it to zero: the very first chunk of an episode still has to be waited for, with nothing in
the queue to execute meanwhile. The residual 1.11% is about 4 steps in a ~360-step episode, which is exactly
one inference at 125 ms - so what is left is the first chunk and nothing else.

500 trials each, same layouts, pairs and seed, latency fixed at 125 ms.

| | dead time | jerk at boundary | jerk overall | switches | success |
|---|---|---|---|---|---|
| k = 0 (sync) | **7.71%** | 110.2 | 110.2 | 7 | 67.0% |
| k = 10 | **1.11%** | **182.5** | 123.7 | 9 | 67.2% |

Dead time drops 7x and success is unchanged (67.0 -> 67.2).

The size of the problem scales with how slow inference is. The dead fraction at k = 0 is `L / (50 + L)`, so at
125 ms on the 5090 (L = 4) it is 7.4%, which is what the table shows. On my Mac M4, where the same model takes
650-700 ms per chunk, L is about 20 and that becomes **29%** - the arm would spend nearly a third of the
episode standing still. Async is worth more the worse the hardware is.

But the jerk column is the part worth reporting. **Async on its own made the arm less smooth, not more** - boundary
jerk went from 110 to 182. Removing the stall replaced it with a discontinuity in the command, because the
arriving chunk was computed from an observation that is now 4-5 steps old and it is switched in hard.

I measure jerk in a window around chunk boundaries, not over the whole episode. The episode average moves
110 -> 124 and understates the effect by about 5x.

### 6.2 Blending the overlap

The discontinuity is fixable. With k > L there are `k − L` steps where the old and new chunks overlap, so I
added `--blend linear` to ramp from one to the other across that window instead of switching hard. This is
also why k = 5 would not have been enough: the overlap would be 0-1 steps, with nothing to ramp over.

It works, and it more than undoes the regression. Same 500-trial protocol as 6.1:

| | dead time | jerk at boundary | jerk overall | switches | success |
|---|---|---|---|---|---|
| k = 0 (sync) | 7.71% | 110.2 | 110.2 | 7 | 67.0% |
| k = 10, hard switch | 1.11% | 182.5 | 123.7 | 9 | 67.2% |
| **k = 10 + linear blend** | **1.11%** | **99.3** | **108.4** | 9 | **67.2%** |

Boundary jerk goes 182.5 -> 99.3, **below** the synchronous baseline of 110.2, at identical dead time and
identical success. So async buys dead time and costs smoothness, and blending pays the smoothness back with
interest.

```bash
python eval_async.py --checkpoint outputs/train/main_250v3/checkpoints/045000/pretrained_model \
    --layouts data/layouts/eval_100_v228.csv --num-layouts 100 --pairs train --seeds 0 \
    --queue-threshold 10 --blend linear --latency fixed --latency-ms 125 \
    --name eval_mainmodel_45k_on_v228_queuesize10_blend_final_report
```

### 6.3 What I did not do

Inference latency itself is unchanged: p50 128 ms, p99 129 ms. I did no engine-level work - no `torch.compile`,
no quantization, no TensorRT - so there is no before/after latency number to report. What changed is the
fraction of wall time the arm spends not moving.

## 7. RECAP: collecting rollouts and corrections

### 7.1 What I built and why

RECAP needs the policy's own experience plus corrections on the episodes it got wrong. I had no operator, so I
automated the correction, and 5.4 told me what to automate: the gripper arrives in roughly the right place over
the destination and releases as if the object were centred in the jaws, when it is not.

That makes the fix arithmetic rather than a new skill. The arm commands the gripper, and while the object is
held the offset `delta = src_xy - gripper_xy` is constant. So aiming the gripper at `dst_xy - delta` puts the
*object* over the destination instead of the gripper.

### 7.2 The pipeline

I generated **100 fresh layouts (seed 322)** for collection and kept the seed-228 set as held-out evaluation.
So nothing in the pipeline ever sees the layouts the final numbers are measured on: the original policy was
trained on seed-3 layouts, the rollouts and corrections come from seed 322, and the value function is fitted on
those rollouts. Seed 228 is only ever used to evaluate.

For each layout x training pair, two passes:

**Pass 1** rolls out the policy with synchronous inference (`k = 0`) and classifies the episode at the end.
Recording stops 30 frames (1 s) after the release event — far enough past the gripper opening to capture the
release action itself, but short of the flailing or waiting that follows. The episode itself keeps running for
the full 4 s settle window, because that is what the success check needs.

Because inference is synchronous the arm stalls at every chunk boundary, and those frames are *excluded from
the recording* rather than trimmed afterwards: a held pose with a repeated action is not a policy decision and
would teach the policy to stand still.

**Pass 2** runs only if the classifier returned `misplaced` or `released_in_air` — the failures where the
object did reach the destination. It replays pass 1's recorded actions up to the transport event, with no
inference at all, then hands over to the IK planner, which drives the gripper to the corrected predrop and
down to the drop using the same place segments as the Task 1 scripted demo. The result is recorded as a second,
complete episode, prefix included.

### 7.3 What I got

```bash
python collect_rollouts.py \
    --checkpoint outputs/train/main_250v3/checkpoints/045000/pretrained_model \
    --layouts data/layouts/eval_100_v322.csv \
    --out data/lerobot/v322_data_for_recap --queue-threshold 0
```

500 rollouts (100 layouts x 5 training pairs):

| outcome | n |
|---|---|
| success | 308 |
| misplaced | 150 |
| released_in_air | 21 |
| no_grasp | 10 |
| knocked_off | 8 |
| dropped_in_transport | 3 |

171 of those were correctable. **156 corrections were accepted and 15 rejected** — a correction is kept only if
the classifier calls the corrected episode a success, and every rejection came back `knocked_off`, meaning the
object was placed inside the footprint but did not stay. 91% accept rate.

Final dataset: **656 episodes, 172 999 recorded frames**, mean 264 frames per episode. Collection took 77
minutes.

### 7.4 Reward

I used a reward close to the paper's. For a successful episode the return at frame `t` is minus the number of
frames until the stack appears; for a failed one it is minus the number of frames until the end of the episode,
minus a constant `C`:

```
success:  R_t = -(stacked_step - t)
failure:  R_t = -(n_frames - t) - C
```

normalised by the longest episode so everything lands in (-1, 0).

`stacked_step` is the first frame where the stack detector fires — object inside the destination footprint, at
stack height, not moving, jaws open. The same detector evaluated at the final frame is what decides success, so
the reward and the success criterion are one condition checked at two different times.

**`C = 250`**, because a success takes roughly 250 frames. That way an episode that fails on the first step
still scores worse than any success: it saves the elapsed-time penalty but pays a cost about equal to doing the
whole task.

That matters here because failures are not reliably longer than successes. Recording stops 30 frames after the
release, and a `misplaced` episode releases just like a success does, so it gets cut at the same point.
Measured over the 500 rollouts:

| | n | episode length | stack appears at |
|---|---|---|---|
| successes | 464 | 238-308 | 210-293 (median 239) |
| failures | 192 | 213-490 (median 253) | - |

### 7.5 The value function

Two options. The faithful one is to reuse SmolVLA — same images, state and instruction through its VLM, with an
MLP head predicting the value. That is what the paper does, but it costs real compute and a flat result would
have been uninterpretable: poorly defined reward, uninformative VLM features, or undertrained head, with no way
to tell which. I had time for one run.

So I used the fact that we are in simulation and gave the value function the world state instead:

| input | dims |
|---|---|
| source / destination / third pose | 7 each |
| `src_pos - dst_pos` (placement error) | 3 |
| gripper position + orientation | 3 + 4 |
| `src_pos - gripper_pos` (grasp offset) | 3 |
| robot state (joints + gripper opening) | 7 |
| shape flags (source cube?, destination cube?) | 2 |

43 inputs, 2x64 MLP with dropout, 201 bins. 20k parameters, under a minute to train.

**Poses are ordered by role, not by object** — source first, then destination, then the remainder. That is the
language conditioning: the instruction is always "put the {source} on the {destination}", so the ordering
encodes it and there is no text to tokenise.

This is a deviation — the value function sees privileged state no real robot would have — but I decided it was
acceptable for two reasons.

It does not limit where the result applies. The value function is only used during training, where I have full
access to the simulator state anyway. At deployment it is not involved at all: the policy is just given
`Advantage: positive` and run.

And it is not the part of the paper being tested. RECAP's idea is not "train a value function on images", it is
"improve the policy by giving it this feedback". How the feedback is produced is an implementation detail; what
has to work is the mechanism that turns it into better behaviour.

### 7.6 Training it

```bash
python train_value.py --dataset data/lerobot/v322_data_for_recap \
    --exclude-corrections --hidden 64 --dropout 0.3 --patience 20
```

Cross-entropy against the two-hot return, split by layout, early stopping on validation loss.

**I excluded the corrections.** A correction replays its parent's bad grasp and then succeeds, so the same
state appears twice with opposite returns and the value function can only fit the average. With them in, it
could not separate a good grasp from a bad one at all; without them it could.

Mean value at the grasp orders the outcomes correctly over 490 episodes:

```
success -0.22   knocked_off -0.30   misplaced -0.37   released_in_air -0.43   dropped_in_transport -0.52
```

**But I am not convinced this is a meaningful value function.** On layouts it trained on it ranks a success
above a failure about 90% of the time; on held-out layouts, 66%, and that is measured on a 25-episode
validation split where the error bar is roughly ±15 points. The ordering above is the stronger evidence, and
it is still only an average over classes. The binding limit is that 656 episodes give ~650 labels, not 173k —
every frame in an episode shares one outcome.

### 7.7 Putting it together

The authors fit the value function on the rollouts **and** the original demonstrations. I fit it on the
rollouts only. The idea was that it should learn the value and generalise to any scene, and I did not want it
overfitting to the large demo set — it already overfits very fast on what it has.

Then I measured the distribution of its predicted advantage over the v322 rollouts to derive the threshold,
per pair, at the 30th percentile:

```bash
python label_advantage.py --dataset data/lerobot/v322_data_for_recap \
    --values outputs/value/nc_h64_dropout_only
```

| pair | eps |
|---|---|
| red cube -> red cylinder | -0.0204 |
| blue cube -> red cylinder | -0.0164 |
| blue cube -> red cube | -0.0115 |
| red cube -> blue cube | -0.0078 |
| red cylinder -> blue cube | -0.0039 |

I derived it from the v322 rollouts only, not from the full `main_250v3` + v322 mixture. Same reasoning as for
the value function: the layouts are drawn from the same distribution, so the advantage over v322 should be
about the same as over the main set, and a threshold estimated on one should transfer to the other.

Per pair rather than global, because the pairs differ in difficulty and a single threshold would make the
indicator partly encode *which pair it is* instead of which action was taken. The resulting rate is 70%
positive within every pair by construction, and across outcome classes it comes out ordered: 81% of frames in
successful episodes are positive, 55% in `misplaced`, 32% in `no_grasp`.

Then I extended the training script to append the advantage text to the instruction — `Advantage: positive` or
`Advantage: negative`, per frame — and trained from the base SmolVLA checkpoint on the mixture of
`main_250v3` and the v322 rollouts and corrections, with the demonstrations forced positive.

**45 000 steps at batch 32, the same budget as the Task 2 run**, and from the same starting checkpoint. So the
comparison below is not confounded by training length or initialisation; the differences are the extra data and
the advantage text.

### 7.8 Result

Evaluated on v228, same protocol as the baseline (500 trials, seed 0, k=0, fixed latency). I ran the
conditioned checkpoint twice: once asked for `Advantage: positive`, which is how it is meant to be run, and
once on the bare instruction.

| pair | baseline | RECAP, tagged | RECAP, no tag |
|---|---|---|---|
| red cylinder -> blue cube | 91% | 68% | 68% |
| red cube -> blue cube | 69% | 54% | 47% |
| blue cube -> red cube | 65% | 54% | 53% |
| blue cube -> red cylinder | 57% | 36% | 31% |
| red cube -> red cylinder | 53% | 35% | 40% |
| **total** | **67.0%** | **49.4%** | **47.8%** |

**It is 18 points worse than the baseline.** The failure mix moved with it: `misplaced` 117 -> 168,
`no_grasp` 15 -> 41, `released_in_air` 20 -> 32. So the policy is worse at every stage, not just at placement.

**And the conditioning did nothing.** Tagged and untagged differ by 1.6 points, where the standard error on a
500-trial rate near 50% is about 2.2 points. The outcome distributions are near-identical too — 247 vs 239
successes, 168 vs 170 `misplaced`. Whatever cost the policy those 18 points, it was not the absence of the tag
at inference, and asking for high advantage does not recover them. The policy learned to ignore the indicator.

This is a negative result. Two things I would attribute it to, and I cannot separate them.

**SmolVLA was never pretrained with a label like this.** `Advantage: positive` is not language it has grounding
for — it is four tokens appended to an instruction, in a format nothing in its pretraining resembles. The
paper's model was trained with the indicator as part of its recipe; mine meets it for the first time during a
45k-step fine-tune, and it may simply be diluting the instruction rather than conditioning on anything.

**The value function is probably too weak.** In principle it had an easy dependency to learn — grasping off
centre leads to failure — and the per-outcome ordering in 7.6 shows it did pick that up on average. But it
barely generalises to unseen states, so the indicator it produced was close to noise where it mattered. An
indicator carrying no information is exactly what a policy should learn to ignore, and tagged-versus-untagged
says that is what happened. Whether the ordering it did learn failed to translate into a useful per-frame
signal, or whether the signal was fine and the conditioning never took, I cannot tell from one run.

A third possibility I cannot rule out: the data itself. The rollout half of the mixture is 38% failure frames,
so some of the drop may simply be the policy being taught worse behaviour, independently of the conditioning.

Underneath all three is that I ran out of time. This was the last part of the project, and it has the most
moving pieces — reward, value function, threshold, conditioning, data mixture — each of which needs testing and
tuning on its own before they can be expected to work together. I got one pass at each and no opportunity to
iterate. More time here would have made a real difference, and I would not read this result as evidence about
the method so much as about how far I got with it.

## 8. Limitations

Each of these is named where it comes up; this is the summary.

- **The policy aims the gripper, not the object.** The scripted planner always grasps at the object's centre,
  so "gripper over the destination" and "object over the destination" are the same thing in every training
  example. The policy learned the easier one. 83% of its failures are placement failures. (5.4)
- **It does not use the destination colour.** On the held-out pair it stacks the cylinder on the blue cube at
  68% — its normal rate — instead of the red cube it was asked for. With five pairs the destination for a given
  source is deterministic in training, so the instruction never has to be read. (5.5)
- **The classifier only covers the failures the training pairs produce.** It attributes placement failures
  well, which is what I built it for, but it has no notion of the policy using the wrong object. On the
  held-out pair that meant the labels were unusable and I had to watch the recordings to find out what the
  policy was actually doing. (5.5)
- **The value function does not generalise.** It ranks a success above a failure ~90% of the time on layouts it
  trained on and ~66% on held-out ones, measured on only 25 validation episodes. 656 episodes give ~650
  labels, not 173k — every frame in an episode shares one outcome, and no amount of regularisation
  manufactures more. (7.6)
- **The value function is privileged.** It reads simulator state, so this part would not transfer to a real
  robot as built. (7.5)
- **The RECAP drop cannot be attributed to one cause.** The conditioning demonstrably did nothing — tagged
  and untagged score within noise of each other — but whether the 18-point drop came from the conditioning,
  the weak value function or the failure-heavy data mixture is not separable from one run. (7.8)
- **Tightly packed scenes never appear.** Minimum 10 cm between object centres in every layout, training and
  evaluation alike, so I have no evidence about crowded scenes. (2.1)
- **No engine-level inference work.** No compile, quantization or TensorRT, so raw latency is unchanged at
  p50 128 ms; what I improved is dead time. (6.3)

## 9. What I would do next

**1. Collect demonstrations with varied grasp offsets.** Deliberately grasp the object off-centre sometimes,
while the planner still places the *object* correctly. That breaks the coincidence between "gripper over the
destination" and "object over the destination", so the policy has to attend to where the object actually is.
It is a change to the data engine, not the policy, and it targets the dominant failure directly: `misplaced`
and `released_in_air` together are 137 of 165 failures, so removing most of them would put success around 90%.

**2. Add a third or fourth colour, then hold out a pair.** With more pairs per source the destination stops
being deterministic, so the policy has to read the instruction to choose between them. Only then does holding
a pair out actually test generalisation — as it stands the held-out result measures a data-structure artifact
rather than the policy's ability to compose.

**3. Give RECAP the time it needs.** Every component has to be right at once — reward, value function,
threshold, conditioning, and the eval protocol. A weak link anywhere produces a number that cannot be
interpreted, which is where I ended up: the value function barely generalises, and the conditioning had no
measurable effect on the policy. Done properly it is a multi-day piece of work on its own, not
something to attach to the end of the other three tasks.

## 10. Results video

<!-- TODO: replace each placeholder with the recording. -->

**1. Reversal test** — inference on mac, that is why you see stalls, in sim those are discrarded, 

https://drive.google.com/file/d/1BCslWGf0QaDrXPDqU3CeGEH4FGeThHFr/view?usp=sharing



**2. A success** on an unseen layout — baseline competence. async at k =0 , so sync

https://drive.google.com/file/d/1VhCmgkYs9nd3KV2iFda2ENr3rj-5hAvE/view?usp=sharing

**3. A `misplaced` failure** — the dominant failure mode.

https://drive.google.com/file/d/15Egfi2IVprh9mfuOUuBGw2U-atGRIReE/view?usp=sharing

**4. Async inference** — the same policy at `k = 10`.

[> _[video: sync vs async, side by side or consecutive]_](https://drive.google.com/file/d/1XNVqIZhR-83-vfiEBHRoTmDCaPDg3lGV/view?usp=sharing)

**5. The held-out pair** — it picks up the correct red cylinder and stacks it neatly on the blue cube, which is
the finding in 5.5 and is immediately obvious on video in a way the table is not.

https://drive.google.com/file/d/1RVetihWz3ehSlli1WBFdcEg7WPgAQEsl/view?usp=sharing
