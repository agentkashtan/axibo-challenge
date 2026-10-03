# AXIBO VLA Challenge: Data Collection, Fine-Tuning & Real-Time Inference

**Target Model:** AgileX PiperX (6-DOF + gripper), https://github.com/agilexrobotics/piper_isaac_sim/tree/master/piper_x_description/urdf

**Simulation Environment:** Genesis (please visit the [official repo](https://github.com/Genesis-Embodied-AI/Genesis) for install instructions)

---

## Introduction

At AXIBO, we are building manipulation policies that run on real hardware in unstructured environments. This challenge evaluates your ability to take a Vision-Language-Action model from nothing to something that actually works: collecting your own data, fine-tuning a checkpoint on it, evaluating it honestly, and getting it to run smoothly in the loop.

We are deliberately not giving you a dataset. In our experience the data engine is where most VLA projects succeed or fail, so we want to see how you build one. You are free to use any open checkpoint and any training framework, but everything must be collected and evaluated in Genesis on the provided PiperX asset.

---

## The Manipulation Task

Everything below is built around one task: **language-conditioned stacking.**

* **Scene:** A tabletop with three objects, a **red cube**, a **red cylinder**, and a **blue cube**. Their positions and orientations are randomized every episode. The arm starts from a fixed home pose.
* **Instruction:** A single sentence of the form *"put the {source} on the {destination}"*, where source and destination are two different objects in the scene. For example, *"put the red cylinder on the blue cube"*.
* **Why these three objects:** Two share a colour and two share a shape, so neither attribute alone identifies anything. Every instruction requires binding colour and shape together, twice, and assigning each referent to a different role.
* **Success:** The source object is resting on top of the destination object and stays there for 2 seconds after the arm retreats, with the destination still upright and the third object undisturbed. Choose your tolerances and state them.
* **Pairs:** Three objects give six ordered source-destination pairs. **Hold one pair out entirely** from data collection and use it as your generalization test.

---

## The Tasks

### Task 1: Build a Data Engine

Collect your own demonstration dataset for the stacking task above using a scripted policy.

* **Method:** Use Genesis's [batched IK](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/tutorials/batched_IK.py) (or any other IK method you prefer) to script the motion and collect across many randomized environments in parallel. Teleoperation is not expected, automated collection is the point.
* **Coverage:** Five of the six ordered pairs, balanced. The sixth is held out. Tell us which one you held out and why.
* **Export:** Save to a standard format (LeRobot is fine, or justify your own).
* **Challenge:** You are choosing the action space, the observation setup, the randomization ranges, and how many episodes to collect. Document each of these decisions and why you made it. We care more about this reasoning than about the episode count.

### Task 2: Fine-Tune and Evaluate Under a Declared Protocol

Fine-tune an open VLA checkpoint on your own data and report how well it works.

* **Model:** Pick any open VLA checkpoint. Tell us why you picked it.
* **Evaluation:** Report success rate per ordered pair, with a **stated trial count, seed list, and failure taxonomy** (wrong object grasped, missed grasp, grasped and dropped, stacked on the wrong destination, knocked the destination over, never reached, etc.). An aggregate number with no protocol behind it tells us very little.
* **The reversal test:** On a fixed scene, run *"put the red cylinder on the blue cube"*, then *"put the blue cube on the red cylinder"*. Same objects, same positions, opposite action. This is the cleanest evidence that the policy is reading the instruction rather than the scene, so report it separately.
* **Generalization:** Report performance on your held-out pair.

### Task 3: Make Inference Smooth, Not Just Fast

Naive synchronous execution stalls the arm at every chunk boundary while the next chunk is computed. Remove that stall.

* **Methods:** At least one of the following: asynchronous inference that overlaps compute with execution, temporal ensembling across overlapping chunks, real-time chunking, or engine-level work such as `torch.compile`, TensorRT, or quantization. More than one is welcome.
* **Deliverable:** p50 and p99 inference latency before and after, a smoothness metric (joint jerk or velocity discontinuity at chunk boundaries), and evidence that task success rate did not degrade.
* **Challenge:** Latency and smoothness are easy to improve if you are willing to break the policy. Show us you didn't.

---

### Task 4 (Optional, but Strongly Recommended): Learn From Your Own Experience with RECAP

Improve the Task 2 policy using its own rollouts rather than more demonstrations, following the core mechanism of [RECAP](https://arxiv.org/abs/2511.14759) (π*0.6). Tasks 1–3 are the bar.

* **The loop:** Deploy your fine-tuned policy autonomously and log rollouts with rewards derived from simulator state. On failed rollouts, hand control to your Task 1 scripted policy to produce a correction. Train a value function on the resulting mixture of demonstrations, autonomous experience, and corrections, then fine-tune the policy conditioned on binarized advantage estimates from that value function, and condition on high advantage at inference.
* **Note on scope:** RECAP's first stage is offline-RL pre-training of a model that supports advantage conditioning. No open checkpoint provides this, so you will need to add the conditioning yourself and skip that stage. We know this, and we are not expecting the paper's results.
* **Deliverable:** Success rate against your Task 2 baseline under the same evaluation protocol, plus how you defined reward, how you triggered corrections, and what your value function actually learned. A correct implementation with a negative result and a clear account of why is a good submission. A reported improvement you cannot attribute to anything is not.
* **Challenge:** Rewards in simulation are cheap, which makes it easy to write a reward the policy can satisfy without doing the task. Knocking the destination over and dropping the source where it used to be should not score. Tell us how you checked for that.

---

## Technical Stack & Constraints

* **Simulation:** Data collection and evaluation entirely native to Genesis.
* **Training:** Any framework. Any open checkpoint.
* **Compute:** Use whatever you have available. We are not evaluating how much compute you have access to, and if access is a problem please reach out, we can help.
* **Timeline:** Roughly one week for Tasks 1–3, and a second week if you take on Task 4. Tell us how you spent the time, and if you ran out of it, what you would have done next.

---

## Provided Assets

1. **PiperX Asset Pack:** URDF and meshes for the AgileX PiperX arm.

That is all. The dataset, the scene, and the evaluation harness are yours to build.

Please find the PiperX model here: https://github.com/agilexrobotics/piper_isaac_sim/tree/master/piper_x_description/urdf

You may use any AI coding tools for this challenge.

---

## Submission Requirements

Submit a private GitHub repository containing:

1. **Implementation Code:** Clean, documented Python for the collection pipeline, the training run, the evaluation harness, and the inference loop. Please add [@RVSagar](https://github.com/RVSagar) and [@ax-anoop](https://github.com/ax-anoop) to the repo.
2. **Trained Policy:** Saved weights, or a clear pointer to them if the checkpoint is large.
3. **README / Technical Report:**
   * Your data engine decisions (action space, observations, randomization ranges) and the reasoning behind them.
   * Your evaluation protocol, your success tolerances, results per pair, and failure breakdown, including the reversal test and the held-out pair.
   * What you did for inference, with the before/after latency and smoothness numbers.
   * If you attempted Task 4: your reward definition, correction trigger, value-function training, and what changed.
   * A short summary of training hyperparameters.
4. **Results Video:** Screen recordings of the policy in the Genesis viewer, including the reversal test run back to back on a fixed scene.

---

## Evaluation Criteria

* **Data Engine Design:** Whether your randomization and action space choices reflect an understanding of what the policy needs to see in order to generalize.
* **Evaluation Rigor:** Whether we can trust your numbers. A well-specified protocol on a mediocre policy beats an impressive number we cannot reproduce.
* **Real-Time Thinking:** Whether you treated inference as a control-loop problem rather than a training-curve problem.
* **Reading a Paper Into Code (Task 4):** Whether you identified the load-bearing mechanism in RECAP and implemented it faithfully, independent of whether it improved your numbers. We weight an honest failed attempt here above a clean submission that skipped it.
* **Engineering Quality:** Effective, readable use of Genesis and your chosen training stack.
* If a task is not feasible or you cannot solve it, please explain to us the limitations of your solution and/or why it could not be done. Learning from engineering failure is a big part of this field and we are interested in hearing your thought process.

---

## Questions

* Please open an issue on this repo (preferred), e-mail [sagar@axibo.com](mailto:sagar@axibo.com) or reach out on Discord (@rvsagar) for any questions or clarifications regarding this challenge.
