# Agent Handoff: Track A — produce a training dataset for SmolVLA

| | |
|---|---|
| **Handoff version** | **H6** |
| **Supersedes** | H5 and everything before it |
| **Plan reference** | `implementation_plan.md` V2.1 — **partly superseded, see §3** |
| **Conventions reference** | `conventions.md` v2 — **partly wrong, see §6** |
| **Mode** | **Simulation only.** No real-robot code, no hardware access. |

This handoff is written to be read cold. §1 is the goal, §2 the vocabulary,
§3 what exists, §4 the audit, §5 the plan. If you read nothing else, read §1,
§4.5 and §5.

---

## 1. What this project is, and what "done" means

**Someone else is going to fine-tune a vision-language-action model called
SmolVLA. Our job is to hand them a training dataset. We do not train it, we do
not evaluate it.**

The dataset is episodes of a simulated robot arm picking up a cube and putting
it in a bin. Each episode is a sequence of frames; each frame is a camera image,
the robot's joint positions, the joint positions it was commanded to move to,
and a text instruction. The model learns to map (image, joints, instruction) →
next joints.

The episodes come from human video. Apple's **EgoDex** dataset contains
recordings of people doing tabletop tasks with hand-pose tracking. We convert a
human's hand motion into a gripper trajectory, re-anchor that trajectory onto a
cube in our simulator (this is the **MimicGen** method), run it in physics, and
keep the attempts where the cube ends up in the bin.

**The point of using human video at all** is a comparison the project wants to
make: dataset **A** (from EgoDex) vs **B** (from human teleoperation of the real
robot) vs **C** (a mix). We can only deliver **A** — B needs teleop recordings
from hardware we do not have. Say this out loud to the team early; the A/B/C
comparison is currently assumed and is not deliverable.

### Status in one line

Everything that turns human video into executable robot trajectories **works**.
Nothing that turns those trajectories into a dataset **exists**. No image has
ever been rendered; no dataset file has ever been written.

---

## 2. Vocabulary

Terms used throughout, in the sense this repo uses them.

| Term | Meaning here |
|---|---|
| **Episode** (EgoDex) | One recording of a person doing a task, ~3 s, hand joint poses at 30 Hz in an HDF5 file |
| **Source demo** | One grasp-and-place motion extracted from an episode and converted to gripper poses, saved as `data/retargeted/egodex_<episode>_<k>.npz`. One episode can yield several |
| **Library entry** | A source demo plus the metadata needed to re-anchor it: where the grasp and release happened, and a 2-number `index` (`dist_xy`, `dz`) describing how far and how high the object travelled |
| **Scene** | A randomly sampled cube position and rotation inside the `pick_zone` |
| **Retarget / re-anchor** | Rigidly transforming a source demo so its grasp lands on *this* scene's cube and its release lands on the bin. This is MimicGen, specified in `conventions.md` §8, implemented in `common/mimic.py` |
| **Trial / attempt** | One (library entry, scene) pair executed in physics. Succeeds or fails |
| **Rendering** | Taking a camera picture of the simulated scene at a moment in time, producing an RGB image array. MuJoCo does this offscreen via `mujoco.Renderer`. **Nothing in this repo does it yet** |
| **MuJoCo** | The physics simulator. A *model* (`MjModel`) is the static scene; *data* (`MjData`) is the evolving state; `mj_step` advances time by one `timestep` |
| **LeRobot** | HuggingFace's robot-learning library. Defines the on-disk dataset format we must produce, and is the framework SmolVLA is implemented in. Installed, version 0.4.4 |
| **SmolVLA** | The model being fine-tuned. `lerobot.policies.smolvla` |
| **tcp** | Tool centre point — the gripper's reference frame. `conventions.md` §2 defines it |
| **`table` frame** | The world coordinate frame. Origin on the table surface. MuJoCo's world frame *is* this frame |

---

## 3. What exists, and what each piece does

Run everything with `uv run`. Python 3.12, all dependencies installed.
`MUJOCO_GL=egl` works on this machine (`osmesa` is a verified fallback).

### 3.1 The pipeline that works

```
data/download_egodex.py   downloads EgoDex           -> data/egodex_raw/*.hdf5   (1077 files)
data/retarget.py          human hand -> gripper path -> data/retargeted/*.npz    (179 demos)
data/build_library.py     adds anchors + index       -> data/library/egodex_v1.json
sim/feasibility.py        "can the arm do this?"     -> sim/plots/feasibility_*.json
sim/validate_library.py   "does it work in physics?" -> data/library/*_validated.json
```

### 3.2 Supporting modules (`common/` — all sound, see §4.1)

| File | Role |
|---|---|
| `transforms.py` | Pose maths. Coordinate frames, the human-hand→gripper construction, and the two free-variable operators `spin_about_approach` / `tilt_about_jaw` |
| `segment.py` | Finds grasp/release in a hand-aperture signal; applies the step-4 quality filters |
| `demos.py` | Reads `.npz` source demos. The only place that knows the §7.2 file schema |
| `mimic.py` | **The MimicGen transform** (`conventions.md` §8) |
| `ik.py` | Inverse kinematics via `mink`. Converts a desired gripper pose to joint angles. Includes a "seed bank" (§4.1) |
| `safety.py` | Enforces the `conventions.md` §10 limits identically in sim and (future) real |
| `joints.py` | `JointIndex` — maps joint names to array slots. **Never hardcode joint order** |
| `config.py` | Loads `config/*.yaml` |
| `robot_interface.py` | The abstract interface `SimRobot` implements |

### 3.3 Simulation (`sim/`)

| File | Role |
|---|---|
| `build_models.py` | **Generates** `robot_ik.xml`, `scene.xml`, `scene_headless.xml`. Never hand-edit those |
| `sim_robot.py` | `SimRobot` — owns one MuJoCo simulation; `stream()` executes a joint trajectory |
| `execute.py` | 938 lines. Turns a library entry + scene into an executable path, then runs it. **The messiest file; see §4.5** |
| `reachability.py` | **Generates** the IK seed bank and a map of where the arm can reach |

Regenerate after any scene or calibration change:
```sh
uv run python sim/build_models.py    # after ANY scene change
uv run python sim/reachability.py    # also, if T_table_robot moved
```

### 3.4 What does NOT exist

* **Any rendering.** No `mujoco.Renderer` anywhere.
* **Any dataset writer.** No `data/lerobot/`, no LeRobot export.
* `policy/`, `perception/` — not needed for our deliverable.

---

## 4. The audit

### 4.1 Sound — keep this, it is expensive to re-derive

These carry measurements that took real work. Do not rewrite them casually.

* **`common/transforms.py`, `common/mimic.py`** — the §8 object-centric
  transform, well tested (`tests/test_mimic.py`, `tests/test_demo_plan.py`).
* **`common/ik.py` + the seed bank.** Differential IK seeded from HOME recovers
  only **3/60** reachable poses, because this arm's top-down configurations live
  near `joint1=-1.2, joint2=+2.9` while HOME is all zeros. Seeding from a
  harvested bank of nearest-by-position configurations recovers 7/7. Never call
  bare `solve()` with a HOME seed.
* **`common/segment.py`.** The grasp detector. The non-obvious finding: a
  *relaxed* human hand sits at a **small** finger aperture, so "hand is closed"
  fires on frame 0 in **79%** of episodes. Only below-threshold runs with an open
  frame on *both* sides count. 17 tests.
* **`common/safety.py`** — the §10 limits map to real hardware (LeRobot's
  `max_relative_target`). These are the one set of constants that are *not*
  invented.
* **Gravity compensation** (`build_models.py` sets `gravcomp=1`). Without it the
  tcp sags 21 mm at HOME and no commanded pose is ever reached. Flagged for
  hardware: if the real driver lacks gravity feedforward, this must come out.
* **The generated-model discipline.** `build_models.py` asserts on every build
  that the headless scene is dynamically identical to the full one.

### 4.2 Bugs fixed this session — do not re-introduce

| Bug | Fix |
|---|---|
| One demo per episode max: rejected 572 of 1077 episodes for containing *more than one* grasp | `segment_all` emits one demo per closure. **106 → 179 demos** |
| Grasp-speed filter measured 3D speed at the aperture-crossing frame, which is mid-descent | Horizontal speed at the *bottom* of the descent. Threshold back to its original 0.15 |
| Top-down gripper constraint | `tilt_about_jaw`, searched per segment. **99 → 115 feasible** |
| `sim/feasibility.py` ran serially on 1 of 22 cores | Parallel. 13 min → ~2 min |
| `validate_library.py` parallelised over *candidates*, so workers idled | Parallel over (candidate, trial) |
| 16 workers OOM-killed a 32 GB machine | Visual meshes (968k verts) stripped from batch models. **2646 → 345 MB per worker** |
| `stream()` silently ran at 29.41 Hz when asked for 30 | `steps_per_cycle` raises instead of rounding |

### 4.3 Known bugs NOT fixed

1. **`_reason_key` in `validate_library.py` destroys the diagnostic data.** It
   collapses `"cube at (0.31,-0.22) outside the bin footprint"` to
   `"cube_not_in_bin"` *before* anything is saved. The dominant failure mode
   therefore cannot be diagnosed from the saved output. **Fix this first** — see
   §5.1.
2. **The bin sits 78 mm outside the safety box.** `config/safety.yaml`
   `workspace_box.y` stops at −0.45; the bin's outer edge is at −0.528. Some
   trajectories are rejected for entering a region the task requires. Either the
   box or the bin is wrong.
3. **`unsafe_trajectory` is undiagnosed** — 437 of 1980 trials. H5 claimed the
   cause was known (a coarse-density screen). **That was tested and disproved**:
   re-screening at full density changed the step-6 result by exactly zero demos.
   Do not repeat that investigation.

### 4.4 Unverified assumptions — the real rot

`config/skills.yaml` has **68 values**. Some were measured; many were guessed
before any measurement existed and then built upon. Treat every one as
unverified unless it has evidence in its comment or in
`docs/conventions_proposals.md`.

**Measured and trustworthy:** the `scene` block (from the reachability map),
`ik.posture_weight`, `ik.seed_k`/`n_restarts`, the `retarget.grasp` and
`retarget.filters` thresholds, `exec.approach_height`, `exec.place_clearance`.

**Guessed, and known to be costing us:**

| Value | Problem |
|---|---|
| `library.max_index_offset: 0.12` | **Measured correlation with success: −0.048.** It rejects 32 of 179 demos on a quantity that does not predict the outcome |
| `validation.min_success_rate: 0.8` | A *runtime retrieval* rule (`conventions.md` §7.3 says "Runtime uses only…") applied as a *data generation* gate. Turned 649 usable episodes into "9 entries" |
| Gate G4's "50 library entries" | Invented before any yield was known. Measures the wrong thing entirely (see §5) |
| `scene.bin.height: 0.08` | The rim creates the `dz` mismatch that H5 called the one structural limit. `move_to_spot` (already in `conventions.md` §11, `scene.spot` already in config) removes it |
| `scene.pick_zone`, `scene.bin.center` | Fitted to a reachability map whose bounds **may be artifacts of the sampling grid** — the map's usable region starts at x=0.15, which is the first column the grid sampled. Nobody probed further in |
| `exec.grasp_tcp_offset: 0.0` | Pins the commanded grasp to the cube's exact centre, discarding ~3 cm of legal grasp height |

### 4.5 The architectural problem

**The pipeline is shaped like a screening system. The goal needs a generator.**

`sim/feasibility.py` (step 6) and `sim/validate_library.py` (step 9) both do the
same thing: take a library entry, retarget it onto a scene, search the wrist
angles, solve IK, and see whether it works. Step 6 throws the result away and
records a boolean. Step 9 throws the result away and records a rate. A generator
would do the identical work a third time and finally keep something.

That is also *why* the gates in §4.4 exist: each stage must decide what to pass
to the next. **If you run the work once and save an episode whenever it
succeeds, there is nothing left for the gates to decide.** This is MimicGen as
actually specified — generate many, keep the successes.

Collapsing this deletes more code than it adds, and removes the thresholds that
§4.4 shows are measuring the wrong things.

### 4.6 Test coverage

161 passing, 0 failing (`uv run pytest tests/` — **not** bare `pytest`, which
tries to collect `third_party/`).

Good coverage: `transforms`, `mimic`, `segment`, `safety`, `joints`, `ik`, and
`test_demo_plan.py` (the synthetic end-to-end pipeline test).

**`sim/execute.py` is 938 lines and the least covered relative to its
complexity.** `fit_yaw_offsets`, `screen_pose_path`, `run_demo` and
`_fit_rate_limits` have no direct tests.

**`tests/test_demo_plan.py` is the most valuable file in the repo.** It runs the
whole transform pipeline on a *synthetic* demo built to match the scene exactly,
so a failure is the code rather than the data. Debugging straight from real-data
rejections produced **three wrong diagnoses in a row**; this harness found the
real bugs. Extend it rather than staring at rejection counts.

---

## 5. The plan

### 5.0 Current measured state, and what is stale

```
1077 EgoDex episodes
  ->  179 source demos        (data/retargeted/,       CURRENT)
  ->  115 feasible            (sim/plots/feasibility_*, CURRENT)
  ->   99 library candidates  (data/library/egodex_v1.json,  STALE — built from the older 99-demo screen)
  ->    9 "validated" entries (data/library/*_validated.json, STALE — and the wrong metric, see below)
```

**The on-disk library files are inconsistent with the feasibility file.** Rebuild
them (`data/build_library.py`) before using them.

The last physics run scored 99 candidates × 20 scenes = 1980 trials: **649
succeeded**. The "9 entries" figure counts *library entries whose success rate
exceeded 0.8*, which is not a quantity anyone needs. **649 successful episodes
is the real number, and it predates the tilt change that took feasibility from
99 to 115.**

### 5.1 Phase 1 — stop throwing away the answer (half a day)

**Goal: one script that produces training episodes, replacing three that produce
verdicts.**

1. **Record what failures actually are.** Before changing anything, make the
   per-trial record keep the full reason string *and the final cube pose*.
   Currently `_reason_key` discards both. Without this, `cube_not_in_bin` — 698
   of 1980 trials — cannot be diagnosed.
2. **Write `data/generate_dataset.py`:**
   ```
   for (entry, seed) in work_queue:
       scene  = sample_scene(seed)              # cube xy + yaw in pick_zone
       rt     = mimic.retarget(entry, scene)    # conventions.md §8
       path   = fit_yaw_offsets(rt)             # free wrist spin + tilt
       q_traj = ik.solve_path(path)
       ep     = robot.stream(q_traj, capture=True)   # physics + RENDER
       if cube_in_target(): write_episode(ep)
       else:                log(reason, final_cube_pose)
   until target episode count reached
   ```
3. **Demote** `sim/feasibility.py` to an optional diagnostic and
   `sim/validate_library.py` to a reporter over the generation log. Delete the
   0.8 gate and the exit-code-2 escalation; they gate nothing now.

*Done when:* one script produces episodes, and every failed trial's cube pose is
on disk.

### 5.2 Phase 2 — make it render (the only real unknown)

Nothing in this repo has ever produced an image. Everything here is unmeasured.

1. **Add rendering to `SimRobot`:** a `mujoco.Renderer`, a `camera=` argument,
   and a frame-capture hook inside `stream()`. Note `stream()` currently records
   only arm joints (`log_q`), **not** the cube, so you cannot replay an episode
   to render it afterwards — rendering must happen during stepping.
2. Camera: **`overhead`** (it already exists in `scene.xml`; the four cameras are
   `front_camera`, `side_camera`, `overhead`, `wrist_right`). **One camera**, by
   decision — two doubles the cost.
3. **Use the full visual model.** `SimRobot(render=False)` loads
   `scene_headless.xml`, which has the visual geometry stripped and *cannot be
   photographed*. Rendering needs `render=True` and costs ~1.65 GB per process,
   so render parallelism is memory-bound at roughly **8–10 workers, not 16**.
4. **Measure** milliseconds per frame and bytes per episode. This number decides
   how many episodes are affordable, which decides whether any further yield work
   is worth doing.

*Done when:* a PNG of the scene exists, and you can state episodes/hour.

### 5.3 Phase 3 — write the LeRobot dataset

**The format facts below were read out of the installed `lerobot` 0.4.4 and
`SmolVLAConfig`. They correct `conventions.md` §7.4, which is wrong.**

* **Container:** LeRobot **v3.0**. `LeRobotDataset.create(repo_id, fps, features,
  robot_type, root)` → `add_frame(frame)` per step → `save_episode()`.
  `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` are added
  automatically — do not write them.
* **Keys:** `observation.state`, `observation.images.overhead`, `action`, and a
  `task` string passed per frame.
* **fps = 50.** This is the single rate used everywhere: source demos
  (`retarget.fps_out`), execution (`exec.rate_hz`), and the dataset. One control
  step = one frame = one action. No resampling anywhere. The 0.002 s physics
  timestep divides 50 Hz exactly (10 steps). **If you change the dataset rate you
  must change `TIMESTEP` in `build_models.py` and re-measure the physics** —
  `steps_per_cycle` will refuse a rate it cannot hit exactly.
* **Images:** uint8, `[0,255]`, HWC or CHW both accepted. SmolVLA resizes to
  **512×512 with padding** (`resize_imgs_with_padding`), so render **square at
  512** — §7.4's 224×224 would be upscaled and waste detail.
* **State / action:** `(8,)` = right arm's 7 joints + gripper. SmolVLA pads to 32
  (`max_state_dim`), so 8 is fine. `action` = **absolute** joint targets; deltas
  raise `NotImplementedError` in this version. Use `JointIndex.arm_slots` /
  `gripper_slot` — the sim has 16 actuators because the model is bimanual.
* **Normalisation:** write **raw** values. SmolVLA applies `MEAN_STD` to
  state/action and `IDENTITY` to images, computing statistics from the dataset.
* **Instruction:** ≤ **48 tokens** (`tokenizer_max_length`). Use a short fixed
  string like `"put the red cube in the bin"`, **not** EgoDex's `llm_description`.
* **Episode length:** `chunk_size: 50` means a sample needs a 50-step action
  horizon — at 50 fps that is 1.0 s. Drop episodes shorter than ~50 frames.
* **Parallel writing:** `LeRobotDataset` is **not** multi-process safe, and raw
  frames are ~140 MB/episode, too much to pass between processes. Have each
  worker write its **own shard dataset**, then merge. Use `use_videos=True` —
  video encoding takes that 140 MB down to a few MB.

*Done when:* a dataset directory loads back with correct shapes, dtypes, fps and
task string.

### 5.4 Phase 4 — generate, then hand off

Only now does yield matter, and only as *compute per usable episode*. In
priority order, each cheap and each attacking geometry rather than symptoms:

1. **Switch the task to `move_to_spot`** — removes the bin rim, and with it the
   `dz` mismatch, `cube_above_rim`, and probably much of `cube_not_in_bin`.
   Already in `conventions.md` §11 and `config/skills.yaml`.
2. **Diagnose `cube_not_in_bin`** with the data Phase 1.1 now saves. The useful
   tell: a cube at *exactly* its spawn pose means the gripper never gripped it,
   as opposed to one that travelled and fell short. Different causes, different
   fixes. Do not guess.
3. **Delete `max_index_offset`** (correlation −0.048) — recovers 32 demos.
4. **Resolve the safety-box/bin contradiction** (§4.3).
5. **Widen the reachability map** past its grid bounds and re-fit the scene.

Hand off the dataset *plus a spec sheet*: fps, keys, shapes, units, action
semantics, episode count, and the scene randomisation ranges.

### 5.5 Known gap: visual diversity

Scenes vary only in cube position and yaw. There is no variation in lighting,
texture, camera pose or distractors. For fine-tuning a *pretrained* vision model,
visual diversity usually matters more than episode count. Not blocking, but it is
the first thing to add once the pipeline runs.

---

## 6. Conventions: what is binding and what is not

`conventions.md` is a coordination device, not an authority. It was written
before most of the measurements existed. **Split it in two:**

**Binding** — these map to real hardware or real geometry:
§10 limits (joint step 0.05 rad/cycle, tcp speed 0.25 m/s, IK tolerances), the
coordinate frames in §2–3, the cube's 90° rotational symmetry.

**Advisory, and partly wrong** — revise when measurement disagrees:

| Where | Status |
|---|---|
| §2 "top-down grasp" | **Superseded.** Removed this session; feasibility 99 → 115 |
| §7.2 file pattern | Extended to `{source}_{episode}_{closure}.npz` (P12) |
| §7.3 `success_rate >= 0.8` | Correct for *runtime retrieval*; wrong as a generation gate |
| §7.4 dataset schema | **Wrong on image size, says nothing about the 48-token or 50-frame limits.** §5.3 above is authoritative |
| §8 | Deviates via P8, P9, P11 — all load-bearing, all config-gated |
| Gate G4's 50 entries | Measures the wrong quantity |

`docs/conventions_proposals.md` holds P1–P12 with the evidence for every
deviation. **Read it before changing anything in `common/segment.py`,
`common/demos.py` or `sim/execute.py`** — it records bugs that the original spec
would have you reintroduce.

Propose changes there; do not edit `conventions.md` directly.

---

## 7. Working rules

* **Run `uv run pytest tests/`** — not bare `pytest`.
* **Measure before optimising, and record the measurement next to the constant.**
  Most of the damage in this codebase is constants guessed before measurement.
* **Run the synthetic-demo tests (`tests/test_demo_plan.py`) before believing a
  real-data yield.** Three wrong diagnoses came from skipping this.
* **Never hardcode joint order** — use `JointIndex`.
* **Regenerate `sim/*.xml` after any scene change**; never hand-edit them.
* **Append config keys inside the right section.** A key appended to the end of
  `config/skills.yaml` lands under the last section, not the one you meant. Verify
  with `config.skills()["<section>"]`.
* **Estimate the cost of a sweep before launching it.** Timings on this machine:
  `retarget.py` over 1077 episodes ~7 s; `feasibility.py` over 179 demos ~2 min
  (16 processes); `validate_library.py` 99 × 20 trials ~22 min (16 processes).
* **`n_processes: 16` is a hard cap** (top of `config/skills.yaml`). The machine
  has 22 cores and 30 GB; more than 16 workers has OOM-killed runs.
* **Commit after each step**, with the measurement in the message.

### Escalate to a human when

* a download would exceed 5 GB;
* the gripper cannot hold the cube after reasonable actuator tuning (physics
  params are a human decision);
* something requires changing scope — **but not for the gates in §4.4**, which
  this handoff has already judged to be measuring the wrong things.
