# Implementation Plan: Human-Video vs Teleop Skill Library for OpenArm

| | |
|---|---|
| **Version** | **V2.1 (draft)** |
| **Timestamp** | **2026-10-17T13:01:09Z** |
| **Status** | Working plan. **Expect changes.** Always check the version and timestamp before acting on it. |
| **Supersedes** | V2 (2026-10-17T12:55:23Z) |
| **Companion file** | `conventions.md`. **Read it first.** V2 requires a conventions update (task 1.ALL.2). |

---

## 0. How to use this document (humans and agents)

1. **Read `conventions.md` before writing any code.** Frames, units, joint order and file schemas are defined there.
2. **Every task has an ID** (`phase.owner.number`, for example `3.P1.2`). Refer to tasks by ID in commits, issues and messages.
3. **A task is done only when its done-criterion is met.**
4. **Each phase ends with a gate.** Don't start work that depends on a failed gate. Apply that gate's fallback.
5. **Tick checkboxes** (`[x]`) and append to the changelog in section 13.
6. **Don't change scope, gates or conventions on your own.** Propose the change, get team agreement, then bump the version and timestamp.

---

## 1. Goal

**The project in one line:** compare two sources of demonstrations, **EgoDex human video** and **20 teleop demos on the real robot**. Each is multiplied by MimicGen in MuJoCo into thousands of rendered episodes, converted to LeRobot format, used to fine-tune SmolVLA with LoRA (one adapter per source), and deployed on an OpenArm v1.

```
TRACK A:  EgoDex basic_pick_place ─► geometric retargeting (hand → gripper) ─┐
                                                                             ├─► validate in MuJoCo ─► library
TRACK B:  20 teleop demos on hardware ─► forward kinematics in the twin ─────┘
library ─► MimicGen plans thousands of random scenes ─► MuJoCo executes (physics) + renders cameras
        ─► keep successes ─► LeRobot dataset (images + state + actions + language)
        ─► LoRA fine-tune of SmolVLA (adapters A, B, C) ─► real OpenArm
```

**Two layers on the robot:**

| Layer | What it is | Role |
|---|---|---|
| **1. Retrieval** | Run the nearest validated library demo, adapted to the scene | **Guaranteed demo** (no learning) |
| **2. SmolVLA + LoRA** | Image-based learned policy trained on MuJoCo-generated data | **The learning result.** Deployed only if its gates pass |

**The experiment, which answers the challenge slide's data-efficiency question ("what's the best combination of human video, teleop and simulation?"):**

| Adapter | Training data |
|---|---|
| **A** | Simulated episodes generated from **EgoDex** |
| **B** | Simulated episodes generated from **teleop** |
| **C** | A + B combined, **plus the 20 real teleop episodes** (real camera images) |

**Honest framing:** human video and teleop provide **how** to move; the camera, the MimicGen transform and physics provide **where**. A and B use zero real robot images; C uses 20 real episodes.

---

## 2. Scope

### In scope
- Task: **pick-and-place** with one cube, **right arm**. Skills:
  - `put_in_bin` (primary),
  - `move_to_spot` (secondary; retrieval only unless time allows).
- Two source tracks (EgoDex, teleop), one shared MimicGen generator, three SmolVLA LoRA adapters.
- Simulation-first development (H0–8), hardware from H8 (needed for teleop).
- Runtime with verification, one retry, run logging, a UI and a results comparison.
- **Stretch:** a two-arm demo, `dual_put_in_bin`. Each arm independently puts its own cube in its own bin, at the same time, **retrieval only**, with no coordination between arms. Attempted only after gate G5b; cut if not solid by the H21 code freeze.

### Out of scope
- Eidon and wiping; other EgoDex tasks.
- The pose-based MLP policy (replaced by SmolVLA in V2).
- Pi0 and other large VLAs; Isaac Lab; the official MimicGen framework; installing Phantom.
- Coordinated bimanual skills (DexMimicGen-style), two-arm SmolVLA, deformables, robot 2.

---

## 3. Approach and anchors

| Component | Anchor | Use |
|---|---|---|
| Human data | EgoDex (Apple), `basic_pick_place`. Test set: <https://ml-site.cdn-apple.com/datasets/egodex/test.zip> (277 episodes); more in `part2.zip` (~27k episodes) | Track A source; per-joint hand transforms; `llm_description` attribute for language |
| Hand → gripper | HumanEgo (arXiv 2605.24934) | Fingertip midpoint = position; wrist → MCP midpoint = forward axis; thumb MCP → index MCP = jaw axis; fingertip distance = opening |
| Retargeting cross-check | Phantom `action_processor.py` | Read the code only, don't install |
| Data multiplication | MimicGen (arXiv 2310.17596) | Object-centric segments, rigid re-anchoring, interpolation, keep successes. Track B is standard MimicGen input (teleop); track A is our adaptation |
| Source selection | Isaac Lab Mimic (nearest object pose) | Retrieval (k = 1) and generation (k = 5) |
| Simulation | MuJoCo + `openarm_mujoco` (`v1/openarm_bimanual.xml`) | Physics, rendering, digital twin |
| IK | mink (`solver="daqp"`) | Gripper poses → joint angles |
| Dataset and training | LeRobot (dataset format v3, `lerobot/smolvla_base`, PEFT/LoRA support) | Training adapters A, B, C |
| Real robot | LeRobot `OpenArmFollower` | Driver; also `lerobot-record` for the teleop demos |

**Why we implement the MimicGen core ourselves in MuJoCo:** the official stack pins Python 3.8, robosuite 1.4 and MuJoCo 2.3.2, while mink needs MuJoCo 3.1.6 or newer; and the official stack expects simulator-recorded demos.

---

## 4. Team and ownership

| ID | Role | Owns |
|---|---|---|
| **P1** | Robot and simulation | MuJoCo scene and cameras, IK, `SimRobot`, physics execution harness, hardware bring-up, `RealRobot`, **teleop recording**, real-robot runs |
| **P2** | Data and learning | EgoDex retargeting, teleop conversion, MimicGen core, library validation, data generation, LeRobot export, SmolVLA LoRA training |
| **P3** | Perception, runtime and product | Simulated and real cameras, detection, calibration (camera), runtime, verification, UI, results, pitch |

---

## 5. Architecture

### Repository layout (`common/`, `config/` and `policy/` added to the existing folders)

| Path | Purpose | Owner |
|---|---|---|
| `common/transforms.py` | Pose math and frame conversions | P2 |
| `common/robot_interface.py` | Abstract robot interface | P1 |
| `common/ik.py` | mink wrapper with error checks | P1 |
| `common/mimic.py` | MimicGen core: heading alignment, anchor transforms, segment bridging | P2 |
| `sim/scene.xml` | OpenArm v1, table, cube, bin, spot, **cameras matching the real ones** | P1 |
| `sim/sim_robot.py` | `SimRobot` | P1 |
| `sim/execute.py` | Physics execution harness: trajectory → simulated run → success, with optional rendering | P1 |
| `sim/validate_library.py` | Library validation (both tracks) | P2 |
| `sim/generate_data.py` | MimicGen generation with physics, visual and dynamics randomization | P2 |
| `data/download_egodex.py`, `data/retarget.py` | Track A source demos | P2 |
| `data/convert_teleop.py` | Track B: recorded teleop → source demos (and real LeRobot episodes in our schema) | P2 |
| `data/build_library.py` | Segments, anchors, index → `data/library/{source}_v{n}.json` | P2 |
| `data/mirror_library.py` | *(Stretch)* Mirror a right-arm library into a left-arm library | P2 |
| `data/export_lerobot.py` | Generated episodes → LeRobot datasets A, B, C | P2 |
| `policy/train_lora.sh`, `config/policy.yaml` | SmolVLA LoRA training runs | P2 |
| `policy/infer_smolvla.py` | Closed-loop SmolVLA executor on the robot interface | P2 + P3 |
| `perception/*` | Simulated/real camera, detection, ChArUco calibration | P3 |
| `runtime/real_robot.py` | `RealRobot` | P1 |
| `runtime/retrieve.py`, `runtime/skills.py`, `runtime/verify.py`, `runtime/db.py`, `runtime/app.py`, `runtime/ui/` | Runtime, both layers, UI | P2 (retrieve) / P3 |

### Dataset schema (V2 addition to `conventions.md`)

| Field | Content |
|---|---|
| `fps` | 30 (policy execution also runs at 30 Hz) |
| `observation.images.overhead` | RGB, 224×224 for training (stored larger if convenient) |
| `observation.images.wrist_right` | Only if the real robot has a right wrist camera; otherwise omit everywhere |
| `observation.state` | 8 values: `right_joint_1 … right_joint_7` (rad), `right_gripper` (0–1) |
| `action` | 8 values: absolute joint targets, same order and units as the state |
| `task` | Language instruction, e.g. "put the red cube in the bin" |

**Sim and real must produce identical keys, image size, units and order.** The real-robot conversions (degrees, joint signs) happen only inside `RealRobot` and `data/convert_teleop.py`.

---

## 6. Sim-to-real transfer

MuJoCo is a **calibrated digital twin**. Same code, three swaps:

| Component | Simulation | Real robot |
|---|---|---|
| Robot interface | `SimRobot` | `RealRobot` (LeRobot `OpenArmFollower`) |
| Cameras | Rendered, at the **calibrated real poses and intrinsics** | Real cameras |
| Calibration | Measured values written back into `sim/scene.xml` | Measured in phase 3 |

**Gaps and fixes:**

| Gap | Fix |
|---|---|
| **Rendered vs real images** (the biggest risk for SmolVLA) | Visual randomization (textures, colors, lighting, ±1 cm / ±1° camera jitter); a plain real scene matching the simulation; adapter C mixes in real images |
| Gravity sag, tracking error | Calibrate from measured joint angles; `time_scale` 0.25 → 0.5 for retrieval |
| Gripper contact and friction | Dynamics randomization; tune grasp offsets on hardware |
| Policy latency | Run SmolVLA on a GPU machine; if the robot PC has none, use LeRobot's async inference |

**Hardware bring-up order:** air runs → cube on a fixed spot → random positions. Report sim and real results separately.

---

## 7. Environment setup

```bash
sudo apt install -y git build-essential libgl1 libegl1 libglfw3 libosmesa6
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv init --bare --python 3.12 && uv python pin 3.12
uv add mujoco mink "qpsolvers[daqp]" numpy scipy h5py remotezip \
       opencv-contrib-python matplotlib pyyaml fastapi uvicorn
uv add lerobot          # brings PyTorch; needed for dataset export, SmolVLA, teleop recording
git clone https://github.com/enactic/openarm_mujoco.git third_party/openarm_mujoco
export MUJOCO_GL=egl    # offscreen rendering (fallback: osmesa)
```

On the GPU training machine, install the SmolVLA extras per the LeRobot docs. `.gitignore`: `.venv/`, `third_party/`, `data/egodex_raw/`, `data/generated/`, `data/lerobot/`.

---

## 8. Phases overview

| Phase | Title | Hours | Lead | Gate |
|---|---|---|---|---|
| 1 | Sim foundations: scene, cameras, IK | H0–4 | All | G1 |
| 2 | Track A: EgoDex retargeting | H4–8 | P2 | G2 |
| 3 | Hardware bring-up, calibration, teleop recording | H8–13 | P1 + P3 | G3 |
| 4 | Source libraries: validate tracks A and B | H8–14 | P2 | G4 |
| 5 | Layer 1: retrieval, sim then real, with UI | H10–18 | P3 + P1 | G5 |
| 5S | *Stretch:* two-arm demo (`dual_put_in_bin`), retrieval only | H18–21 | P1 + P3 (+P2) | GS |
| 6 | MimicGen data generation with rendering → LeRobot datasets | H14–17 | P2 (+P1) | G6 |
| 7 | SmolVLA LoRA training (A, B, C) + simulation evaluation | H17–21 | P2 | G7 |
| 8 | SmolVLA on hardware + full evaluation | H21–25 | All | — |
| 9 | Pitch and rehearsal | H25–30 | P3 | — |

**Code freeze at H21.** After that, only evaluation, bug fixes and the pitch.

**GPU priority** if only one GPU is available: train C → A → B.

---

## 9. Phase details

### Phase 1: Sim foundations (H0–4)

**Everyone**
- [ ] **1.ALL.1** Create `common/`, `config/`, `policy/`. Define `common/robot_interface.py`.
- [ ] **1.ALL.2** Update `conventions.md` to v2 with the dataset schema from section 5 (30 fps policy rate, 8-D state and action, camera names). Bump `conventions_version` to 2.
- [ ] **1.ALL.3** Props: 2–3 colored cubes (4–5 cm), a bin, painter's tape, a ChArUco board on something rigid, an ArUco marker for the bin, a matte plain tablecloth.
- [ ] **1.ALL.4** Find out which real cameras exist (overhead/head, right wrist) and their resolutions. This decides the camera keys.

**P1**
- [ ] **1.P1.0** *(30 min)* Connect to the real arms and read joint angles only.
- [ ] **1.P1.1** Load `v1/openarm_bimanual.xml`; write `config/joint_names.yaml`; add position actuators if missing; add `tcp_left` and `tcp_right` sites.
- [ ] **1.P1.2** `sim/scene.xml`: world = `table` frame, robot at nominal `T_table_robot`, table, free cube, bin, spot, and cameras at the planned real poses. Also define a **left zone**: a second cube (different color) and a second bin on the robot's left side, disabled by default. It's cheap to add now and needed for the stretch.
- [ ] **1.P1.3** `common/ik.py`, `sim/sim_robot.py`, `sim/execute.py`.
- [ ] **1.P1.4** Hand-coded pick-and-place. *Done:* 10/10 with random cube positions.

**P2**
- [ ] **1.P2.1** Download test-set `basic_pick_place` HDF5 files (277), plus 3 MP4s for the pitch.
- [ ] **1.P2.2** HDF5 loader; confirm the thumb MCP joint name; `common/transforms.py`. *Done:* 10 demos retargeted and plotted.

**P3**
- [ ] **1.P3.1** `perception/sim_camera.py`, `perception/detect.py`. *Done:* detected positions within 1 cm of the truth on 20 scenes.
- [ ] **1.P3.2** Runtime skeleton: `runtime/app.py`, `runtime/db.py`, live view from `SimRobot`.

**Gate G1 (H4):** hand-coded pick-and-place 10/10; 10 demos retargeted; detection within 1 cm. **If it fails:** all three fix the failing item.

---

### Phase 2: Track A, EgoDex retargeting (H4–8, P2)

- [ ] **2.P2.1** `data/retarget.py`: ARKit → `demo` frame; HumanEgo gripper pose and opening; binary grip; resample to 50 Hz; save `data/retargeted/egodex_*.npz`.
- [ ] **2.P2.2** Grasp and release detection, plus quality filters: one grasp-release pair, confidence threshold, transport ≥ 10 cm, no jumps over 5 cm.
- [ ] **2.P2.3** IK feasibility with the grasp anchored at a nominal cube position.
- [ ] **2.P2.4** Pitch asset: the human clip next to the simulated replay, with human keypoints drawn as spheres.
- [ ] **2.P2.5** Cross-check against Phantom's `action_processor.py` (read only).

**Gate G2 (H8):** at least 100 demos pass. **If it fails:** loosen thresholds (logged), then pull more from `part2.zip`. Minimum 40.

---

### Phase 3: Hardware bring-up, calibration, teleop recording (H8–13, P1 + P3)

- [ ] **3.P1.1** Bring up both arms with `OpenArmFollower` (`max_relative_target = 3.0`). Test disconnect behavior **with the arm supported**. Real HOME into `config/home.yaml`.
- [ ] **3.P1.2** Joint check at 5 poses **for both arms** → `config/joint_map_real.yaml`; verify both gripper mappings. Implement `runtime/real_robot.py`.
- [ ] **3.P3.1** Mount the cameras (they never move again); intrinsics into `config/camera.yaml`; ChArUco → `T_table_cam`; pixel → table within 1 cm at 5 points.
- [ ] **3.J.1** Robot base calibration from 3 touched board corners (**measured** joints) → `T_table_robot`. **Write the real camera and base poses back into `sim/scene.xml`.**
- [ ] **3.P1.3** Set up teleop: the organizers' VR rig, or leader arms if available. Practice for 10 minutes.
- [ ] **3.P1.4** Record **20 `put_in_bin` teleop demos** with `lerobot-record`, using camera names that match the dataset schema:
  - cube at varied positions across the pick zone;
  - every demo starts and ends at HOME;
  - same strategy every time (top-down grasp).
  - Before each demo, save the camera's detected cube pose.

**Gate G3 (H13):** click-to-hover within 2 cm 10/10 times, **and** at least 20 teleop demos recorded.

**If it fails:**
- calibration fails → **fixed-spot mode** for real runs;
- teleop unavailable → **drop track B**: adapter A only, plus C = A + nothing real; record this in the changelog.

---

### Phase 4: Source libraries for tracks A and B (H8–14, P2)

- [ ] **4.P2.1** `common/mimic.py` per `conventions.md` section 8: heading alignment, approach re-anchored to the cube (yaw snap), transport re-anchored to the target, 0.5 s bridging.
- [ ] **4.P2.2** Track A library: `data/build_library.py` on EgoDex demos → `sim/validate_library.py` (20 scenes each, success = cube in bin 1 s after release) → `data/library/egodex_v1.json`, keeping entries with success ≥ 0.8.
- [ ] **4.P2.3** Track B: `data/convert_teleop.py`.
  - Recorded joints → (joint map) → forward kinematics in the calibrated twin → gripper poses in the `table` frame.
  - Object anchor = the **measured** cube pose (no guessing, unlike track A).
  - Write source demos in the same schema as track A, plus a LeRobot copy of the 20 real episodes in the V2 dataset schema.
- [ ] **4.P2.4** Track B library: validate → `data/library/teleop_v1.json`.
- [ ] **4.P2.5** *(Stretch, only if G4 is met before H14; otherwise do it in phase 5S)* `data/mirror_library.py`:
  - mirror the best right-arm library across the robot's left–right plane: position `p' = M p`, rotation `R' = M R M`, with `M = diag(1, −1, 1)` in the `table` frame;
  - validate the result for the left arm (`tcp_left`, left zone);
  - output `data/library/{source}_left_v1.json`, keeping entries with success ≥ 0.8.

**Gate G4 (H14):** EgoDex library ≥ 50 entries (minimum 20); teleop library ≥ 15 entries.
**If it fails:** inspect failures (grasp slips → P1 tunes gripper friction or close value; unreachable → tighten the pick zone).

---

### Phase 5: Layer 1, retrieval, sim then real, with UI (H10–18, P3 + P1)

- [ ] **5.P2.1** `runtime/retrieve.py`: nearest entry by `(dist_xy, dz)`, with a source filter (`egodex | teleop | any`).
- [ ] **5.P3.1** `runtime/skills.py`, retrieval mode:
  1. detect;
  2. retrieve;
  3. transform;
  4. run IK;
  5. **simulate first** (fall back to the next entry, up to 3);
  6. execute;
  7. verify;
  8. retry once;
  9. log.
- [ ] **5.P3.2** `runtime/verify.py`: cube in the bin region or within 3 cm of the spot; grasp-miss check.
- [ ] **5.P3.3** UI:
  - skill cards with source, N demos, live success per mode and per sim/real;
  - Run button with mode and source selectors;
  - live view, run history, the side-by-side video.
- [ ] **5.P3.4** Simulation evaluation: 50 retrieval runs of `put_in_bin`. **Gate G5a (H14):** ≥ 80%.
- [ ] **5.P1.1** Real retrieval: air runs at `time_scale = 0.25` → fixed spot → random positions. Tune only the grasp height offset, close value and `place_clearance`.

**Gate G5b (H18):** real retrieval `put_in_bin` ≥ 7/10.
**If it fails:** shrink the pick zone, use a bigger cube; last resort is fixed-spot mode.

---

### Phase 5S (Stretch): Two-arm demo, retrieval only (H18–21, P1 + P3, P2 for the mirrored library)

**Start only if G5b has passed.** Each arm runs its own independent single-arm skill; there's no coordination between them.

- [ ] **5S.P2.1** Left-arm library ready (task 4.P2.5, if not already done).
- [ ] **5S.J.1** Real setup: tape separate left and right zones so neither arm's pick zone or bin is reachable by the other arm's trajectories. Place the second cube and bin; measure the left bin position into `config/calibration.yaml`.
- [ ] **5S.P3.1** `runtime/skills.py` → `dual_put_in_bin`:
  1. detect both cubes;
  2. retrieve independently from the right and left libraries;
  3. transform and run IK per arm;
  4. **merge into one 16-value joint stream**, aligned to a common start;
  5. **simulate both arms together first**; reject the run on any arm-to-arm or arm-to-object collision that isn't the intended grasp;
  6. execute;
  7. verify both bins, retrying only the arm that failed;
  8. log one run with two results.
- [ ] **5S.P3.2** UI: a "Two skills at once" card showing combined and per-arm success rates.
- [ ] **5S.P1.1** Real bring-up order: left-arm air runs → left arm alone with its cube → both arms together at `time_scale = 0.25`, e-stop in hand.

**Gate GS (H21):** `dual_put_in_bin` succeeds on **both** arms in at least 4/5 real runs.
**If it fails:** cut the stretch completely (UI card hidden). Single-arm retrieval remains the demo.

---

### Phase 6: MimicGen data generation with rendering → LeRobot datasets (H14–17, P2, P1 supports performance)

- [ ] **6.P2.1** `sim/generate_data.py` per random scene:
  - sample cube pose and target;
  - pick a source from the 5 nearest library entries of the chosen library;
  - transform, IK, execute with physics;
  - **render all dataset cameras at 30 fps**;
  - keep successes.
- [ ] **6.P2.2** Randomization (`config/datagen.yaml`):
  - **visual:** table texture and color, lighting, cube shade, camera jitter ±1 cm / ±1°;
  - **dynamics:** cube size ±5 mm, mass and friction ±30%, base pose ±1 cm;
  - **noise:** action noise, 0–2 step latency.
- [ ] **6.P2.3** Generate **1,000 successful episodes from the EgoDex library** and **1,000 from the teleop library**, in parallel processes. Log each generation's own success rate.
- [ ] **6.P2.4** `data/export_lerobot.py` → three LeRobot datasets:
  - **A** = 1,000 EgoDex-sourced episodes;
  - **B** = 1,000 teleop-sourced episodes;
  - **C** = 500 + 500 sampled from A and B, **plus the 20 real teleop episodes**.
  - Language from EgoDex `llm_description` phrasings plus fixed templates.
- [ ] **6.P2.5** Sanity check: load each dataset with LeRobot, visualize 3 episodes, check that keys, shapes and units match the real teleop episodes exactly.

**Gate G6 (H17):** datasets A and C exist and load (B too if track B survived).
**If it fails:** cut to 500 episodes per dataset; drop rendering of the wrist camera.

---

### Phase 7: SmolVLA LoRA training + simulation evaluation (H17–21, P2)

- [ ] **7.P2.1** Train LoRA adapters on `lerobot/smolvla_base`, one per dataset (C → A → B), using LeRobot's PEFT/LoRA options. Check `lerobot-train --help` in the installed version for the exact flags.
  - Hyperparameters in `config/policy.yaml`.
  - Steps sized to 5–10 epochs.
  - **Fallback if the PEFT route breaks:** full fine-tune of SmolVLA (still one checkpoint per dataset).
- [ ] **7.P2.2** `policy/infer_smolvla.py`: load the policy with LeRobot's pre/post-processors; closed loop at 30 Hz with action chunks; output to the robot interface.
- [ ] **7.P2.3** Simulation evaluation: 50 closed-loop episodes per adapter **with visual randomization on**, using the simulated cameras.
- [ ] **7.P3.1** Add SmolVLA mode (adapter selector) to `runtime/skills.py` and the UI.

**Gate G7 (H21):** the best adapter reaches ≥ 60% in simulation.
**If it fails:** don't run SmolVLA on hardware. Present the simulation comparison (A vs B vs C) as the learning result; retrieval stays the live demo.

**CODE FREEZE AT H21.**

---

### Phase 8: SmolVLA on hardware + full evaluation (H21–25, all)

- [ ] **8.P1.1** *(If G7 passed)* Best adapter on the real robot: air runs, then 5 cube runs at reduced speed with the e-stop in hand. If it's safe and working, 20 trials with that adapter; if time allows, 10 trials each for the others.
- [ ] **8.ALL.1** Real retrieval: 20 trials of `put_in_bin` from the UI, with random positions.
- [ ] **8.ALL.1S** *(If GS passed)* 10 real trials of `dual_put_in_bin`; report the both-arms success rate and the per-arm success rates.
- [ ] **8.ALL.2** Results table:

  | mode | source | sim success | real success | N source demos | N generated episodes | real images in training |
  |---|---|---|---|---|---|---|
  | retrieval | EgoDex / teleop | | | | — | — |
  | SmolVLA A | EgoDex | | | | 1,000 | 0 |
  | SmolVLA B | teleop | | | 20 | 1,000 | 0 |
  | SmolVLA C | both + real | | | | 1,000 | 20 |
  | retrieval, two arms (stretch) | mirrored library | | | | — | — |

- [ ] **8.ALL.3** Videos: successful real runs per mode, plus one failure with its retry.

---

### Phase 9: Pitch and rehearsal (H25–30, P3 lead)

- [ ] **9.P3.1** Deck:
  1. the question (human video vs teleop vs both);
  2. the pipeline diagram (section 1);
  3. the results table and chart;
  4. the honest framing;
  5. the roadmap: one LoRA adapter per skill as the skill-store format, robot-collected real data, more EgoDex tasks, other robots via URDF.
- [ ] **9.P3.2** Live demo: the UI library → a judge runs `put_in_bin` (retrieval) → `dual_put_in_bin` if GS passed (both arms at once) → SmolVLA mode if deployed → the side-by-side video.
- [ ] **9.ALL.1** Backup videos for every demo path; three full rehearsals.

---

## 10. Safety rules (real robot)

- E-stop within reach during **every** real run, including teleop.
- No one inside the arm's reach during autonomous runs.
- Two-arm runs: both arms' workspaces must be clear of people; the trajectory must have passed the combined two-arm simulation check in the same session.
- Retrieval trajectories run in the calibrated simulation first. SmolVLA starts at reduced speed with the workspace box and per-step cap enforced.
- All limits in `conventions.md` section 10 are enforced in the robot interface, never bypassed.

---

## 11. Known gaps and risks

| Risk | Impact | Mitigation |
|---|---|---|
| Rendered images ≠ real images | SmolVLA may fail on hardware | Visual randomization, plain scene, adapter C with real images, retrieval fallback |
| No VR teleop tested on OpenArm v1 | Track B could be lost | Leader arms; else drop track B (G3 fallback) |
| EgoDex has no object poses | Track A anchors are guesses | Simulation validation; track B uses measured poses |
| Rendering slows generation | Fewer episodes | Small images, parallel processes, G6 fallback |
| LeRobot PEFT API differences | Training blocked | Check `--help`; full fine-tune fallback |
| Only one GPU | Not all three adapters get trained | Priority C → A → B |
| SmolVLA inference latency | Jerky control | GPU machine, async inference, action chunks |
| EgoDex is non-commercial | Not shippable | State it; product needs licensed data |
| Mirrored motions may not suit the left arm | Two-arm stretch fails | Simulation validation of the mirrored library; cut at GS |
| Arm-to-arm collision in the two-arm demo | Hardware damage | Separate taped zones; combined simulation check before every two-arm run |

---

## 12. Decision log

| # | Decision | Reason |
|---|---|---|
| D1–D6 | As in V1: retrieval library, simulation-first, one task and arm, HumanEgo frame, yaw-only anchors, simulate before executing | Unchanged |
| D7 | MimicGen core implemented ourselves in MuJoCo | The official stack clashes with mink and expects simulator-recorded demos |
| D8 | MuJoCo over Isaac Lab | Hardware requirements; photorealism not worth the cost here |
| D9 | Phantom as code reference only | Different problem; no OpenArm support |
| D12 | **SmolVLA + LoRA replaces the pose-based MLP** as the learned layer | One learned layer for 3 people; aligns with the skills-as-adapters story |
| D13 | **Two source tracks (EgoDex, teleop)** feeding one generator | Directly answers the challenge's data-efficiency question; teleop de-risks the human-video track |
| D14 | Hardware bring-up moved to H8 | Needed for teleop recording and calibrated cameras before generation |
| D15 | One adapter per data source (A, B, C) | Makes the comparison clean; mirrors one-adapter-per-skill |
| D16 | Two-arm demo as **independent** skills, retrieval only, stretch | Strong visual for little effort; no DexMimicGen-style coordination needed; SmolVLA stays single-arm |

---

## 13. Changelog

| Version | Timestamp (UTC) | Change |
|---|---|---|
| V0 | 2026-10-17T11:38:21Z | Initial plan |
| V1 | 2026-10-17T12:11:17Z | MimicGen framing, pose-based policy layer, sim-to-real section |
| V2 | 2026-10-17T12:55:23Z | Two source tracks (EgoDex vs teleop). MimicGen generation with rendering → LeRobot datasets → SmolVLA LoRA adapters A/B/C replaces the pose MLP. Hardware bring-up and teleop moved to H8–13. New dataset schema (conventions v2 required). Phases and gates renumbered; code freeze moved to H21. |
| V2.1 | 2026-10-17T13:01:09Z | Added the stretch phase 5S: two-arm demo `dual_put_in_bin` (independent arms, retrieval only, mirrored library, combined collision check, gate GS). Left zone added to the simulated scene, joint check for both arms, evaluation, demo script, safety, risks and decision D16 updated. |
