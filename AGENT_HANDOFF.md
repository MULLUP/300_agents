# Agent Handoff: Track A (EgoDex → Retargeting → MimicGen Library)

| | |
|---|---|
| **Handoff version** | **H2** |
| **Timestamp** | 2026-10-17T16:20:00Z |
| **Supersedes** | H1 (2026-10-17T13:03:15Z) |
| **Plan reference** | `implementation_plan.md` **V2.1** (2026-10-17T13:01:09Z) |
| **Conventions reference** | `conventions.md` **v2** (2026-10-17T13:09:39Z) |
| **Mode** | **Simulation only.** No real-robot code, no hardware access. |

> If the plan or conventions version above doesn't match the files, stop and ask a human.

---

## 1. Where the previous agent got to

Steps 1–3 of H1 are **done and committed**. Step 4 is **done but under its gate**,
and the project **gained a second skill** on the way. Steps 5–9 are **untouched**.

| H1 step | Status | Key numbers |
|---|---|---|
| 1 Download | ✅ done | 277 `basic_pick_place` + **800 more from `part2.zip`** (`p2_` prefix); 44 `clean_surface`; 16 `wipe_screen` |
| 2 Loader + joint ID | ✅ done | **thumb MCP = `rightThumbIntermediateBase`** (confirmed, evidence in `data/README.md`) |
| 3 Transforms | ✅ done | `pytest tests/test_transforms.py` → **21 passed** |
| 4 Retarget + filters | ⚠️ **done, gate G2 missed** | ~15% keep rate; see §3 |
| 5 Scene, IK, harness | ❌ **not started** | — |
| 6 Feasibility | ❌ not started | — |
| 7 Library candidates | ❌ not started | — |
| 8 MimicGen core | ❌ not started | — |
| 9 Validation | ❌ not started | — |

**Commits so far:** `996999b` (1.P2.2). Step 4 work is staged but see §9.

---

## 2. Read first, in this order

1. **`conventions.md` v2** — binding. Units, frames, schemas, limits.
2. **`docs/conventions_proposals.md`** — **read this before touching step 4.** It
   records five findings (P1–P5) including a detector bug the H1 spec would have
   you reintroduce, and the approved scope change.
3. `implementation_plan.md` V2.1, sections 1–5 and phases 1, 2, 4.
4. `data/README.md` — the MCP decision and its evidence.
5. `sim/README.md` — the pinned `openarm_mujoco` commit and rendering notes.

---

## 3. What changed since H1, and why it matters

### 3a. The H1 grasp detector was wrong (fixed — don't revert it)

H1 step 4 says `grasp_idx` = first frame after which the aperture stays below
`thr`. **That fires on frame 0 in 79% of episodes.** A *relaxed* human hand sits
at a **small** aperture, so the signal is

```
relaxed (small) → open to grasp (large) → close (small) → carry (small)
→ open to release (large) → relaxed (small)
```

Measured on all 277 test episodes: **79%** start below `thr`, **46%** end below it.
`common/segment.py` now only accepts below-threshold runs with an open frame on
**both** sides (`flanked_runs`). Full write-up: proposal **P1**.

### 3b. Thresholds loosened under the gate G2 fallback (user-approved)

All in `config/skills.yaml`. Handoff → now: `min_transport_xy` 0.10 → **0.07 m**,
`min_confidence` 0.5 → **0.3**, `max_speed` 0.15 → **0.35 m/s**. New key
`conf_tail_s: 0.5` — confidence is judged over `[0, release + 0.5 s]`, the frames
we actually keep, not the whole episode. Evidence per threshold: proposal **P2**.

### 3c. `rightIndexFingerTip` is the one bad joint (know this)

Median min-confidence: wrist **0.977**, MCPs **0.77**, `rightIndexFingerTip`
**0.404**. The HumanEgo tcp *position* and *aperture* both come from the two least
reliable joints; orientation (wrist + MCPs) is solid.

**Do not spend time swapping the aperture joint** — the previous agent measured
it: using `IndexFingerIntermediateTip` moves the "exactly one dip" rate only
**50% → 52%**. The rest is genuine human behaviour (people re-adjust grip
mid-carry). Proposal **P3**.

### 3d. Scope change: second skill `wipe_surface` (user-approved, needs conventions v3)

The plan lists wiping as out of scope; **the user approved adding it**. Reason:
wiping needs **no grasp detection at all** (the hand holds a cloth, grip is
constant) and needs **only the wrist pose** (confidence 0.96 vs 0.40).

| | `put_in_bin` | `wipe_surface` |
|---|---|---|
| Episodes | 277 + 800 | 44 `clean_surface` (+16 +6 +7 related) |
| Grasp detection | yes, ~50% ceiling | **none needed** |
| Joints needed | fingertips (0.40) | **wrist only (0.96)** |
| Planarity | 0.191 | **0.107** (flat sweep) |
| Hardware risk | grasp slip, dropped cube | **sponge fixed in gripper** |

**This is blocked on a human** for the conventions v3 items listed in proposal
**P4**: a `surface` frame, optional `grasp_idx`/`release_idx` in the npz schema, a
single-segment transform, a coverage-based success definition, per-skill library
naming, and whether wiping gets its own LoRA adapter. **Ask before implementing
the wiping pipeline** — but §4 step 5 (sim foundations) is unaffected, so start there.

---

## 4. Your tasks, in priority order

### Step 5 is the critical path — start here (plan 1.P1.1–1.P1.4)

Nothing downstream works without it, and it is **independent of all the step 4
trouble**. Everything H1 §5 step 5 says still applies. Facts already established
so you don't have to rediscover them:

**Robot model** (`third_party/openarm_mujoco/v1/openarm_bimanual.xml`, commit
`36d4a443533478a6a1b6e1d6e0d29b35583afc50`, pinned in `sim/README.md`):

| Fact | Value |
|---|---|
| `njnt` / `nu` / `nq` | 18 / 16 / 18 |
| Joint names | `openarm_{left,right}_joint1..7`, `openarm_{left,right}_finger_joint1..2` |
| Right-arm qpos indices | joints 9–15, fingers 16–17 |
| Finger coupling | `neq=2`, joint-equality `finger_joint2 = finger_joint1` — **one actuator per gripper** |
| Finger range | 0 → 0.044 m each; body gap 0.010 → 0.098 m |
| Position actuators | **already present**: `{left,right}_joint{1..7}_ctrl`, `{left,right}_finger1_ctrl` (kp 230/190/30, finger kp 5664) |
| Existing TCP | body `openarm_right_hand_tcp`, 0.09 m along the hand's +z. **Use this for the `tcp_right` site.** |
| Hand local +z | the approach direction (matches `conventions.md` §2) |
| `arm_origin` site | z = 0.697 — the robot base sits high; account for it in `T_table_robot` |
| HOME keyframe | `key_qpos[0]`, joint4 = 1.571, fingers 0.044 (open) |
| Model defaults | `timestep=0.001`, `cone=pyramidal`, `impratio=1` → **override to 0.002 / elliptic / 10 in `sim/scene.xml`** |

`config/skills.yaml` already holds nominal pick-zone, bin, spot and left-zone
values **but they are guesses** — the reachability sweep (H1 step 5) must confirm
and overwrite them. `config/joint_names.yaml`, `config/home.yaml`,
`config/calibration.yaml` **do not exist yet — create them.** `common/config.py`
already has the loaders (`joint_names()`, `home_q()`, `calibration()`).

### Then step 8 (MimicGen core), also independent of demo yield

`common/mimic.py` + `tests/test_mimic.py` per H1 §5 step 8 and `conventions.md`
§8. `common/transforms.py` already provides everything you need: `pos_yaw_pose`,
`yaw_of`, `rot_z_T`, `interp_poses`, `slerp`, `resample_poses`, `inv_T`,
`mirror_lr`.

### Then steps 6, 7, 9 on whatever `put_in_bin` demos exist

Unchanged from H1. Note step 6 needs a minimal version of the step 8 transform —
just use `common/mimic.py` once it exists.

### Finally, `wipe_surface` — only after a human answers proposal P4

---

## 5. Step 4 re-run on the full pool — DONE, gate G2 cleared

Re-ran `uv run python data/retarget.py` over all 1077 episodes (277 test-split +
800 `p2_`). Result: **106 kept / 1077 (9.8%)**, which **clears gate G2's target of
100**. No load errors; 0 episodes missing confidences.

| Reject reason | Count |
|---|---|
| `not_one_grasp_release_pair` | 572 |
| `grasp_speed_too_high` | 149 |
| `transport_too_short` | 138 |
| **kept** | **106** |
| `event_too_close_to_end` | 57 |
| `low_confidence` | 55 |

Index spread of the kept demos: `dist_xy` 7.1–47.2 cm (median 18.2), `dz`
−13.2 to +18.0 cm (median +4.1), duration 1.3–14.6 s (median 3.0).

Note the realized keep rate is **9.8%, not the ~15%** proposal P5 predicted — P5's
estimate came from the 277-episode test split, and the `p2_` episodes segment
slightly worse. The absolute target is still met, so no escalation. The dominant
reject reason remains `not_one_grasp_release_pair` (53%), which is the genuine
human re-gripping behaviour measured in **P3** — not a threshold problem.

---

## 6. Repo state

**Exists and working:**

```
common/transforms.py     pose math, conventions.md §2-3 (21 tests pass)
common/segment.py        grasp/release detection + filters (P1 fix applied)
common/config.py         config/*.yaml loaders, conventions_version check
data/egodex_io.py        HDF5 loader; TCP_JOINT_ROLES is the ONE place the
                         five HumanEgo joint names are encoded
data/inspect_joints.py   MCP identification evidence
data/retarget.py         the step 4 pipeline
data/download_egodex.py  now has --limit, --prefix, and a 5 GB cap
config/skills.yaml       all step 4 + scene + IK + validation tunables
tests/test_transforms.py 21 tests
docs/conventions_proposals.md  P1-P5. Read it.
```

**Missing, you create:** `common/ik.py`, `common/robot_interface.py`,
`common/mimic.py`, `sim/robot_ik.xml`, `sim/scene.xml`, `sim/sim_robot.py`,
`sim/execute.py`, `sim/validate_library.py`, `data/build_library.py`,
`config/joint_names.yaml`, `config/home.yaml`, `config/calibration.yaml`,
`policy/` (may stay empty), `tests/test_mimic.py`.

**Raw data on disk** (all git-ignored):

| Path | Contents |
|---|---|
| `data/egodex_raw/` | `basic_pick_place`: 277 test-split + 800 `p2_`-prefixed from `part2.zip` |
| `data/egodex_raw_clean_surface/` | 44 episodes |
| `data/egodex_raw_wipe_screen/` | 16 episodes |

**Note the layout is inconsistent** — `basic_pick_place` is flat in
`egodex_raw/` while the wiping tasks are in sibling `egodex_raw_<task>/` dirs.
With two skills now, consider moving to `data/egodex_raw/<task>/` and giving
`retarget.py` a `--task` flag. The previous agent left this alone to avoid
breaking the in-flight download.

**Environment:** `uv`, Python 3.12, all deps installed (incl. `pytest`, `lerobot`).
Run everything with `uv run`. `MUJOCO_GL=egl` works on this machine
(`osmesa` is a verified fallback). Call `Renderer.close()` explicitly — letting it
be collected at shutdown raises a harmless `EGLError`.

---

## 7. Rules (unchanged from H1, plus two)

- **Follow `conventions.md` exactly.** Propose changes in
  `docs/conventions_proposals.md`; never edit `conventions.md` yourself.
- **Never hardcode joint order or names** — read `config/joint_names.yaml`.
- **Every tunable goes in `config/*.yaml`.**
- **Every saved artifact carries `conventions_version: 2`** and its source path.
- **Write tests for math.** Plots and videos are the evidence for data steps.
- Small pure functions in `common/`; thin scripts in `data/` and `sim/`.
- No real-robot code, no LeRobot hardware calls, no network beyond EgoDex.
- **Commit after each step**, with the plan task ID in the message.
- **NEW: before "fixing" a low yield in step 4, read proposals P1–P3.** The
  previous agent swept the full threshold grid (27 combinations) and measured the
  ceiling. Thresholds are not the lever.
- **NEW: don't implement `wipe_surface` until a human answers P4.**

---

## 8. Escalate to a human when

- fewer than 40 `put_in_bin` demos survive step 4 after the §5 re-run;
- fewer than 20 library entries survive validation (step 9);
- the v1 gripper can't hold the cube in the hand-coded test after reasonable
  actuator tuning (step 5) — **physics/contact params are a human decision**;
- a download would exceed 5 GB;
- anything would require changing `conventions.md`, scope or gates — including
  **all of proposal P4**.

---

## 9. Status board (update as you go)

| Step | Status | Key numbers / notes | Commit |
|---|---|---|---|
| 1 Download | ☑ | 277 + 800 `p2_` pick_place; 44 + 16 wiping | `996999b` |
| 2 Loader + joint ID | ☑ | thumb MCP = `rightThumbIntermediateBase` | `996999b` |
| 3 Transforms | ☑ | 21 tests pass | `996999b` |
| 4 Retarget + filters | ☑ | kept / total = **106 / 1077** (9.8%) — **G2 target of 100 cleared** | `6ac34bb` |
| 5 Scene, IK, harness | ☐ | hand-coded pick-and-place ?/10 | |
| 6 Feasibility | ☐ | feasible = ? | |
| 7 Library candidates | ☐ | candidates = ? | |
| 8 MimicGen core | ☐ | tests pass? | |
| 9 Validation | ☐ | library entries = ? | |
| P4 `wipe_surface` | ⛔ blocked | awaiting human decision on conventions v3 | |

---

## 10. Next handoff (not yours)

MimicGen data generation with rendering → LeRobot datasets → SmolVLA LoRA (plan
phases 6–7). The dataset schema is in `conventions.md` v2 §7.4. Still needed
before it can start: calibrated camera poses from hardware phase 3, and — if P4 is
accepted — a decision on whether skill becomes a third adapter axis alongside the
A/B/C data-source axis.
