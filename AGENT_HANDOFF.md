# Agent Handoff: Track A (EgoDex → Retargeting → MimicGen Library)

| | |
|---|---|
| **Handoff version** | **H4** |
| **Timestamp** | 2026-10-17T21:40:00Z |
| **Supersedes** | H3 (2026-10-17T18:05:00Z), H2 (2026-10-17T16:20:00Z), H1 (2026-10-17T13:03:15Z) |
| **Plan reference** | `implementation_plan.md` **V2.1** (2026-10-17T13:01:09Z) |
| **Conventions reference** | `conventions.md` **v2** (2026-10-17T13:09:39Z) |
| **Mode** | **Simulation only.** No real-robot code, no hardware access. |

> If the plan or conventions version above doesn't match the files, stop and ask a human.

---

## 0. START HERE — the two open numbers

H3's section 0 items are all resolved: the stale FK assertion is fixed, P7 is
verified, and everything through step 6 is committed (`b89aec5`). The test suite
is **138 passing, 0 failing**. Two numbers are open, and neither is a mystery —
both have a diagnosis and a proposed next action.

### 0a. Gate G1 is 9/10, and the last one is a pick-zone edge pose

```
trial 6: cube=(0.261,-0.209) yaw=-32.9 -> FAIL unsafe_trajectory
         tcp_speed 0.251 -> 0.274 m/s against the 0.25 limit
```

P7's grasp tilt **did** fix the other failure, `(0.311, -0.240)`. This one is
different and the `tcp_speed` message is a symptom, not the cause:

`plan_put_in_bin` searches (tilt, grasp yaw, release yaw) for a plan that is
smooth under `_plan_ok`. For this cube pose **nothing in the search passes**, and
the function then returns its `fallback` — a plan it has already established
fails. Execution inherits a branch flip, `_fit_rate_limits` stretches the
trajectory 27x trying to respect the rate limits (hence sample 13556), and the
tcp still swings too fast through the flip. Stretching time cannot fix a
discontinuous joint path.

The pose is at **y = -0.2086**, right at the pick zone's upper y edge
(`[-0.34, -0.20]`). `config/skills.yaml` already records that `y >= -0.20` has
cube yaws with no legal reachable grasp. **Proposed:** narrow `pick_zone.size` to
`[0.12, 0.12]` so the zone becomes x `[0.20, 0.32]`, y `[-0.33, -0.21]`, then
re-run. This is a scene-design choice, not a fudge — the arm demonstrably cannot
do reliable top-down picks above y = -0.21 — but **it has not been measured yet**,
so measure the smooth-plan rate across the boundary before committing to it.

Honest alternative: accept 9/10 and record it. Do not let this block step 7.

### 0b. Step 6 yields 13/106 feasible, 6/13 under physics

Below gate G4's minimum of 20 library entries. This is the real risk and
§10 is the whole story. Three bugs were found and fixed getting from 1 to 13
(see `b89aec5` and P8/P9/P10); the remaining rejects are **mostly genuine
geometry**, not obviously more bugs. Read §10 before trying to raise it.

### 0c. P8 and P9 need a human decision

Both deviate from `conventions.md` §8 and are implemented config-gated behind
`library.orientation` and `library.grasp_window_s`. Neither is a tuning knob —
without them the feasible count is zero. See §8.

---

## 1. Where the previous agents got to

Steps 1–4 and 8 are **done**. Step 5 is **done but under gate G1** (§0b).
Steps 6, 7, 9 are untouched.

| Step | Status | Key numbers |
|---|---|---|
| 1 Download | ✅ done, committed | 277 `basic_pick_place` + **800 from `part2.zip`** (`p2_` prefix); 44 `clean_surface`; 16 `wipe_screen` |
| 2 Loader + joint ID | ✅ done, committed | **thumb MCP = `rightThumbIntermediateBase`** (evidence in `data/README.md`) |
| 3 Transforms | ✅ done, committed | 21 tests pass |
| 4 Retarget + filters | ✅ **done, gate G2 CLEARED** | **106 / 1077 kept (9.8%)**; see §5 |
| 5 Scene, IK, harness | ⚠️ **done, gate G1 at 9/10** | committed; see §0a |
| 6 Feasibility | ⚠️ **done, 13/106** | 6/13 succeed under physics; **read §10** |
| 7 Library candidates | ❌ not started | `data/build_library.py` — next up, see §4 |
| 8 MimicGen core | ✅ **done** | `common/mimic.py`, 36 tests |
| 9 Validation | ❌ not started | `sim/validate_library.py` |

**Commits:** `996999b` (1.P2.2), `e576d3d` (2.P2.1/2.P2.2, step 4), `b89aec5`
(steps 5, 6, 8). The working tree is clean.

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

### Step 7 is the critical path — start here (plan 4.P2.2)

`data/build_library.py`: turn the feasible demos into
`data/library/egodex_v1.json` per `conventions.md` §7.3, then
`sim/validate_library.py` (step 9) scores each entry over
`validation.n_trials` random scenes and keeps `success_rate >= 0.8`.

**Almost everything you need already exists.** Do not rebuild it:

| You need | Use |
|---|---|
| demo npz -> `SourceDemo` | `common.demos.load_source_demo` (applies P8 flattening, trim and P9 anchor z) |
| provenance for the §7.3 entry | `common.demos.load_metadata` |
| the `(dist_xy, dz)` index | `SourceDemo.index` |
| anchors for the §7.3 entry | `common.mimic.demo_anchors` |
| demo + cube pose -> streamable path | `sim.execute.plan_from_demo` |
| cheap geometry screen | `sim.execute.screen_pose_path` |
| full physics attempt, scored | `sim.execute.run_demo` (same contract as `run_put_in_bin`) |
| which demos are worth trying | `sim/plots/feasibility_egodex_top_down.json` |

So step 9 is mostly a loop over `run_demo` with a different cube pose per trial,
plus `validation.n_processes` parallelism. Sampling the cube pose is the one piece
not written: `sim/execute.py main()` shows the pattern.

### Then the two open numbers in §0

§0a (gate G1 9/10) and §0b (feasible count) in whichever order. §0a is a
~30-minute measurement; §0b is open-ended, so timebox it.

### Finally, `wipe_surface` — only after a human answers proposal P4

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
| 4 Retarget + filters | ☑ | kept / total = **106 / 1077** (9.8%) — **G2 target of 100 cleared** | `e576d3d` |
| 5 Scene, IK, harness | ⚠️ | hand-coded pick-and-place **8/10** verified; P7 tilt wired in but **unverified**; shoulder height, zones and IK seeding all **measured**, see §10 | uncommitted |
| 6 Feasibility | ☐ | feasible = ? — **read §10 first**, the usable workspace is ~25 cm across | |
| 7 Library candidates | ☐ | candidates = ? | |
| 8 MimicGen core | ☑ | `common/mimic.py`, 36 tests in `tests/test_mimic.py` | uncommitted |
| 9 Validation | ☐ | library entries = ? | |
| — test suite | ⚠️ | **1 failing test**, stale assertion — see §0a | |
| P4 `wipe_surface` | ⛔ blocked | awaiting human decision on conventions v3 | |

---

## 10. Step 6 findings — read before trying to raise the feasible count

`uv run python sim/feasibility.py` -> **13 / 106 feasible**, and 6 of those 13
put the cube in the bin under physics. Full per-demo detail, including the
rejection reason and the index offset, is in
`sim/plots/feasibility_egodex_top_down.json`.

### 10a. What "feasible" means here

Pure kinematics, no physics: the demo is retargeted onto **one fixed scene** (cube
at the pick-zone centre, yaw 0), and the resulting path must clear the table by
`library.min_tcp_z`, solve from a cold IK start, converge at every sample, and
contain no branch flip. Plus a free pre-screen on the `(dist_xy, dz)` index. The
cube is not even simulated. Step 9 is what answers "does it work".

### 10b. The reject buckets, and which are real

| Reason | Count | Real, or worth another look? |
|---|---|---|
| `ik_diverged` | 28 | **Mostly real.** Median 29% of samples bad, median worst error 74 mm — not marginal. Only 1 of 28 is within 15 mm of passing. |
| `start_unreachable` | 23 | **Mixed.** Errors run 6-138 mm; the low end is worth a look, the high end is not. |
| `index_mismatch` | 18 | **Real and structural.** See §10c. |
| `tcp_below_table` | 18 | **Real, and localized.** Every one is in the *transport* segment; the approach is clean by construction after P9. See §10d. |
| `branch_flip` | 6 | Real. |

**Do not read a low count as a bug without evidence.** Three real bugs were found
and fixed getting from 1/106 to 13/106, so the instinct was right once — but the
way they were found was `tests/test_demo_plan.py`, which runs the pipeline on a
**synthetic demo built to match the scene exactly**. If that file's 24 tests pass,
the transform is sound and what is left is geometry. Extend those tests rather
than staring at real-data failures.

### 10c. The `dz` mismatch is the structural limit

The scene's cube -> place-target motion, which is the retrieval query, against
what the demos supply:

| | scene needs | demos (median) | |
|---|---|---|---|
| `dist_xy` | 0.180 m | 0.182 m | **matches well** |
| `dz` | 0.093 m | 0.041 m | **~5 cm short** |

Humans lift an object a few centimetres to drop it in a bin; our bin rim is 8 cm
up. A demo whose index is off by `e` starts its transport segment `e` away from
the cube, and the §8 step 5 bridge has to cover that **while holding the cube**,
inside a workspace ~25 cm across.

`exec.place_clearance` was lowered 0.06 -> 0.035 for exactly this reason, which is
most of the way to the floor: 0.025 leaves the cube only 2.5 mm above the rim.
Measured demos within 0.10 m of the scene index, by `place_clearance`:
0.060 -> 53, 0.045 -> 62, 0.035 -> 63, 0.025 -> 67. **This lever is nearly spent.**

The remaining lever is the **bin rim height** (`scene.bin.height`, 0.08). Lowering
it moves `need_dz` down one-for-one and is by far the highest-leverage single
change available — but it is a change to a physical prop, so it is a human
decision. Raise it if the feasible count has to go up.

### 10d. `tcp_below_table` is all in the transport, by design

P9 corrects the *approach* anchor's z so the lowest point of the descent lands on
the cube centre; the approach now clears the table by construction (verified: min
z equals the cube centre z exactly). The release anchor is deliberately left as
§8 specifies, because it is a *commanded* height rather than an observation —
the reasoning is in P9. The price is these 18 demos, where the human's carry
dipped below their own release height.

If that trade looks wrong, the symmetric fix is one line in
`common/demos.with_grasp_anchor_z`'s sibling — but note it shrinks `dz`, which
§10c says is already the binding constraint. **Measure before assuming it helps.**

### 10e. Physics failures look different from screen failures

Of the 13 feasible demos run through `run_demo` at the nominal pose: 6 succeed,
3 leave the cube **exactly at its spawn pose** (the gripper never gripped it),
1 nudges it a few mm, 3 are `unsafe_trajectory`.

"Cube at its spawn pose" was the signal that found the P9 window bug, and it is
worth watching for: it means the commanded grasp was not where the cube was. The
screen cannot see this at all, because the screen has no cube.

---

## 11. Step 5 findings — still all valid

Everything here was **measured**, and every number contradicts something that was
guessed before. Full write-up of the IK items: `docs/conventions_proposals.md` P6.
Step 6 confirmed every number here.

### 11a. The usable top-down workspace is small

At the chosen mount height, the region where the right arm can reach the table
top-down at the grasp height is

```
x in [0.15, 0.40]   y in [-0.50, +0.10]
```

about **25 cm across**. The handoff's guessed `scene` values were partly *outside*
it (`pick_zone` reached x = 0.50, `spot` sat at x = 0.52); they have been replaced
from the measured map in `sim/plots/reachability_right.{png,json}`.

**Why this matters for step 6:** the 106 kept demos have `dist_xy` of 7–47 cm
(median 18). The median fits; the long tail cannot. Expect a substantial fraction
to be rejected on geometry alone, independently of the transform being correct.
Budget for that before reading a low feasible-count as a bug.

### 11b. Poses that solve individually often cannot be *connected*

This cost most of step 5. Three successive checks were each insufficient:

| Check | Why it was not enough |
|---|---|
| both endpoints reachable | the straight line between them left the reachable set mid-way (53–72% of samples converged) |
| every sample converges | a converged path still contained a **branch flip**: 2.3 rad of `joint_1` between adjacent 50 Hz samples, 48x the §10 step limit |
| each leg validated separately | each leg got its own cold start; execution solves one concatenated path, warm-started |

What works is validating the plan **exactly as it will be executed** — one
concatenated path, warm-started from the post-bridge configuration, checked for
convergence *and* for §10 joint steps. `sim/execute.py` `_plan_ok` does this, and
the yaw search is built on it. **Don't simplify that back to endpoint checks.**

Consequences baked into the code:

* **HOME → first waypoint runs in joint space**, not tcp space. HOME is tucked and
  in a different IK branch, so interpolating the pose walks out of the reachable
  set (24–70 mm error on every attempt).
* **`SimRobot.move_to` stretches the requested duration** to respect the §10 tcp
  speed limit, because a joint-space move's tcp arc length is not known up front.
  The requested duration is a floor, not a promise.
* **The carry turns the wrist in place, then translates** at constant yaw. Doing
  both at once leaves the reachable set.
* **Grasp yaw has exactly two legal choices** (`cube_yaw` mod 90°); a single fixed
  choice is reachable for only ~80% of pick-zone poses. The release yaw is free and
  is searched over the **full circle** — `yaw` and `yaw + 180°` are the same jaw
  line but different arm configurations, and `joint7` (±90°) usually reaches only
  one of them.

### 11c. The bin belongs on the arm's own side

Measured fraction of pick-zone cube poses whose carry solves as a constant-yaw
Cartesian path, by bin centre:

| bin centre | (0.26,−0.02) | (0.34,−0.10) | (0.22,−0.08) | (0.16,−0.12) | (0.38,−0.22) | (0.36,−0.38) | (0.18,−0.45) | (0.20,−0.46) |
|---|---|---|---|---|---|---|---|---|
| ok | 0/12 | 0/12 | 4/12 | 7/12 | 7/12 | 8/12 | **10/12** | **10/12** |

A bin near the centreline forces a different wrist branch, so the carry has to
rotate ~110° while travelling. The bin is now at negative y.

### 11d. Gravity compensation is an assumption about the real robot

`sim/build_models.py` sets `gravcomp = 1` on every `openarm*` body. Without it the
tcp sags **21 mm** at HOME — 4x the §10 IK tolerance — so no commanded pose is ever
reached. With it, 0.00 mm.

This models a driver with gravity feedforward. **Flag for hardware phase 3:** if
the real driver has none, the simulation is easier than reality and this must come
out, with actuator gains revisited instead.

### 11e. Regenerate derived artifacts when calibration changes

`sim/robot_ik.xml`, `sim/scene.xml` and `sim/ik_seeds_right.npz` are all
**generated**. The seed bank stores tcp positions in the `table` frame, so it is
invalidated by any change to `T_table_robot`.

```sh
uv run python sim/build_models.py     # both models
uv run python sim/reachability.py     # seed bank + reachability map
```

---

## 12. Next handoff (not yours)

MimicGen data generation with rendering → LeRobot datasets → SmolVLA LoRA (plan
phases 6–7). The dataset schema is in `conventions.md` v2 §7.4. Still needed
before it can start: calibrated camera poses from hardware phase 3, and — if P4 is
accepted — a decision on whether skill becomes a third adapter axis alongside the
A/B/C data-source axis.
