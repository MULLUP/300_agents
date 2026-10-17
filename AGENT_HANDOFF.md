# Agent Handoff: Track A (EgoDex → Retargeting → MimicGen Library)

| | |
|---|---|
| **Handoff version** | **H5** |
| **Timestamp** | 2026-10-17T23:55:00Z |
| **Supersedes** | H4, H3, H2, H1 (all 2026-10-17) |
| **Plan reference** | `implementation_plan.md` **V2.1** (2026-10-17T13:01:09Z) |
| **Conventions reference** | `conventions.md` **v2** (2026-10-17T13:09:39Z) |
| **Mode** | **Simulation only.** No real-robot code, no hardware access. |

> If the plan or conventions version above doesn't match the files, stop and ask a human.

---

## 0. START HERE — the pipeline runs end to end, and it is leaking data

Everything through step 9 is **built, committed and measured**. Test suite: 138
passing, 0 failing. The full pipeline:

```
1077 raw episodes  ->  106 retargeted  ->  62 feasible  ->  20 validated entries
      (step 4 quality filters)  (step 6 geometry)   (step 9 physics)
```

**Gate G4 wants 50 library entries, minimum 20. We are at 20 — the bare floor.**

### 0a. The 20 is not a fact about EgoDex. It is mostly our own filters.

This is the single most important thing in this handoff, and it is the previous
agent's conclusion *against* its own earlier framing. Do not start by tuning
thresholds or hunting for more IK tricks. Start by recovering data we are
discarding for bad reasons. §10 has the four levers, in order of leverage.

The short version: we reject **572 of 1077** episodes for containing more than
one grasp, when a human who re-grips (or picks up three objects in a row) has
given us *more* usable motions, not fewer. We treat one episode as yielding at
most one demo. That is a convenience in `data/retarget.py`, not a property of the
data.

### 0b. The 20 was measured at 8 trials, not the configured 20

`uv run python sim/validate_library.py --trials 8` was run for speed. At 8
trials the 0.8 threshold effectively means >= 7/8. Note the distribution:

```
>= 1.0 :  6      >= 0.8 : 20      >= 0.6 : 30      mean 0.45  median 0.50
```

The entire 6 -> 20 jump is the 7/8 bucket, so the headline number is sensitive to
trial count. **Re-measure at the configured 20 trials before quoting it as the
G4 figure.** It can move either way.

### 0c. Gate G1 is 9/10, and the last one is understood

```
trial 6: cube=(0.261,-0.209) -> FAIL unsafe_trajectory, tcp_speed 0.251-0.274 m/s
```

The `tcp_speed` message is a symptom. `plan_put_in_bin` searches for a smooth
plan, finds none for this cube pose, and then returns its `fallback` — a plan it
has already established fails. Execution inherits a branch flip and
`_fit_rate_limits` stretches the trajectory 27x trying to respect the limits.
**Stretching time cannot fix a discontinuous joint path.**

The pose is at y = -0.2086, at the pick zone's upper y edge (`[-0.34, -0.20]`),
and `config/skills.yaml` already records that `y >= -0.20` has cube yaws with no
legal reachable grasp. Proposed: narrow `pick_zone.size` to `[0.12, 0.12]`.
**Unmeasured.** Honest alternative: accept 9/10 and record it. Either way, do not
let it block §10.

### 0d. Three proposals need a human: P8, P9, P11

All three deviate from `conventions.md` §8 and all three are load-bearing —
without them the feasible count is 0, not lower. They are implemented
config-gated. See `docs/conventions_proposals.md` and §8 here.

---

## 1. Where the previous agents got to

Steps 1-9 are all **built and committed**. None of them is "finished" in the
sense of hitting its target; the numbers are below.

| Step | Status | Key numbers |
|---|---|---|
| 1 Download | OK | 277 `basic_pick_place` + 800 from `part2.zip` (`p2_`); 44 `clean_surface`; 16 `wipe_screen` |
| 2 Loader + joint ID | OK | thumb MCP = `rightThumbIntermediateBase` (evidence in `data/README.md`) |
| 3 Transforms | OK | tests pass |
| 4 Retarget + filters | OK, **G2 cleared** | **106 / 1077 kept (9.8%)** — but see §10a, most of the loss is recoverable |
| 5 Scene, IK, harness | **G1 at 9/10** | see §0c |
| 6 Feasibility | **62 / 106** | was 13 before the yaw work; see §11 |
| 7 Library candidates | OK | `data/build_library.py` -> 62 candidates |
| 8 MimicGen core | OK | `common/mimic.py` |
| 9 Validation | **20 / 62 at >= 0.8** | G4 floor exactly; see §0b and §12 |

**Commits**, oldest first:

| Commit | What |
|---|---|
| `996999b` | 1.P2.2 download, loader, transforms |
| `e576d3d` | 2.P2.1/2.P2.2 retargeting + grasp segmentation, gate G2 |
| `b89aec5` | steps 5, 6, 8: sim foundations, MimicGen core, first feasibility |
| `f4660ac` | free the wrist yaw in IK (`AxisAlignTask`) — measured net-neutral |
| `0267880` | per-segment wrist-spin search -> 62/106 feasible, 31 succeed |
| `7ea898d` | build + score the library -> 20/62 survive |

Working tree clean at `7ea898d`.

---

## 2. Read first, in this order

1. **`conventions.md` v2** — binding. Units, frames, schemas, limits.
2. **`docs/conventions_proposals.md`** — **P1-P11. Read it before touching
   anything.** It records a detector bug the H1 spec would have you reintroduce,
   the three load-bearing §8 deviations (P8/P9/P11), and every threshold's
   evidence.
3. `implementation_plan.md` V2.1, sections 1–5 and phases 1, 2, 4.
4. `tests/test_demo_plan.py` — the synthetic-demo harness. §10b explains why it matters more than it looks.
5. `data/README.md` — the MCP decision and its evidence.
6. `sim/README.md` — the pinned `openarm_mujoco` commit and rendering notes.

---

## 3. Findings from step 4 that are still binding

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

**All four of these are about recovering data we are throwing away.** The
pipeline works; it is leaking. Do them in this order — the order is by measured
or estimated leverage, not by convenience.

### 4a. Stop assuming one demo per episode (biggest lever by far)

`not_one_grasp_release_pair` rejects **572 of 1077 episodes** — more than half the
dataset — because the finger-aperture signal opens and closes more than once.
That rejection is wrong in principle. An episode where someone re-grips, or picks
up three objects in sequence, contains *more* usable pick-and-place motions, not
zero.

`common/segment.py` already computes `flanked_runs` (closed runs with an open
frame on both sides) — it finds all of them and then **discards everything except
the single-pair case**. The change is in `data/retarget.py`: emit **N demos per
episode** instead of 0-or-1, one per flanked run, with unique ids.

Watch out for:
* **Ids.** `data/retarget.py` writes `egodex_{idx}.npz` and `common/demos.demo_id`
  is the file stem. A second demo from the same episode needs something like
  `egodex_{idx}_{k}`. `demo_paths` sorts by a trailing integer — check it still
  orders sensibly.
* **Near-duplicates.** Several runs from one episode may be near-identical. The
  library's `(dist_xy, dz)` index is the natural dedup key.
* `conventions.md` §7.2 names the file pattern. Confirm the suffix is allowed or
  raise it as a proposal.

**Unmeasured.** Nobody has checked how many of the 572 yield at least one valid
flanked run. Measure that first — it bounds the whole exercise and is a cheap
read-only pass over the existing npz inputs.

### 4b. The grasp-speed filter samples the wrong frame (known bug)

`grasp_speed_too_high` rejects **149 episodes** for the hand moving > `max_speed`
0.35 m/s "at the grasp". But **P9 established that `grasp_idx` is the
aperture-crossing frame, which happens mid-descent** — P2 measured median
0.211 m/s there precisely *because* the hand is still falling. The real grasp is
the bottom of the descent, where the hand is near-stationary.

So the filter measures speed at a frame we have already concluded is not the
grasp. `common/demos.with_grasp_anchor_z` already computes the corrected frame
(the z-minimum over `library.grasp_window_s`). Use the same frame for the speed
test. Expect to recover a large share of the 149; re-tune `max_speed` downward
afterwards, since it was loosened from 0.15 to 0.35 (P2) only to compensate for
this bug.

### 4c. Fix the two physics failure modes (one cause is known)

Across the 496 validation trials: `unsafe_trajectory` **128**, `cube_not_in_bin`
**123**, then `no_feasible_yaw` 12, `bridge_unsafe` 6, `ik_error` 3,
`cube_above_rim` 3.

* **`unsafe_trajectory` — cause known.** `fit_yaw_offsets` screens at
  `PLAN_COARSE_FACTOR` (every 5th sample) while execution runs at full 50 Hz, so
  a path passes the screen and still breaks a §10 limit. Either re-screen the
  winning candidate at full density (cheap — one path, not 12) or widen
  `PLAN_MAX_JOINT_STEP`'s margin. Prefer the former; the latter trades one
  false-accept rate for another.
* **`cube_not_in_bin` — not diagnosed. Do not guess.** The cube moves but ends up
  elsewhere. Grasp slip and a systematically short release need *different*
  fixes. Get the per-trial final cube poses out of
  `data/library/egodex_v1_validated.json` and look at the distribution before
  changing anything. A useful tell: a cube left at *exactly* its spawn pose means
  the gripper never gripped it (that signal is what found the P9 window bug), as
  distinct from a cube that travelled and fell short.

### 4d. Download more data

1077 episodes is the `basic_pick_place` test split plus one part file, against a
**5 GB cap** (`data/download_egodex.py --limit --prefix`). Exceeding 5 GB needs a
human (§8). This is last not because it is unimportant but because 4a-4c multiply
whatever is downloaded — doing it first just means re-running everything.

### Then the open items in §0

§0b (re-measure at 20 trials — do this *after* 4c, since 4c changes the answer)
and §0c (gate G1). Both are short.

### Finally, `wipe_surface` — only after a human answers proposal P4

---

## 5. Step 4 on the full pool — gate G2 cleared, but see §4a

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

**Everything below exists and is committed.** Nothing is left to scaffold.

```
common/transforms.py     pose math, conventions.md §2-3. flatten_top_down (P8),
                         spin_about_approach (P11), grasp_pose tilt (P7)
common/segment.py        grasp/release detection + filters. flanked_runs already
                         finds ALL runs -- see §4a
common/demos.py          npz -> SourceDemo. Applies P8 flattening, approach trim
                         and the P9 anchor z. ONE place that knows §7.2
common/mimic.py          the §8 object-centric transform
common/ik.py             mink IK. free_yaw mode (AxisAlignTask), SeedBank
common/safety.py         §10 limits, shared by sim and real
common/config.py         config/*.yaml loaders
common/joints.py         JointIndex; never hardcode joint order
common/robot_interface.py
data/egodex_io.py        HDF5 loader; TCP_JOINT_ROLES is the ONE place the five
                         HumanEgo joint names are encoded
data/retarget.py         step 4. **Emits 0-or-1 demo per episode -- see §4a**
data/build_library.py    step 7, §7.3 schema, `sim` left null
data/download_egodex.py  --limit, --prefix, 5 GB cap
sim/build_models.py      GENERATES robot_ik.xml and scene.xml
sim/reachability.py      GENERATES ik_seeds_right.npz + the reachability map
sim/sim_robot.py         SimRobot: move_to, stream, tcp_path, settle
sim/execute.py           executor, hand-coded put_in_bin, plan_from_demo,
                         fit_yaw_offsets, run_demo
sim/feasibility.py       step 6
sim/validate_library.py  step 9, parallel, exit code 2 if < 20 entries survive
tests/                   138 tests. test_demo_plan.py is the one to extend (§10b)
docs/conventions_proposals.md  P1-P11. Read it.
```

**Generated — never hand-edit, regenerate (§11e):**
`sim/robot_ik.xml`, `sim/scene.xml`, `sim/ik_seeds_right.npz`,
`sim/plots/*`, `data/library/*`.

```sh
uv run python sim/build_models.py   # after ANY scene or calibration change
uv run python sim/reachability.py   # seed bank + reachability map
```

**Still missing:** `policy/`, `perception/` (plan task 1.P3.1),
`data/mirror_library.py` (stretch).

**Raw data on disk** (git-ignored):

| Path | Contents |
|---|---|
| `data/egodex_raw/` | `basic_pick_place`: 277 test-split + 800 `p2_` from `part2.zip` |
| `data/egodex_raw_clean_surface/` | 44 episodes |
| `data/egodex_raw_wipe_screen/` | 16 episodes |

**Note the layout is inconsistent** — `basic_pick_place` is flat in
`egodex_raw/` while the wiping tasks are in sibling `egodex_raw_<task>/` dirs.
With two skills, consider `data/egodex_raw/<task>/` plus a `--task` flag on
`retarget.py`. Left alone so far to avoid breaking an in-flight download.

**A config inconsistency nobody has resolved.** The bin's outer edge is at
y = **-0.528** (`center` -0.44, half-extent 0.088) but `config/safety.yaml`'s
workspace box stops at y = **-0.45**. So part of the bin is outside the box the
§10 check enforces, and 23 of the failing trajectory samples were past that edge.
Either the box or the bin position is wrong. Independent of everything in §4.

**Environment:** `uv`, Python 3.12, all deps installed (incl. `pytest`,
`lerobot`). Run everything with `uv run`. `MUJOCO_GL=egl` works here (`osmesa` is
a verified fallback). Call `Renderer.close()` explicitly — letting it be
collected at shutdown raises a harmless `EGLError`.

Timings, so you can plan: `sim/feasibility.py` over 106 demos ~20 min;
`sim/validate_library.py --trials 8 --processes 8` over 62 candidates ~40 min;
a 20-trial run ~1.5-2 h. **Run long jobs in the background and do not pipe to
`tail`** — you lose the progress stream.

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
- **Thresholds are not the lever in step 4.** The full 27-combination grid was
  swept and the ceiling measured (P1-P3). §4a/§4b are the levers — they recover
  data rather than loosening limits.
- **Don't implement `wipe_surface` until a human answers P4.**
- **Run the synthetic-demo tests before believing a real-data yield.** §10b.
  Three wrong diagnoses in a row came from skipping this.
- **Re-run `sim/build_models.py` after any scene or calibration change**, and
  `sim/reachability.py` too if `T_table_robot` moved (§11e).
- **Append config keys inside the right section.** A key appended to the end of
  `config/skills.yaml` lands under `validation:`, not `ik:` — this silently cost
  two 20-minute runs. Verify with `config.skills()["<section>"]`.
- **Estimate the cost of a sweep before launching it**, and prefer early exit and
  fewer samples: the decision is usually binary, not a precise number.

---

## 8. Escalate to a human when

- fewer than 40 `put_in_bin` demos survive step 4;
- **fewer than 20 library entries survive validation (step 9) — WE ARE AT 20.**
  `sim/validate_library.py` returns exit code 2 below 20 so this cannot pass
  silently. Gate G4's *target* is 50;
- the v1 gripper can't hold the cube after reasonable actuator tuning —
  physics/contact params are a human decision;
- a download would exceed 5 GB;
- anything would require changing `conventions.md`, scope or gates.

**Open proposals needing a human decision.** All are implemented and
config-gated; all three deviations are load-bearing (without them the feasible
count is 0, not merely lower):

| | What | Needs |
|---|---|---|
| **P4** | second skill `wipe_surface` | conventions v3; **do not implement until answered** |
| **P7** | grasp tilt tolerance | conventions v3 (user already approved the idea) |
| **P8** | demos reduced to position + yaw for the whole trajectory, not just anchors. A human's palm direction is a median **71 deg** off top-down | conventions v3 §8 |
| **P9** | the `grasp_proxy` anchor takes its z from the bottom of the descent, not from `grasp_idx` | conventions v3 §8 step 2 |
| **P11** | the place target's yaw is searched per segment rather than fixed to the demo's aligned release yaw | conventions v3 §8 step 4 |

---

## 9. Status board (update as you go)

| Step | Status | Key numbers / notes | Commit |
|---|---|---|---|
| 1 Download | OK | 277 + 800 `p2_` pick_place; 44 + 16 wiping. **More available, §4d** | `996999b` |
| 2 Loader + joint ID | OK | thumb MCP = `rightThumbIntermediateBase` | `996999b` |
| 3 Transforms | OK | tests pass | `996999b` |
| 4 Retarget + filters | OK | **106 / 1077** (9.8%) — G2 target of 100 cleared, but **§4a/§4b say most of the loss is recoverable** | `e576d3d` |
| 5 Scene, IK, harness | **9/10** | last failure understood, §0c | `b89aec5` |
| 6 Feasibility | OK | **62 / 106**; 31/62 succeed at the nominal pose | `0267880` |
| 7 Library candidates | OK | 62 candidates in `data/library/egodex_v1.json` | `7ea898d` |
| 8 MimicGen core | OK | `common/mimic.py` | `b89aec5` |
| 9 Validation | **20 / 62** | at >= 0.8 over **8** trials — **G4 floor exactly**, re-measure at 20 (§0b) | `7ea898d` |
| — test suite | OK | **138 passing, 0 failing** | |
| P4 `wipe_surface` | blocked | awaiting human decision on conventions v3 | |
| P8 / P9 / P11 | blocked | implemented, config-gated, awaiting conventions v3 | |

---

## 10. Steps 6-9 findings — read before touching the yield

### 10a. Where the 1077 actually goes, and whose fault each loss is

| Stage | In -> out | Dominant losses | Genuine, or ours? |
|---|---|---|---|
| 4 quality filters | 1077 -> 106 | `not_one_grasp_release_pair` 572, `grasp_speed_too_high` 149, `transport_too_short` 138, `event_too_close_to_end` 57, `low_confidence` 55 | **Mostly OURS.** 572 is the one-demo-per-episode assumption (§4a); 149 measures speed at the wrong frame (§4b) |
| 6 geometry | 106 -> 62 | `index_mismatch` 18, `tcp_below_table` 13, `approach` 11, `ik_diverged` 2 | `index_mismatch` is genuine (§10c). The rest is mostly real geometry |
| 9 physics | 62 -> 20 | `unsafe_trajectory` 128, `cube_not_in_bin` 123 (of 496 trials) | **Bugs**, one with a known cause (§4c) |

Read that table before optimising anything. The previous agent spent most of a
session on the 106 -> 62 step, which is the *smallest* of the three losses.

### 10b. How to tell a transform bug from real geometry

`tests/test_demo_plan.py` runs the whole pipeline on a **synthetic demo
constructed to match the scene exactly**. If those tests pass, the transform is
sound and what is left is geometry; if a yield collapses, run them first.

This is not a style preference — it is how three of the real bugs were found.
Debugging straight from real-data failures produced **three wrong diagnoses in a
row** (out-of-reach, then warm-start drift, then ill-conditioned yaw), each
disproved by measurement. **Extend this file rather than staring at rejections.**

### 10c. The `dz` mismatch is the one genuinely structural limit

The scene's cube -> place-target motion, against what the demos supply:

| | scene needs | demos (median) | |
|---|---|---|---|
| `dist_xy` | 0.180 m | 0.182 m | **matches well** |
| `dz` | 0.093 m | 0.041 m | **~5 cm short** |

EgoDex humans slide objects across a flat surface with no barrier to clear; our
bin rim is 8 cm up, and the cube's underside has to clear it. A demo whose index
is off by `e` starts its transport segment `e` away from the cube, and the §8
step 5 bridge must cover that **while holding the cube**.

Levers, in order of what is left:

* `exec.place_clearance` already lowered 0.06 -> 0.035. Near the floor: 0.025
  leaves the cube 2.5 mm above the rim.
* **`scene.bin.height` (the rim) was tried and is NOT a free win.** At 0.05 it
  measured *worse* on both metrics (13 -> 11 feasible, 6 -> 4 physics), because
  lowering the release height drops the whole transport segment and costs more in
  table clearance than it gains in index match. Measured, by rim:
  `0.08` -> index<=0.10: 70, below-table 19; `0.05` -> 77 / 29; `0.04` -> 75 / 39.
* **The real unlock is `move_to_spot` instead of `put_in_bin`.** It is already in
  `conventions.md` §11 ("cube centre within 3 cm of the spot centre") and `spot`
  is already in `config/skills.yaml`. No rim means `need_dz` drops to ~0.01 and
  this constraint disappears. The spot is currently only 0.085 m from the pick
  zone against a demo median `dist_xy` of 0.182, so move it further out — and
  keep it at negative y (§11c).

### 10d. The wrist spin is a free variable — this was the big win

13 -> 62 feasible came from one insight: **a top-down cube task does not
constrain the wrist's rotation about its own approach axis, except where the jaws
close.** The evidence, in the order it was established:

* 87% (479/548) of IK-failing samples lie **inside** the region measured as
  81-85% reachable; median position = the pick-zone centre. Not a reach limit.
* 0/548 recover when re-solved cold at the commanded orientation. Not warm-start
  drift.
* yaw along the flattened demos is smooth (3.2 deg median max step) and well
  conditioned. Not the P8 residual.
* **120/120** sampled failures are reachable once the yaw is free — median 8 deg,
  **max 44 deg**. That bound is structural: 45 deg is the cube's yaw symmetry
  half-period.

Two implementations, and the difference matters:

* **Per-sample free yaw** (`IKSolver.solve(free_yaw=True)`, `mink.AxisAlignTask`,
  rank two so it genuinely relaxes rather than reweights). Measured
  **net-neutral**: `ik_diverged` 29 -> 18 and `start_unreachable` 23 -> 16, but
  `branch_flip` 6 -> 23. The solver resolves the 2-dof null space differently at
  adjacent samples. A stronger posture task does **not** fix it: 0.0 -> 4
  feasible, 0.01 -> 6, 0.02 -> 6, 0.05 -> 2, 0.1 -> 0, 0.5 -> 0; above ~0.02 it
  fights the position task (same trade as P6). `branch_flip` holds at 7/34
  regardless.
* **One constant offset per segment** (`fit_yaw_offsets`), ramped across the
  existing §8 mid-bridge. **Continuous by construction**, which is exactly what
  the per-sample version cannot be. 13 -> **62**.

Why the offsets are safe: `spin_about_approach` post-multiplies, so **every
position is untouched** — they cannot move the grasp off the cube, the release
off the bin, or disturb the heading alignment and the index. The approach is
restricted to multiples of 90 deg (the jaws must straddle opposite cube faces,
and `retarget` already snapped to a legal yaw); the transport is unrestricted
(the cube is held, and `cube_in_bin` reads its centre, never its orientation).

Both modes are kept. The per-sample one is still used nowhere by default — it is
retained because it is correct and may help once continuity is solved properly
(a velocity limit, or a path-level solve).

### 10e. The step 6 screen tests ONE cube position. That is a known flaw.

`sim/feasibility.py` places the cube at the pick-zone centre, per plan task
2.P2.3 ("anchored at a nominal cube position"). It is cheap and comparable
between runs, but it is a **weak predictor**: 62 demos passed at that one spot,
and their mean success rate over *random* positions was **0.45**.

Screening against a handful of sampled positions would make step 6 predict step 9
instead of merely preceding it, at 3-5x the screen time. Expect it to cut the 62
substantially — which is fine, because the 62 was already soft.

### 10f. Physics failures look different from screen failures

A cube left at **exactly** its spawn pose means the gripper never gripped it —
the commanded grasp was not where the cube was. The screen cannot see this at
all, because the screen has no cube. That signal is what found the P9 window bug
(taking the anchor z over the whole episode instead of a window near the grasp
put the commanded grasp *below* the cube, so the fingers closed on the table).
Distinguish it from a cube that travelled and fell short — different cause,
different fix.

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
