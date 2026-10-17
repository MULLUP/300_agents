# Conventions and handoff proposals

Proposals from implementation work, per handoff rule 6 ("If something there is
wrong or missing, write a proposal in `docs/conventions_proposals.md` and ask.
Don't edit `conventions.md` yourself").

| | |
|---|---|
| Against | `conventions.md` v2, `AGENT_HANDOFF.md` H1, `implementation_plan.md` V2.1 |
| Raised by | Track A agent, step 4 |
| Date | 2026-10-17 |

Status key: **ASK** = needs a human decision. **DONE** = decided in-session (the
decision and who made it is recorded). **FYI** = measurement, no change needed.

---

## P1 — The grasp detector in handoff step 4 does not match the data (DONE: fixed)

**The handoff specifies** `grasp_idx` = first frame after which the aperture stays
below `thr` for >= 8 frames. **This fires on frame 0 in most episodes.**

**Why:** the aperture is *not monotonic in grasp state*. A relaxed human hand sits
at a **small** aperture, so the real signal over an episode is

```
relaxed (small) -> open to grasp (large) -> close on object (small)
-> carry (small) -> open to release (large) -> relaxed (small)
```

**Measured over all 277 test-split episodes:**

| Observation | Count |
|---|---|
| Aperture below `thr` at frame 0 (relaxed hand reads as grasped) | **218 (79%)** |
| Aperture below `thr` at the final frame (post-release relax = phantom 2nd grasp) | **127 (46%)** |
| Both ends below `thr` | 106 (38%) |

So the naive detector grasps at frame 0 (-> `event_too_close_to_end`) and sees the
trailing relax as a second grasp (-> `not_one_grasp_release_pair`).

**Fix applied** in `common/segment.py`: a below-threshold run is only a grasp
candidate if it has an open frame on **both** sides (`flanked_runs`). Runs touching
either end are relaxed-hand artifacts and are not counted. This does not change
any schema or anything in `conventions.md` — the detection *algorithm* is specified
only in the handoff, not in `conventions.md` (section 7.2 says merely "Detected
(EgoDex)").

**Proposal:** fold this into the next handoff's step 4 description so the next
agent does not reimplement the naive version.

---

## P2 — Step 4 filter thresholds loosened under the gate G2 fallback (DONE: user approved)

Plan gate G2's documented fallback is "loosen thresholds (logged), then pull more
from `part2.zip`". Both were applied. Loosened values, all in
`config/skills.yaml` under `retarget.filters` / `retarget.grasp`:

| Parameter | Handoff | Now | Evidence |
|---|---|---|---|
| `min_transport_xy` | 0.10 m | **0.07 m** | Grasp->release distance p25 = 2.5 cm, p50 = 12.1 cm. The 10 cm cut drops 43% of otherwise-valid demos; many real tabletop placements are shorter. |
| `min_confidence` | 0.5 | **0.3** | Only 34% of episodes reach min-confidence >= 0.5; see P3 — one joint causes this. |
| `max_speed` (at grasp) | 0.15 m/s | **0.35 m/s** (proposed) | Speed at the grasp frame: p50 = 0.211 m/s. The 0.15 cut rejects 58% of segmentable demos. The aperture-crossing frame falls during the *descent*, not at rest. Our robot caps at 0.25 m/s (`conventions.md` section 10) so replay is time-scaled regardless. |

**Added, not in the handoff:** `conf_tail_s: 0.5`. Confidence is judged over the
frames the library actually keeps — `[0, release_idx + 0.5 s]` — not the whole
episode. Tracking degrades after the hand retracts and those frames are discarded
by step 7 anyway.

---

## P3 — `rightIndexFingerTip` is the single unreliable joint (FYI, important)

Minimum tracking confidence per joint, median over episodes:

| Joint | `basic_pick_place` | `clean_surface` | `wipe_screen` |
|---|---|---|---|
| `rightHand` (wrist) | **0.977** (97% >= 0.5) | **0.955** (100%) | **0.955** (90%) |
| `rightArm` | 1.000 | 1.000 | 1.000 |
| `rightThumbIntermediateBase` (thumb MCP) | 0.768 | 0.746 | 0.734 |
| `rightIndexFingerKnuckle` (index MCP) | 0.776 | 0.664 | 0.585 |
| `rightThumbTip` | 0.540 | 0.591 | 0.434 |
| **`rightIndexFingerTip`** | **0.404** (40%) | **0.270** (8%) | **0.188** (0%) |

**Consequence:** the HumanEgo construction takes the tcp **position** as
`(thumbTip + indexTip)/2` and the **aperture** as `||thumbTip - indexTip||` — both
from the two *least* reliable joints. The wrist and MCPs, which set the tcp
*orientation*, are solid.

This is the main reason the confidence filter is the binding constraint.

**But it is not the reason segmentation fails.** Recomputing the aperture from
`rightIndexFingerIntermediateTip` (confidence 0.558 instead of 0.404) moves the
"exactly one aperture dip" rate only **50% -> 52%**. The remaining ~50% is
**genuine human behaviour**: people re-adjust their grip mid-carry, producing
several real dips. No threshold choice recovers those episodes.

**Proposal (not applied):** if more `put_in_bin` demos are needed later, consider
deriving the grip signal from finger *curl* (MCP->tip angle, using the
well-tracked MCPs) rather than tip-to-tip distance.

---

## P4 — Second skill `wipe_surface`, grasp-free (DONE: user approved, needs conventions v3)

`implementation_plan.md` section 2 lists wiping as **out of scope**. The user
approved adding it as a second skill after the evidence below. **This needs a plan
and `conventions.md` update that only a human should make.**

**Rationale — wiping removes the entire failure mode above:**

| | `basic_pick_place` | `clean_surface` |
|---|---|---|
| Episodes (test split) | 277 | 44 (+16 `wipe_screen`, +6 `wipe_kitchen_surfaces`, +7 `sweep_dustpan`) |
| Needs grasp detection | yes, ~50% ceiling | **no** — grip is constant (hand holds a cloth) |
| Aperture range (p90-p10) | 2.21 cm, noisy | 1.14 cm, constant |
| Joints required | fingertips, conf **0.40** | **wrist only, conf 0.96** |
| Planarity (s3/s1 of the path) | 0.191 | **0.107** (flat sweep) |
| Episode duration | 3.3 s | 10.0 s |
| Hardware risk | grasp slip, dropped cube | **sponge fixed in the gripper, nothing to drop** |

**What this needs that does not exist yet:**

1. **`conventions.md` section 2** — a `surface` frame (the wipe region) and its
   anchor. The wipe anchor is a *region*, not a point pose.
2. **Section 7.2** — the source-demo schema assumes `grasp_idx`/`release_idx`.
   For a grasp-free skill both are absent and `grip` is constant 0. Proposal: make
   them optional and add `skill: "put_in_bin" | "wipe_surface"` to the npz.
3. **Section 8** — the object-centric transform has no approach/transport split
   for wiping. It is a **single segment** rigidly re-anchored to the detected
   region, which is *simpler* than the pick-place case.
4. **Section 11** — a success definition. Proposal: fraction of the target region
   covered by the tool's swept path >= some threshold, judged on the true state in
   sim.
5. **`sim/scene.xml`** — a flat wipe surface plus a sponge **welded into the right
   gripper** (no grasping needed), and a dirt/target region.
6. **Library naming** — `conventions.md` section 12 names libraries
   `{source}_v{n}`. With two skills they need to be per-skill, e.g.
   `egodex_put_in_bin_v1`, `egodex_wipe_surface_v1`.

**Open question for the human:** does `wipe_surface` get its own LeRobot dataset
and LoRA adapter, or is it mixed into A/B/C? The plan's A/B/C axis is *data
source*; skill is an orthogonal axis. Mixing both into one adapter set muddles the
comparison the challenge slide asks for.

---

## P5 — Gate G2 cannot be met from the test split alone (FYI)

Keep rate after the P1 fix and P2 loosening is ~15% of episodes. The 277-episode
test split therefore yields ~40-45 demos, against the G2 target of 100.
Per the G2 fallback the user approved pulling `basic_pick_place` from
`part2.zip` (800 episodes requested, `p2_` filename prefix to avoid collisions —
every part zip numbers its episodes from 0).

`data/download_egodex.py` now enforces the handoff's 5 GB cap and supports
`--limit` / `--prefix`.

---

## P6 — Two IK settings from the handoff are wrong (DONE: fixed, measured)

Both were found while bringing up `common/ik.py` (step 5). Neither touches
`conventions.md`; both contradict the **handoff's** step 5 text, so they belong in
the next handoff.

### 6a. `posture_weight: 0.05` is not "low" — it costs 5 mm of accuracy

The handoff specifies `posture_weight: 0.05`, annotated "low". Against
`position_cost: 1.0` it is not low: the posture task reaches a **force balance**
with the frame task and the solver settles at a *standing* position error instead
of converging.

Measured on 7 top-down targets over the table, cold-started from the seed bank:

| `posture_weight` | converged | position error (min / median / max) |
|---|---|---|
| **0.05** (handoff) | **6/7** | 2.64 / 3.99 / **5.75 mm** |
| 0.01 | **7/7** | 0.37 / 0.78 / **1.15 mm** |
| 0.001 | 7/7 | 0.01 / 0.93 / 1.37 mm |
| 0.0 | 7/7 | 0.03 / 0.58 / 1.46 mm |

At 0.05 the residual sits exactly on the `conventions.md` section 10 rejection
limit (5 mm), so solutions were being rejected for a reason that had nothing to do
with reachability. **Now 0.01** — still enough nullspace regularization to keep the
joint path continuous, but two orders of magnitude below the frame task.

### 6b. Differential IK cannot be seeded from HOME on this arm

The handoff implies seeding IK from HOME. This arm's **top-down** configurations
live near `joint1 = -1.2, joint2 = +2.9` while **HOME is all zeros**, so a
differential solver started at HOME walks into a joint-limit corner and stops.

| Seeding strategy | Recovery of *provably reachable* poses |
|---|---|
| HOME | **3/60 (5%)** |
| 12 spread top-down seeds + 10 restarts | 18/60 (30%) |
| **nearest-by-tcp from a harvested bank + restarts** | **7/7** on top-down table targets |

"Provably reachable" = the target is the forward kinematics of a random valid
configuration, so a correct solver must find *something*.

**Fix applied:** `common/ik.py` grows a `SeedBank` — a few hundred near-top-down
configurations harvested by rejection-sampling the joint box, stored with their
tcp positions in `sim/ik_seeds_right.npz`, and queried for the `k` nearest in
*task* space. Plus:

* `IKSolver.solve_path()` warm-starts each waypoint from the previous solution.
  This is the mode skills should use — fastest, most reliable, and it keeps the
  joint path continuous so the section 10 per-cycle step limit holds.
* `polish_factor` (0.3): iterate to a tighter target than the acceptance
  threshold, so an accepted solution has margin rather than sitting on the limit.

**Consequence for the next agent:** never call bare `IKSolver.solve()` with a
HOME seed for a grasp pose. Use `solve_cold()` for an isolated target or
`solve_path()` for a trajectory. `sim/ik_seeds_right.npz` must be regenerated
(`uv run python sim/reachability.py`) whenever `T_table_robot` changes, because
the harvested tcp positions are expressed in the `table` frame.

---

## P7 — A grasp needs a few degrees of tilt tolerance (DONE: user approved, needs conventions v3)

**`conventions.md` §2 says:** "**Top-down grasp:** `tcp` +z points along `table`
−z." Read strictly, that means *exactly* vertical.

**Problem:** an exactly-vertical approach is reachable in only a thin slice of
this arm's configuration space. Of 400k uniformly sampled configurations, the
fraction whose approach axis is within a given angle of straight down is

| within | fraction |
|---|---|
| 10° | 0.59% |
| 30° | 5.5% |

and the usable set shrinks further once a *specific* position and a *legal* grasp
yaw are also required. This cost **2 of 10** gate-G1 trials outright: for those
cube poses neither legal grasp yaw (`cube_yaw` mod 90°) was reachable vertically,
giving 11.9 mm and 98.9 mm IK errors — not a solver problem, no vertical solution
exists there.

**Proposal:** allow the approach axis to deviate from `table` −z by a small
tolerance, tried in increasing order so vertical is always preferred.
`config/skills.yaml` `exec.grasp_tilt_deg: [0.0, 7.5, -7.5, 15.0, -15.0]`.

**Why this is safe for the task.** The tilt is applied about the gripper's **own
+y axis**, which is the jaw-closing axis (`common/transforms.py` `grasp_pose`).
Consequences, all verified numerically:

* the jaw axis stays **exactly horizontal** (`jaw_z = 0.000000` at every tilt), so
  the jaws still straddle a cube's vertical faces;
* **both fingers tilt together** — neither dips toward the table, because the
  rotation is in the plane perpendicular to the jaw axis;
* the off-vertical angle equals the requested tilt exactly;
* `yaw_of` is unchanged, so the §8 anchor math still reads the yaw it expects;
* `tilt = 0` reproduces `top_down_pose` bit for bit.

**Suggested conventions v3 wording** for §2, replacing the one-line definition:

> **Top-down grasp:** `tcp` +z points along `table` −z, within
> `exec.grasp_tilt_deg`. The tilt is taken about `tcp` +y so the jaw axis stays
> horizontal. Prefer the smallest tilt that is kinematically reachable.

**Also relevant to step 6.** The same thin-slice problem limits how many of the
106 source demos are feasible. A demo whose grasp happens to be a few degrees off
vertical is currently rejected for that alone, so this tolerance should raise the
feasible count — worth re-measuring when step 6 runs.

---

## P8 — A human's gripper orientation cannot be replayed on this arm (ASK)

**`conventions.md` section 8 is silent on orientation.** Step 4 transform (step 4)
maps full 4x4 poses, so the natural reading is that the human's recorded wrist
orientation is replayed verbatim. **That is not executable here.**

**Why:** section 2's HumanEgo construction sets `z_tcp = normalize(mid(thumbMCP,
indexMCP) - wrist)` — the *palm's forward direction*. A human picks a cube off a
table with the hand pointing forward and down, not straight down. Measured angle
between that axis and `table` -z over the 106 kept demos:

| | min | p25 | med | p75 | max |
|---|---|---|---|---|---|
| at the grasp | 51.2 | 63.1 | **70.7** | 83.8 | 143.4 |
| at the release | 50.2 | 72.1 | **81.7** | 100.3 | 150.1 |
| max over the transport | 57.7 | 79.5 | 91.3 | 114.0 | 151.2 |

degrees. A 71 deg approach puts the fingers into the table, and it is far outside
the top-down map the `config/skills.yaml` `scene` values were measured on
(`AGENT_HANDOFF.md` section 11a). P7's tilt tolerance is +/- 15 deg, nowhere near
enough.

**Fix applied** (`common/transforms.flatten_top_down`, config
`library.orientation: top_down`): keep each pose's position and yaw, force the
approach axis straight down. This is `pos_yaw_pose` applied to the *trajectory*
rather than to the anchor, and it composes safely with section 8 — the retarget
rigid transform is a z-rotation plus a translation, so flattening commutes with
it (`tests/test_demo_plan.py`). `library.orientation: replay` keeps the
alternative measurable via `sim/feasibility.py --orientation replay`.

**What is kept and what is discarded.** Kept: the position path shape, the
timing, and the `(dist_xy, dz)` variety — which is what the library is *for*.
Discarded: the wrist orientation, which is anatomically specific to a human hand
and carries no information this gripper can use.

**Proposal:** state in section 8 that EgoDex source demos are reduced to
position + yaw for their whole trajectory, not only at the anchors. Needs
conventions v3.

**Known residual.** The flattening takes yaw from `yaw_of` (the frame's x-axis
heading). For a near-top-down pose that is fine, but for a pose tilted 71 deg the
tcp x-axis is close to vertical and its horizontal heading is numerically
ill-conditioned. The physically meaningful quantity is the **jaw axis** (tcp +y,
thumb -> index), whose horizontal heading is where the fingers close across the
object. Switching to it is a one-line change in `flatten_top_down` and is
**untested and unmeasured** — flagged, not done.

---

## P9 — `grasp_idx` is mid-descent, so it is the wrong anchor height (ASK)

**`conventions.md` section 8 step 2** defines the EgoDex approach anchor as
`anchor_type = "grasp_proxy"`, "the gripper pose at the grasp". Taken literally
this maps the cube *centre* onto the frame where the hand's aperture crossed its
threshold — and that frame is **not** where the object is.

**Why:** the aperture crossing happens during the descent, not at rest. P2
measured the hand speed there at median **0.211 m/s**. So the hand keeps
descending after `grasp_idx`, by a median of **6.7 mm**, p90 **84 mm**, max
**173 mm** (103 of 106 demos descend further). Anchoring on it sends the rest of
the trajectory through the table: **24 of 106** demos were rejected as
`tcp_below_table`.

**Fix applied** (`common/demos.with_grasp_anchor_z`, new `SourceDemo.grasp_anchor_z`):
the approach anchor keeps the grasp frame's x, y and yaw but takes its **z as the
minimum over `[0, grasp_idx + library.grasp_window_s]`**. The lowest point of the
descent then lands on the cube centre, so table clearance holds by construction,
and `dz` moves toward what the scene needs — demos within 0.12 m of the scene
index go from **81 to 93**.

**The window is load-bearing.** Taking the minimum over the whole kept span
`[0, release_idx]` instead picks up dips belonging to the *carry* and anchors the
commanded grasp **below** the cube; the fingers then close on the table and the
cube is left untouched at its spawn pose — observed on **3 of 6** physics trials
before the window was narrowed to 0.3 s.

**Deliberately asymmetric: the release anchor is left exactly as section 8
specifies.** The justification is that the two anchors are different kinds of
thing. The approach anchor is an *estimate of the object's pose* — that is what
"proxy" means — so correcting it toward where the object actually was is a better
estimate. The release target is a *commanded* height (bin rim +
`place_clearance`), not an observation, so there is nothing to correct. The cost
of that choice is the **18** demos still rejected as `tcp_below_table`, all of
them in the transport segment, where the human's carry dipped below their own
release height.

**Proposal:** redefine the section 8 step 2 `grasp_proxy` anchor as "position
x, y and yaw from the grasp frame; z from the minimum of the descent". Needs
conventions v3.

---

## P10 — `execute_pose_path` enforced the joint-step limit but not the tcp speed limit (DONE: fixed)

Gate G1 failed 9/10 on cube pose `(0.261, -0.209)` with

```
tcp_speed at sample 13556: tcp_right speed 0.251 m/s exceeds 0.25 m/s
... rising to 0.274 m/s
```

Not a geometry failure — a **10% overshoot of the section 10 tcp speed limit**.
`_fit_joint_step_limit` stretched a trajectory in time until the per-cycle *joint*
step fit, but nothing applied the same treatment to the *tcp speed*.
`SimRobot.move_to` already did this for joint-space moves
(`AGENT_HANDOFF.md` section 11b); `execute_pose_path` did not, so every Cartesian
plan was unprotected.

**Fix applied:** `_fit_joint_step_limit` -> `_fit_rate_limits`, which takes both
limits, derives one combined stretch factor and resamples once. Pure time scaling
scales the joint step and the tcp speed down by the same factor, so a single pass
settles both.

The tcp path is taken from **forward kinematics of the solved trajectory**, not
from the commanded poses: a 5 mm IK error between adjacent 50 Hz samples is
0.25 m/s of apparent speed on its own, the entire budget. `SimRobot._fk_path` is
now public as `SimRobot.tcp_path` for this.

No conventions change — this is code catching up with section 10.
