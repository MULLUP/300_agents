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
