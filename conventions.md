# Conventions

| | |
|---|---|
| **conventions_version** | **2** |
| **Timestamp** | 2026-10-17T13:09:39Z |
| **Matches** | `implementation_plan.md` V2.1, `AGENT_HANDOFF.md` H1 |

These conventions apply to all code in `sim/`, `data/`, `perception/`, `runtime/`, `policy/`, plus `common/` (shared code) and `config/` (all YAML).

**If code disagrees with this file, the code is wrong.** Change this file only by team agreement, bump `conventions_version` and add a changelog entry (section 14).

---

## 1. Units

| Quantity | Unit |
|---|---|
| Length | meters |
| Angles (everywhere in our code **and in our datasets**) | radians |
| Angles at the real-robot driver | degrees (LeRobot `OpenArmFollower`), converted **only** inside `RealRobot` and `data/convert_teleop.py` |
| Time | seconds, float64 |
| Gripper command/state | normalized: `0.0` = fully closed, `1.0` = fully open |
| Human grip aperture | meters (thumb tip ↔ index tip distance) |

Numeric arrays are `numpy.float64` in code; datasets may store `float32`. Document array shapes in docstrings.

---

## 2. Coordinate frames

All frames are right-handed and use lowercase names.

| Frame | Origin | Axes | Notes |
|---|---|---|---|
| `table` | Taped origin cross on the table surface, in front of the robot | +x away from the robot, +y to the robot's left, +z up | **World frame.** MuJoCo world frame = `table`. Store and log everything in `table` unless stated. |
| `robot_base` | Root body of the OpenArm v1 model | Model-defined | Pose `T_table_robot` in `config/calibration.yaml` (nominal in sim, measured on hardware, then written back into `sim/scene.xml`). |
| `tcp_left`, `tcp_right` | Center between the gripper finger pads | +z = approach (out of the gripper); +y = jaw-closing axis; +x = y × z | MuJoCo sites `tcp_left`, `tcp_right`. |
| `cam_<name>` | Camera optical center | **OpenCV:** +x right, +y down, +z forward | Pose `T_table_cam_<name>`. |
| `board` | ChArUco board (OpenCV convention) | — | Never used downstream; converted once via `T_table_board`. |
| `arkit` | EgoDex origin (stationary within an episode, on the ground) | +y up | Converted on load; never used downstream. **Not consistent across episodes.** |
| `demo` | Source-demo frame | +z up | EgoDex: z-up version of `arkit` (arbitrary heading). **Teleop: `demo` = `table`.** |
| `cube` | Geometric center | +z up; +x normal to one face | Yaw has 90° symmetry. |
| `bin` | Center of the opening, at rim height | +z up | Place target = `bin` origin + `place_clearance`. |
| `spot` | Center of the marked spot, at table height | +z up | Target for `move_to_spot`. |

**Top-down grasp:** `tcp` +z points along `table` −z.

### MuJoCo camera → OpenCV camera
MuJoCo cameras look along their −z with +y up. Convert once, in `perception/sim_camera.py`:
`R_table_cam_cv = R_table_cam_mj @ diag(1, -1, -1)`.
Sim intrinsics: `f = (H/2) / tan(fovy/2)`, `cx = W/2`, `cy = H/2`, zero distortion.

### EgoDex `arkit` → `demo`
`R_zup = [[1,0,0],[0,0,-1],[0,1,0]]` (arkit +y becomes demo +z), applied as `T_demo_x = T_zup @ T_arkit_x`.

### Hand → `tcp` (HumanEgo construction)
Using the wrist, thumb MCP, index MCP, thumb tip and index tip positions in the `demo` frame:
- `p = (thumbTip + indexTip) / 2`
- `a = ‖thumbTip − indexTip‖`
- `f = normalize(mid(thumbMCP, indexMCP) − wrist)` → **`z_tcp = f`**
- `j = normalize(indexMCP − thumbMCP)`; `j = normalize(j − (j·f) f)` → **`y_tcp = j`**
- **`x_tcp = y_tcp × z_tcp`**; check `det(R) = +1`.
- **Joint names:** index MCP = `{side}IndexFingerKnuckle`. Thumb MCP is expected to be `{side}ThumbIntermediateBase`; **the verified answer is recorded in `data/README.md` and is binding once recorded.**

### Left–right mirroring
Mirroring a right-arm source demo for the left arm uses `M = diag(1, −1, 1)` in the `table` frame:
`p' = M p`, `R' = M R M`. Mirror **before** anchoring (section 8). Mirrored entries carry `arm: "left"` and `mirrored_from: <entry id>`.

---

## 3. Transforms

- **Pose type:** 4×4 homogeneous matrix.
- **Naming:** `T_a_b` = pose of frame `b` in frame `a`; `p_a = T_a_b @ p_b`; `T_a_c = T_a_b @ T_b_c`; `T_b_a = inv_T(T_a_b)`.
- **Quaternions:** `(w, x, y, z)`, matching MuJoCo and mink. SciPy uses `(x, y, z, w)`; convert only via `common/transforms.py`.
- **Serialized poses (JSON/YAML):** `{"pos": [x, y, z], "quat_wxyz": [w, x, y, z]}`. In `.npz`: `(N, 4, 4)`.
- **`yaw_of(T)`** = `atan2(R[1,0], R[0,0])` (heading of the frame's x-axis projected onto the table plane).
- **Helpers in `common/transforms.py`:** `make_T`, `inv_T`, `quat_wxyz_to_mat`, `mat_to_quat_wxyz`, `yaw_of`, `rot_z`, `arkit_to_demo`, `mj_cam_to_cv`, `gripper_frame`, `mirror_lr`.

---

## 4. Robot

- **Model:** OpenArm v1, `v1/openarm_bimanual.xml` from `openarm_mujoco` (commit hash recorded in `sim/README.md`).
- **Model files:**
  - `sim/robot_ik.xml`: robot only, with `tcp_*` sites and **no free objects**. **All IK uses this model.**
  - `sim/scene.xml`: robot, table, objects and cameras. All physics and rendering use this model.
- **Canonical joint names** (match LeRobot): `left_joint_1` … `left_joint_7`, `left_gripper`, `right_joint_1` … `right_joint_7`, `right_gripper`. Mapping to MJCF names: `config/joint_names.yaml`.
- **Robot interface vector (16 values):** `[left_joint_1 … left_joint_7, left_gripper, right_joint_1 … right_joint_7, right_gripper]`.
- **Per-arm dataset vector (8 values):** `[{side}_joint_1 … {side}_joint_7, {side}_gripper]`.
- Never infer joint order from a dataset or from dict ordering.
- **Arm assignment:** single-arm skills use the **right** arm; the left arm holds HOME. The two-arm stretch uses the left arm only with mirrored libraries.
- **HOME:** `config/home.yaml` (16-vector). Every skill starts and ends at HOME.
- **Sim → real (inside `RealRobot` only):** `q_real_deg[i] = degrees(sign[i] * q[i] + offset[i])` (`config/joint_map_real.yaml`); gripper `real = -65 * (1 - g)`, to be verified on hardware.

### Robot interface (`common/robot_interface.py`)
Implemented by `SimRobot` and `RealRobot`. Nothing else touches MuJoCo actuators or LeRobot directly.

| Method | Contract |
|---|---|
| `get_state()` | `{"t": float, "q": (16,)}` |
| `move_to(q_target, duration_s)` | Smooth interpolation to a 16-vector |
| `stream(q_traj, rate_hz)` | Executes `(N,16)`; blocks; `rate_hz` is 50 (retrieval) or 30 (policy) |
| `set_gripper(side, g)` | `side` ∈ {`"left"`, `"right"`}; `g` ∈ [0, 1] |
| `home()` | `move_to(HOME, 3.0)` |

---

## 5. Timing and rates

| Item | Value |
|---|---|
| MuJoCo physics timestep | 0.002 s |
| Retrieval control / streaming | 50 Hz |
| **Policy execution and LeRobot datasets** | **30 Hz** |
| EgoDex native rate | 30 Hz; source demos are resampled to 50 Hz by time |
| 50 Hz → 30 Hz (dataset export) | Resample by time: linear for joints, nearest for gripper and images |
| Retrieval execution speed | `exec.time_scale` in `config/skills.yaml` (0.25 → 0.5 on hardware) |
| Segment bridging | 0.5 s, linear position + slerp rotation |

---

## 6. Perception

- **Images:** RGB `uint8` `(H, W, 3)`. Convert BGR → RGB at capture, inside the camera class only.
- **Pixels:** `(u, v)` floats; origin top-left, u right, v down.
- **Camera names** (identical in sim, real, datasets): `overhead` (required), `wrist_right` (only if the real robot has it; otherwise omitted everywhere).
- **Simulated cameras** use the **calibrated real** intrinsics and poses once available (`config/camera.yaml`, `config/calibration.yaml`).
- **Pixel → table:** ray from the camera through `(u, v)`, intersected with `z = h_obj / 2`.
- **Detector output:** `{"name": str, "T_table_obj": (4,4), "confidence": float}`. Cube yaw from `cv2.minAreaRect`, wrapped to `[-π/4, π/4)`.

---

## 7. Data files

### 7.1 Raw sources
- `data/egodex_raw/` (EgoDex, unmodified, never committed).
- `data/teleop_raw/` (`lerobot-record` output, unmodified).

### 7.2 Source demos: `data/retargeted/{source}_{idx}.npz`, with `source` ∈ {`egodex`, `teleop`}

| Key | Shape/type | Meaning |
|---|---|---|
| `t` | (N,) | Seconds from the demo's start, 50 Hz |
| `T_demo_tcp` | (N,4,4) | Gripper pose in `demo` frame |
| `aperture` | (N,) | Meters (EgoDex); for teleop, gripper value × max opening |
| `grip` | (N,) | 1 = open, 0 = closed |
| `grasp_idx`, `release_idx` | int | Detected (EgoDex) or from the gripper command (teleop) |
| `T_demo_obj0` | (4,4) or absent | **Teleop only:** measured initial cube pose |
| `conf_min` | float | EgoDex only; NaN if confidences are missing |
| `source`, `source_path`, `arm` | str | Provenance; `arm` ∈ {`right`, `left`} |
| `conventions_version` | int | 2 |

### 7.3 Libraries: `data/library/{source}[_left]_v{n}.json`

```
{
  "conventions_version": 2,
  "entries": [{
    "id": str, "source": "egodex"|"teleop", "source_path": str, "arm": "right"|"left",
    "mirrored_from": str|null,
    "anchor_type": "grasp_proxy" | "measured",
    "segments": {"approach": [i0, grasp_idx], "transport": [grasp_idx, i_end]},
    "anchors":  {"T_demo_grasp": pose, "T_demo_release": pose},
    "index":    {"dist_xy": m, "dz": m},
    "sim":      {"n_trials": int, "success_rate": float, "seed": int, "failures": {reason: count}}
  }]
}
```
Runtime uses only entries with `sim.success_rate >= 0.8`.

### 7.4 LeRobot datasets: `data/lerobot/{A|B|C}_v{n}/` (LeRobot format v3)

| Key | Shape | Content |
|---|---|---|
| `fps` | — | 30 |
| `observation.images.overhead` | (224, 224, 3) | RGB |
| `observation.images.wrist_right` | (224, 224, 3) | Only if the camera exists |
| `observation.state` | (8,) | Right-arm joints (rad) + gripper (0–1), per-arm order (section 4) |
| `action` | (8,) | **Absolute** joint targets, same order and units as the state |
| `task` | str | Language instruction |

Per-episode metadata: `origin` ∈ {`sim`, `real`}, `source` ∈ {`egodex`, `teleop`}, `library_entry_id`, `seed`, randomization parameters. **Sim and real episodes must match exactly in keys, shapes, units and order.**

- **A** = sim episodes from the EgoDex library.
- **B** = sim episodes from the teleop library.
- **C** = samples of A + B plus all real teleop episodes.

---

## 8. Object-centric transform (MimicGen)

1. **(Left arm only)** Mirror the source demo (section 2).
2. **Anchors** have position plus yaw only (roll and pitch zeroed).
   - EgoDex: `anchor_type = "grasp_proxy"` (gripper pose at the grasp).
   - Teleop: `anchor_type = "measured"` (`T_demo_obj0` for the approach; gripper pose at release for transport).
3. **Global heading:** rotate the demo about +z around the grasp anchor so its grasp → release heading matches the scene's object → target heading.
4. **Segment transform:** `T_table_tcp[k] = T_table_tgt @ inv_T(T_demo_src) @ T_demo_tcp[k]`.
   - Approach: `src` = grasp (or measured object) anchor; `tgt` = detected cube, yaw snapped to the symmetric yaw (`cube_yaw + n·90°`) closest to the demo's anchor yaw after heading alignment.
   - Transport: `src` = release anchor; `tgt` = place target (bin + `place_clearance`, or spot), yaw = demo release yaw after heading alignment.
5. **Bridging:** 0.5 s from the current pose to the approach start, and 0.5 s between the approach end and the transport start, with the gripper closed.
6. **Selection:**
   - retrieval: nearest `(dist_xy, dz)`;
   - data generation: uniform among the k = 5 nearest.

---

## 9. Configuration files (`config/`)

| File | Contents |
|---|---|
| `calibration.yaml` | `T_table_robot`, `T_table_cam_<name>`, `T_table_board`, measured bin/spot poses |
| `camera.yaml` | Per camera: `K`, distortion, resolution |
| `home.yaml` | HOME as a 16-vector |
| `joint_names.yaml` | Canonical → MJCF joint and actuator names |
| `joint_map_real.yaml` | Per-joint `sign`, `offset`; gripper mapping |
| `skills.yaml` | Pick zones, bin/spot poses (sim nominal), `place_clearance`, `time_scale`, grasp/release thresholds, filter thresholds, validation scene count and seed, retry count |
| `datagen.yaml` | Episodes per dataset, k-nearest, visual/dynamics/noise randomization ranges, seeds, number of processes |
| `policy.yaml` | SmolVLA/LoRA hyperparameters, steps, batch size, action chunk settings, adapter names |
| `safety.yaml` | Limits from section 10 |

---

## 10. Safety limits

Enforced identically in `SimRobot` and `RealRobot`.
- **Gripper-center workspace box (`table` frame):** start with x ∈ [0.15, 0.65], y ∈ [−0.45, 0.45], z ∈ [0.01, 0.45] m; tighten after hardware calibration.
- **Maximum gripper speed:** 0.25 m/s.
- **Maximum joint step per 50 Hz cycle:** 0.05 rad (LeRobot `max_relative_target = 3.0`); scale accordingly at 30 Hz.
- **Reject any trajectory that:** fails the simulation check, leaves the box, has IK error > 5 mm or 3°, violates joint limits, or (two-arm) shows any arm-to-arm contact in the combined simulation.

---

## 11. Success definitions

| Skill | Success |
|---|---|
| `put_in_bin` | Cube inside the bin interior, below the rim, 1 s after release, nearly at rest |
| `move_to_spot` | Cube center within 3 cm of the spot center, 1 s after release |
| `dual_put_in_bin` | Both arms' `put_in_bin` succeed in the same run |
| Grasp miss | Gripper value ≤ 0.05 (fully closed) after the close command → retry |

On hardware, success is judged by the camera verifier; in simulation, by the true object state.

---

## 12. Naming

- snake_case files and functions; frame names as in section 2.
- **Skills:** `put_in_bin`, `move_to_spot`, `dual_put_in_bin`.
- **Libraries:** `egodex_v1`, `teleop_v1`, `egodex_left_v1`.
- **Datasets:** `A_v1`, `B_v1`, `C_v1`. **Adapters:** `smolvla_lora_{A|B|C}_v{n}`.
- **Run IDs:** `{ISO timestamp}_{skill}_{sim|real}`.

---

## 13. Logging and versioning

**Run log:** `runtime/runs.db`, table `runs`:
```
id, run_group, t_start, skill, robot ("sim"|"real"), arm ("right"|"left"),
mode ("retrieval"|"smolvla"), source ("egodex"|"teleop"|"any"|null), adapter (str|null),
library_entry_id (str|null), success (0/1), attempt (1|2), failure_reason,
before_png, after_png
```
Two-arm runs write one row per arm, sharing `run_group`.

**Versioning:** every saved artifact (`.npz`, `.json`, `.yaml`, dataset metadata) includes `conventions_version: 2`.

---

## 14. Changelog

| Version | Timestamp (UTC) | Change |
|---|---|---|
| 1 | 2026-10-17 | Initial conventions |
| 2 | 2026-10-17T13:09:39Z | Added: the explicit HumanEgo `tcp` construction (Gram–Schmidt) and thumb-MCP joint naming; left–right mirroring; `sim/robot_ik.xml` (IK model without free objects); 8-value per-arm dataset vector; 30 Hz policy and dataset rate with the resampling rule; camera names; source-demo schema for both EgoDex and teleop (`T_demo_obj0`, `demo = table` for teleop); library schema with `source`, `arm`, `anchor_type`, `mirrored_from`; LeRobot dataset schema A/B/C; `datagen.yaml` and `policy.yaml`; success definitions; extended run log; naming for libraries, datasets and adapters. Changed: file names `data/retargeted/{source}_{idx}.npz` and `data/library/{source}[_left]_v{n}.json`. |