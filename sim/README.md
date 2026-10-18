# sim

MuJoCo simulation for the OpenArm bimanual setup.

## Model source

The model lives outside this repo in `third_party/openarm_mujoco` (git-ignored, clone it yourself):

```sh
git clone https://github.com/enactic/openarm_mujoco third_party/openarm_mujoco
git -C third_party/openarm_mujoco checkout 36d4a443533478a6a1b6e1d6e0d29b35583afc50
```

| | |
|---|---|
| Upstream | https://github.com/enactic/openarm_mujoco |
| Pinned commit | `36d4a443533478a6a1b6e1d6e0d29b35583afc50` |
| Commit date | 2026-10-16 |
| Subject | Recalculate V1 link7, hand and finger inertias from measured masses (#90) |
| Model file | `third_party/openarm_mujoco/v1/openarm_bimanual.xml` |

Loading `v1/openarm_bimanual.xml` at that commit gives `njnt = 18`, `nu = 16`
(7 arm joints + 2 finger joints per arm; 7 arm actuators + 1 gripper actuator per arm).

## `robot_ik.xml` and `scene.xml` are GENERATED — don't hand-edit them

```sh
uv run python sim/build_models.py
```

Re-run after changing `config/calibration.yaml` (`T_table_robot`, camera poses),
`config/skills.yaml` (`scene`), or the pinned upstream commit.

| File | Contents | Used for |
|---|---|---|
| `robot_ik.xml` | robot only, `tcp_*` sites, no free objects | **all IK** |
| `scene.xml` | + table, cube, bin, spot, cameras, lights | all physics and rendering |

**Why generated rather than `<include>`:** the upstream model declares
`meshdir="meshes"` relative to its own directory, and MuJoCo prefixes an
*included* file's directory **before** applying `meshdir`. An include therefore
resolves meshes to `meshes/third_party/openarm_mujoco/v1/collision/arm/link0.stl`
and fails, and no `meshdir` value can fix it because the inserted prefix comes
first. `build_models.py` loads the upstream file as an `MjSpec` (whose assets
resolve correctly), edits it, compiles it to catch errors, and rewrites the single
`meshdir` attribute in the emitted text so the result loads from `sim/`.

### Changes `build_models.py` makes to the upstream model

| Change | Why |
|---|---|
| `timestep` 0.001 → **0.002** | `conventions.md` section 5 |
| `cone` pyramidal → **elliptic**, `impratio` 1 → **10** | far more stable grasp contacts |
| root body placed at `T_table_robot` | MuJoCo's world frame **is** `table` (section 2) |
| `tcp_left` / `tcp_right` sites | section 2; identity on the upstream `openarm_{side}_hand_tcp` bodies, which already have +z = approach and +y = jaw axis |
| upstream keyframe deleted | it is sized for `nq=18`; `scene.xml` adds a free cube. HOME comes from `config/home.yaml` |
| **`gravcomp = 1` on every `openarm*` body** | see below |

### Gravity compensation is an assumption about the real robot

Without it the position actuators cannot hold the arm up: measured tcp sag at
HOME was **21 mm**, four times the `conventions.md` section 10 IK tolerance, so no
commanded pose would ever be reached. With `gravcomp = 1` the sag is **0.00 mm**.

This models a controller with gravity feedforward. **Flag for hardware phase 3:**
if the real driver has no gravity feedforward, the simulation is easier than
reality and this must be removed (and the actuator gains revisited instead). The
cube is deliberately *not* compensated.

## IK needs its seed bank

`sim/ik_seeds_right.npz` holds a few hundred near-top-down configurations with
their tcp positions; `common/ik.py` seeds cold solves from the nearest ones.
Without it, cold-start IK recovers only ~5% of reachable top-down poses, because
this arm's top-down configurations sit near `joint1 = -1.2, joint2 = +2.9` while
HOME is all zeros. Full write-up: `docs/conventions_proposals.md` P6.

```sh
uv run python sim/reachability.py          # rebuilds the bank + the reachability map
```

**Regenerate the bank whenever `T_table_robot` changes** — the stored tcp
positions are in the `table` frame.

## Headless rendering

`MUJOCO_GL=egl` works on this machine; `MUJOCO_GL=osmesa` also works as a fallback.

```sh
MUJOCO_GL=egl uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('third_party/openarm_mujoco/v1/openarm_bimanual.xml'); r=mujoco.Renderer(m); r.close()"
```

Call `Renderer.close()` explicitly — letting the renderer be collected at interpreter
shutdown raises a harmless `EGLError` from its destructor.
