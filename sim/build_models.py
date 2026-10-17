"""Generate `sim/robot_ik.xml` and `sim/scene.xml` (plan tasks 1.P1.1, 1.P1.2).

Both models are **generated, not hand-edited**. Re-run after changing
`config/skills.yaml` (`scene`), `config/calibration.yaml` (`T_table_robot`,
camera poses) or the pinned `openarm_mujoco` commit:

    uv run python sim/build_models.py

Why generated: the upstream model declares `meshdir="meshes"` relative to its own
directory, and MuJoCo's `<include>` prefixes the *included* file's directory
*before* applying `meshdir` — so an include resolves meshes to
`meshes/third_party/.../link0.stl` and fails. Loading the upstream file as a spec
resolves its assets correctly; we then rewrite `meshdir` to a path relative to
`sim/` and serialize. See `sim/README.md`.

Per `conventions.md` section 4:
  * `sim/robot_ik.xml` — robot only, `tcp_*` sites, **no free objects**. All IK.
  * `sim/scene.xml`    — robot, table, objects, cameras. All physics and rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.transforms import mat_to_quat_wxyz, quat_wxyz_to_mat  # noqa: E402

UPSTREAM = config.path("third_party", "openarm_mujoco", "v1", "openarm_bimanual.xml")
SIM_DIR = config.path("sim")

# The upstream model's own `meshdir`, and the equivalent as seen from `sim/`,
# where the generated XMLs live.
UPSTREAM_MESHDIR = "meshes"
MESHDIR_FROM_SIM = "../third_party/openarm_mujoco/v1/meshes"

# conventions.md section 5 / handoff step 5: override the upstream model defaults.

# 0.002 s = 10 physics steps per 50 Hz control cycle, EXACTLY.
#
# 50 Hz is the one rate in this project: source demos are resampled to it
# (`retarget.fps_out`), skills execute at it (`exec.rate_hz`), and the LeRobot
# dataset is written at it. Keeping a single rate end to end means no resampling
# anywhere and no rounding in `SimRobot.steps_per_cycle`.
#
# Do not change this without re-measuring the physics results, and if you ever
# want a capture rate that is not 50 Hz, change it here rather than letting
# `stream` round: at 30 Hz this timestep gives round((1/30)/0.002) = 17 steps,
# i.e. a silent 29.41 Hz. `steps_per_cycle` now rejects that loudly. A timestep
# dividing both 30 and 50 Hz needs a physics rate that is a multiple of 150
# (e.g. 1/600 s), which is a non-terminating decimal and costs 20% more steps.
TIMESTEP = 0.002
IMPRATIO = 10.0


def _robot_base_spec() -> mujoco.MjSpec:
    """The upstream robot, re-rooted into the `table` frame with `tcp_*` sites."""
    spec = mujoco.MjSpec.from_file(str(UPSTREAM))

    # conventions.md section 5 and the handoff: elliptic friction cones with a
    # high impratio give far more stable grasp contacts than the model's
    # pyramidal / impratio=1 defaults.
    spec.option.timestep = TIMESTEP
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO

    # Place the robot at T_table_robot: MuJoCo's world frame IS `table`
    # (conventions.md section 2), so the model root carries the calibration.
    calib = config.calibration()["T_table_robot"]
    root = spec.body("openarm_body_link0")
    root.pos = np.asarray(calib["pos"], dtype=np.float64)
    root.quat = np.asarray(calib["quat_wxyz"], dtype=np.float64)

    # tcp_left / tcp_right (conventions.md section 2: +z approach, +y jaw axis).
    # The upstream `openarm_{side}_hand_tcp` body already has exactly that
    # orientation and sits 0.09 m along the hand's +z, so the sites are identity
    # on those bodies.
    tcp_sites = config.joint_names()["tcp_sites"]
    for side, site_name in tcp_sites.items():
        spec.body(f"openarm_{side}_hand_tcp").add_site(
            name=site_name,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            size=[0.005, 0.005, 0.005],
            group=3,  # group 3 is hidden in the default viewer/render
        )

    # Gravity compensation on the robot's own bodies.
    #
    # Without it the position actuators cannot hold the arm up: measured sag at
    # HOME was 21 mm at the tcp, four times the conventions.md section 10 IK
    # tolerance, so no commanded pose would ever be reached. Raising kp instead
    # would need stiffnesses far outside the upstream model's calibrated values.
    #
    # This models a controller with gravity feedforward, which is what the real
    # arm's torque-mode driver does. It is an ASSUMPTION ABOUT THE REAL ROBOT and
    # is flagged for hardware phase 3: if the real driver has no gravity
    # feedforward, the sim is easier than reality and this must come back out.
    # Objects (the cube) are deliberately NOT compensated.
    for body in spec.bodies:
        if body.name.startswith("openarm"):
            body.gravcomp = 1.0

    # Drop the upstream keyframe: it is sized for nq=18, and `sim/scene.xml` adds
    # a free cube. HOME comes from `config/home.yaml` via `common/config.home_q()`.
    for key in list(spec.keys):
        spec.delete(key)

    return spec


def strip_visual(spec: mujoco.MjSpec) -> tuple[int, int]:
    """Delete visual-only geoms and the meshes left unreferenced. Returns the counts.

    The upstream model ships full-resolution CAD for rendering: **968,553 mesh
    vertices** over 64 meshes, 322 MB of MuJoCo model buffer and ~1.65 GB of RSS
    per loaded model. A batch step that never renders pays that twice per worker
    (once for `SimRobot`, once for `IKSolver`), which at 16 workers is 46 GB and
    gets the run OOM-killed on a 32 GB machine. Stripping leaves 17 collision
    meshes, 64,358 vertices and 24.7 MB.

    **This cannot change the physics**, and `main` asserts as much on every build:
    visual geoms have `contype = conaffinity = 0` so they never collide, and the
    upstream bodies carry explicit inertials so the geoms carry no mass. Verified
    identical: `body_mass`, `body_inertia`, `body_ipos`, `dof_armature`, `nq`,
    `nu`, `nsite` and the HOME contact set.
    """
    visual = [g for g in spec.geoms if g.contype == 0 and g.conaffinity == 0]
    keep = {
        g.meshname
        for g in spec.geoms
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and not (g.contype == 0 and g.conaffinity == 0)
    }
    for geom in visual:
        spec.delete(geom)
    orphans = [m for m in list(spec.meshes) if m.name not in keep]
    for mesh in orphans:
        spec.delete(mesh)
    return len(visual), len(orphans)


def _compile_and_write(spec: mujoco.MjSpec, out_path: Path) -> mujoco.MjModel:
    """Compile (so errors surface here), then serialize for loading from `sim/`.

    `meshdir` is retargeted in the emitted *text*, not on the spec: both
    `compile()` and `to_xml()` resolve relative asset paths against the upstream
    model's own directory, so a spec-level rewrite fails before it can be
    serialized. The upstream spec emits exactly one `meshdir` attribute.
    """
    model = spec.compile()
    xml = spec.to_xml()

    old = f'meshdir="{UPSTREAM_MESHDIR}"'
    if xml.count(old) != 1:
        raise RuntimeError(
            f"expected exactly one {old!r} in the generated XML, found {xml.count(old)}; "
            "the upstream model's compiler element changed — update build_models.py"
        )
    out_path.write_text(xml.replace(old, f'meshdir="{MESHDIR_FROM_SIM}"'))
    return model


def build_robot_ik(out_path: Path) -> mujoco.MjModel:
    """Robot-only IK model. No table, no free objects (conventions.md section 4).

    Always stripped of visual geoms: section 4 gives this model to IK alone, and
    `common/ik.py` runs pure kinematics with no collision task, so nothing here
    can ever be rendered or collided.
    """
    spec = _robot_base_spec()
    strip_visual(spec)
    return _compile_and_write(spec, out_path)


# --- scene ------------------------------------------------------------------

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
PLANE = mujoco.mjtGeom.mjGEOM_PLANE

# Table top: spans the conventions.md section 10 workspace box in x and y, and
# starts at x = 0.10 so it clears the torso, whose collision geometry is only a
# +/-0.03 m column (measured, not assumed).
TABLE_X = (0.10, 0.74)
TABLE_Y = (-0.55, 0.55)  # wider than the workspace box, so the bin never overhangs
TABLE_THICK = 0.04

CUBE_RGBA = (0.85, 0.20, 0.20, 1.0)
CUBE_LEFT_RGBA = (0.20, 0.35, 0.85, 1.0)  # "different color", plan task 1.P1.2
BIN_RGBA = (0.35, 0.35, 0.40, 1.0)
SPOT_RGBA = (0.95, 0.85, 0.15, 1.0)


def _add_table(spec: mujoco.MjSpec) -> None:
    """Table surface at z = 0: the `table` frame origin is the surface itself."""
    half = [
        (TABLE_X[1] - TABLE_X[0]) / 2.0,
        (TABLE_Y[1] - TABLE_Y[0]) / 2.0,
        TABLE_THICK / 2.0,
    ]
    centre = [
        (TABLE_X[0] + TABLE_X[1]) / 2.0,
        (TABLE_Y[0] + TABLE_Y[1]) / 2.0,
        -TABLE_THICK / 2.0,
    ]
    spec.worldbody.add_geom(
        name="table", type=BOX, size=half, pos=centre, rgba=(0.80, 0.76, 0.70, 1.0), friction=[1.0, 0.005, 0.0001]
    )

    # Cosmetic bench surface at the torso's base, so renders have a background.
    # The robot is bench/pedestal-mounted, not standing on the room floor.
    spec.worldbody.add_geom(
        name="bench",
        type=PLANE,
        size=[2.0, 2.0, 0.05],
        pos=[0.0, 0.0, float(config.calibration()["T_table_robot"]["pos"][2])],
        rgba=(0.45, 0.45, 0.48, 1.0),
    )


def _add_cube(spec: mujoco.MjSpec, name: str, centre_xy, rgba) -> None:
    """A free-jointed cube resting on the table, per `skills.yaml` `scene.cube`."""
    cube = config.skills()["scene"]["cube"]
    half = float(cube["size"]) / 2.0
    body = spec.worldbody.add_body(name=name, pos=[centre_xy[0], centre_xy[1], half])
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(
        name=f"{name}_geom",
        type=BOX,
        size=[half, half, half],
        rgba=rgba,
        mass=float(cube["mass"]),
        friction=[float(cube["friction"]), 0.005, 0.0001],
        condim=4,  # torsional friction: a box on a plane otherwise spins freely
    )


def _add_bin(spec: mujoco.MjSpec, name: str, cfg: dict) -> None:
    """Open-topped bin: a floor plus four walls, from `skills.yaml` `scene.bin`."""
    cx, cy = float(cfg["center"][0]), float(cfg["center"][1])
    ix, iy = float(cfg["inner_size"][0]) / 2.0, float(cfg["inner_size"][1]) / 2.0
    h = float(cfg["height"])
    t = float(cfg["wall_thickness"])

    body = spec.worldbody.add_body(name=name, pos=[cx, cy, 0.0])
    body.add_geom(name=f"{name}_floor", type=BOX, size=[ix + t, iy + t, t / 2.0], pos=[0, 0, t / 2.0], rgba=BIN_RGBA)
    for sx, sy, label in ((1, 0, "px"), (-1, 0, "nx"), (0, 1, "py"), (0, -1, "ny")):
        if sx:
            size, pos = [t / 2.0, iy + t, h / 2.0], [sx * (ix + t / 2.0), 0.0, h / 2.0]
        else:
            size, pos = [ix + t, t / 2.0, h / 2.0], [0.0, sy * (iy + t / 2.0), h / 2.0]
        body.add_geom(name=f"{name}_wall_{label}", type=BOX, size=size, pos=pos, rgba=BIN_RGBA)


def _add_spot(spec: mujoco.MjSpec, name: str, cfg: dict) -> None:
    """The `move_to_spot` target: a visual marker only, no collision."""
    cx, cy = float(cfg["center"][0]), float(cfg["center"][1])
    spec.worldbody.add_geom(
        name=name,
        type=CYL,
        size=[float(cfg["radius"]), 0.0005],
        pos=[cx, cy, 0.0005],
        rgba=SPOT_RGBA,
        contype=0,
        conaffinity=0,
    )


def _add_cameras(spec: mujoco.MjSpec) -> None:
    """Cameras at the planned real poses (conventions.md section 6).

    `config/calibration.yaml` stores OpenCV-convention extrinsics; MuJoCo cameras
    look along their own -z with +y up, so convert with the section 2 identity
    `R_table_cam_mj = R_table_cam_cv @ diag(1, -1, -1)`.
    """
    flip = np.diag([1.0, -1.0, -1.0])
    calib = config.calibration()

    cam = calib["T_table_cam_overhead"]
    R_cv = quat_wxyz_to_mat(np.asarray(cam["quat_wxyz"], dtype=np.float64))
    spec.worldbody.add_camera(
        name="overhead",
        pos=np.asarray(cam["pos"], dtype=np.float64),
        quat=mat_to_quat_wxyz(R_cv @ flip),
        fovy=58.0,
    )

    # The wrist camera rides on the hand, so it has no `table`-frame pose. Looks
    # along the approach axis (hand +z) from just behind the gripper.
    spec.body("openarm_right_hand").add_camera(
        name="wrist_right",
        pos=[0.0, -0.06, 0.0],
        quat=mat_to_quat_wxyz(np.eye(3) @ flip),
        fovy=70.0,
    )


def build_scene(out_path: Path, headless: bool = False) -> mujoco.MjModel:
    """Robot + table + objects + cameras. All physics and rendering use this.

    `headless` drops the visual geoms (see :func:`strip_visual`). It is a
    separate file rather than the default because `conventions.md` section 4
    assigns rendering to this model, and plan phase 6 needs it.
    """
    spec = _robot_base_spec()
    scene = config.skills()["scene"]

    _add_table(spec)
    _add_cube(spec, "cube", scene["pick_zone"]["center"], CUBE_RGBA)
    _add_bin(spec, "bin", scene["bin"])
    _add_spot(spec, "spot", scene["spot"])
    _add_cameras(spec)

    # The left zone exists for the two-arm stretch (plan phase 5S). It is
    # generated only when enabled, so flipping the flag and re-running this
    # script is the whole "enable" step — no disabled clutter in the scene.
    left = scene.get("left_zone", {})
    if left.get("enabled"):
        _add_cube(spec, "cube_left", left["pick_zone"]["center"], CUBE_LEFT_RGBA)
        _add_bin(spec, "bin_left", {**scene["bin"], **left["bin"]})

    spec.worldbody.add_light(name="key", pos=[0.4, 0.0, 1.5], dir=[0.0, 0.0, -1.0])
    spec.worldbody.add_light(name="fill", pos=[0.0, -0.8, 1.2], dir=[0.3, 0.6, -1.0])

    if headless:
        strip_visual(spec)
    return _compile_and_write(spec, out_path)


def main() -> int:
    ik_path = SIM_DIR / "robot_ik.xml"
    model = build_robot_ik(ik_path)
    print(f"wrote {ik_path.relative_to(config.REPO_ROOT)}")
    print(f"  njnt={model.njnt} nu={model.nu} nq={model.nq} nsite={model.nsite}")

    scene_path = SIM_DIR / "scene.xml"
    scene = build_scene(scene_path)
    print(f"wrote {scene_path.relative_to(config.REPO_ROOT)}")
    print(
        f"  njnt={scene.njnt} nu={scene.nu} nq={scene.nq} "
        f"nbody={scene.nbody} ncam={scene.ncam} "
        f"timestep={scene.opt.timestep} cone={'elliptic' if scene.opt.cone else 'pyramidal'} "
        f"impratio={scene.opt.impratio:g}"
    )
    _report_scene_contacts(scene_path)

    headless_path = SIM_DIR / "scene_headless.xml"
    headless = build_scene(headless_path, headless=True)
    print(f"wrote {headless_path.relative_to(config.REPO_ROOT)}")
    print(
        f"  nmesh {scene.nmesh} -> {headless.nmesh}, "
        f"nmeshvert {scene.nmeshvert} -> {headless.nmeshvert}, "
        f"model buffer {scene.nbuffer / 1e6:.0f} -> {headless.nbuffer / 1e6:.0f} MB"
    )
    _assert_same_dynamics(scene, headless)
    _report_scene_contacts(headless_path)

    # Re-load from disk: proves the serialized file stands on its own, meshes included.
    check = mujoco.MjModel.from_xml_path(str(ik_path))
    data = mujoco.MjData(check)
    names = config.joint_names()
    q = config.home_q()
    _apply_home(check, data, names, q)
    mujoco.mj_forward(check, data)
    for side in ("left", "right"):
        sid = mujoco.mj_name2id(check, mujoco.mjtObj.mjOBJ_SITE, names["tcp_sites"][side])
        pos = data.site_xpos[sid]
        R = data.site_xmat[sid].reshape(3, 3)
        print(f"  HOME {names['tcp_sites'][side]:>10s}: pos={np.round(pos, 4)} approach_z={np.round(R[:, 2], 3)}")
    return 0


def _assert_same_dynamics(full: mujoco.MjModel, headless: mujoco.MjModel) -> None:
    """Fail the build if stripping visual geoms changed anything physical.

    The whole argument for a headless model is that visual geoms are massless and
    non-colliding, so dropping them is a no-op for physics and IK. That holds for
    the current upstream model, but it is a property of someone else's CAD, not a
    guarantee — so it is checked on every build rather than trusted.
    """
    for field in ("body_mass", "body_inertia", "body_ipos", "dof_armature", "jnt_range"):
        a, b = getattr(full, field), getattr(headless, field)
        if not np.array_equal(a, b):
            worst = float(np.abs(np.asarray(a) - np.asarray(b)).max())
            raise RuntimeError(
                f"stripping visual geoms changed {field} (max delta {worst:.3e}) — "
                "the upstream model's visual geoms now carry mass, so the headless "
                "scene is NOT dynamically equivalent. Do not use it for physics."
            )
    for field in ("nq", "nu", "nv", "nbody", "nsite", "njnt", "ncam"):
        if getattr(full, field) != getattr(headless, field):
            raise RuntimeError(
                f"stripping visual geoms changed {field}: "
                f"{getattr(full, field)} -> {getattr(headless, field)}"
            )
    print("  verified: mass, inertia, ipos, armature, joint ranges and counts unchanged")


def _report_scene_contacts(scene_path: Path) -> None:
    """Warn if the robot starts in contact at HOME — e.g. the table intersecting it."""
    from common.joints import JointIndex

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    JointIndex(model).write_qpos(config.home_q(), data.qpos)
    mujoco.mj_forward(model, data)
    pairs = []
    # A resting object touches at dist ~= 0; only flag real interpenetration.
    penetration_tol = -1e-4
    for i in range(data.ncon):
        con = data.contact[i]
        if con.dist < penetration_tol:
            pairs.append(
                f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)}"
                f" <-> {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)}"
                f" ({con.dist * 1000:.1f} mm)"
            )
    if pairs:
        print(f"  WARNING: {len(pairs)} penetrating contact(s) at HOME: {'; '.join(pairs[:5])}")
    else:
        print(f"  no penetrating contacts at HOME ({data.ncon} contact point(s) total)")


def _apply_home(model, data, names: dict, q: np.ndarray) -> None:
    """Write the 16-vector into qpos using the canonical order (never positional)."""
    for i, canon in enumerate(names["order"]):
        if canon in names["joints"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names["joints"][canon])
            data.qpos[model.jnt_qposadr[jid]] = q[i]
        else:  # a gripper: normalized -> meters, on both coupled finger joints
            travel = float(names["gripper_open_m"]) * float(q[i])
            for key in ("gripper_joints", "gripper_mimic_joints"):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[key][canon])
                data.qpos[model.jnt_qposadr[jid]] = travel


if __name__ == "__main__":
    raise SystemExit(main())
