"""Execute a gripper-pose trajectory, and the hand-coded `put_in_bin` baseline
(plan tasks 1.P1.3, 1.P1.4).

Two layers:

* :func:`execute_pose_path` — the generic executor. Takes a `table`-frame tcp
  pose path plus a gripper channel, solves IK along it (warm-started, so the
  joint path stays continuous), applies the `conventions.md` section 10 checks,
  and streams it. Retrieval (plan phase 5) uses this too.
* :func:`plan_put_in_bin` / :func:`run_put_in_bin` — the hand-coded skill that
  gate G1 measures: approach above the cube, descend, close, lift, carry to the
  bin, release.

    uv run python sim/execute.py --trials 10      # the 1.P1.4 done-criterion
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, safety  # noqa: E402
from common.ik import IKSolver  # noqa: E402
from common.mimic import demo_anchors, heading_rotation, retarget, snap_yaw  # noqa: E402
from common.transforms import (  # noqa: E402
    grasp_pose,
    interp_poses,
    spin_about_approach,
    tilt_about_jaw,
    top_down_pose,
    yaw_of,
)
from sim.sim_robot import SimRobot, UnsafeTrajectory  # noqa: E402


@dataclass
class Waypoint:
    """One leg of a hand-coded skill: go to `pose`, with `grip`, over `duration_s`."""

    pose: np.ndarray
    grip: float
    duration_s: float
    label: str = ""


@dataclass
class ExecResult:
    success: bool
    reason: str = ""
    cube_pose: np.ndarray | None = None
    ik_max_pos_err: float = 0.0
    ik_max_rot_err_deg: float = 0.0
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


def wrap_to_symmetry(yaw: float, period: float = np.pi / 2.0) -> float:
    """Fold an angle into `[-period/2, period/2)` — a cube's yaw is 90 deg-symmetric."""
    return float((yaw + period / 2.0) % period - period / 2.0)


# Candidate yaws over the FULL circle, not [0, 180).
#
# `top_down_pose(p, yaw + 180)` is `top_down_pose(p, yaw)` spun 180 deg about its
# own approach axis, so the two describe the same jaw line and the same grasp —
# but they are different ARM configurations, and joint7's range is only +/-90 deg,
# so typically only one of them is achievable. Collapsing them (as an earlier
# version did) threw away the reachable half and drove IK error from 13 mm to
# 144 mm.
CANDIDATE_YAWS = np.deg2rad(np.arange(0.0, 360.0, 22.5))


def _angle_gap(a: float, b: float) -> float:
    """Smallest absolute angle between two headings, in [0, pi]."""
    return abs(float(np.arctan2(np.sin(a - b), np.cos(a - b))))


def _path_ok(
    solver: IKSolver,
    T_a: np.ndarray,
    T_b: np.ndarray,
    side: str,
    index,
    n: int,
    rate_hz: float,
) -> bool:
    """Is the straight `T_a` -> `T_b` interpolation solvable AND smooth?

    Convergence alone is not enough. A single warm-started solve integrates for
    `max_iters * dt`, which is long enough to wander to a different IK branch
    even when consecutive poses are millimetres apart — measured as a 2.34 rad
    `joint_1` step between adjacent 50 Hz samples. Such a path cannot be executed
    smoothly at all, so the planner has to reject it here rather than let the
    section 10 step check reject it after the fact.

    Validated at the execution sample count, because the step limit is per cycle.
    """
    path = interp_poses(np.asarray(T_a), np.asarray(T_b), n)
    first = solver.solve_cold(path[0], side)
    if not first.converged:
        return False
    q_traj, results = solver.solve_path(path, side, q_seed=first.q)
    if not all(r.converged for r in results):
        return False
    return not safety.check_joint_steps(q_traj, rate_hz, index)


# The plan search validates at a COARSER sampling than execution, or it dominates
# the runtime: at full 50 Hz density one search costs ~14k IK solves and ~1.5 min
# per cube pose, which would make step 7's 106 demos x 20 trials hopeless.
PLAN_COARSE_FACTOR = 5

# A branch flip is ~2 rad; legitimate motion over one coarse interval is about
# `0.05 * PLAN_COARSE_FACTOR` = 0.25 rad. 0.5 rad separates the two cleanly
# without being sensitive to where exactly the coarse samples land.
PLAN_MAX_JOINT_STEP = 0.5


def _fit_rate_limits(
    q_traj: np.ndarray,
    rate_hz: float,
    index,
    side: str,
    tcp_pos: np.ndarray | None = None,
    margin: float = 0.9,
) -> np.ndarray:
    """Resample `q_traj` onto more cycles until it fits the section 10 rate limits.

    Pure time scaling: the joint path is unchanged, it is just traversed over more
    50 Hz cycles, which scales **both** the per-cycle joint step and the tcp speed
    down by the same factor. So one combined stretch factor settles both and a
    single resample is enough. Returns `q_traj` untouched when it already fits.

    `tcp_pos` is the `(N, 3)` tcp path of `q_traj`, from forward kinematics rather
    than from the commanded poses: the limit is checked on where the robot
    actually is, and a 5 mm IK error between adjacent samples is 0.25 m/s of
    apparent speed on its own, the entire budget.
    """
    n = len(q_traj)
    if n < 2:
        return q_traj
    lim = safety.limits()
    stretch = 1.0

    steps = np.abs(np.diff(q_traj[:, index.arm_slots[side]], axis=0))
    if steps.size:
        stretch = max(stretch, float(steps.max()) / (lim.joint_step_limit(rate_hz) * margin))

    if tcp_pos is not None and len(tcp_pos) == n:
        speed = np.linalg.norm(np.diff(np.asarray(tcp_pos), axis=0), axis=1) * float(rate_hz)
        if speed.size:
            stretch = max(stretch, float(speed.max()) / (lim.max_tcp_speed * margin))

    if stretch <= 1.0:
        return q_traj

    n_out = int(np.ceil((n - 1) * stretch)) + 1
    src = np.linspace(0.0, 1.0, len(q_traj))
    dst = np.linspace(0.0, 1.0, n_out)
    out = np.empty((n_out, q_traj.shape[1]), dtype=np.float64)
    for j in range(q_traj.shape[1]):
        out[:, j] = np.interp(dst, src, q_traj[:, j])
    # The gripper is a command, not an angle: take nearest rather than blend, so
    # a close/open edge never becomes a ramp through half-closed.
    g = index.gripper_slot[side]
    out[:, g] = q_traj[np.clip(np.round(dst * (len(q_traj) - 1)).astype(int), 0, len(q_traj) - 1), g]
    return out


def _plan_ok(solver: IKSolver, waypoints: list[Waypoint], side: str, rate_hz: float) -> bool:
    """Is the whole plan solvable and free of branch flips?

    Validating legs *independently* is not equivalent: each would get its own
    cold start, while execution solves one concatenated path warm-started from
    the pose the robot reaches after the joint-space bridge. Two cube poses
    passed per-leg validation and still broke at run time.

    This is a *screen*, not the safety check: it runs at coarse density and uses
    its own step threshold. The executed trajectory is still checked against the
    real `conventions.md` section 10 limits inside `SimRobot.stream`, so a cheap
    screen here cannot let an unsafe trajectory through — it only risks letting a
    bad candidate reach execution and fail there.
    """
    pose_path, _ = densify(waypoints, rate_hz / PLAN_COARSE_FACTOR)
    first = solver.solve_cold(waypoints[0].pose, side)
    if not first.converged:
        return False
    q_traj, results = solver.solve_path(pose_path, side, q_seed=first.q)
    if not all(r.converged for r in results):
        return False

    arm_slots = solver.index.arm_slots[side]
    steps = np.abs(np.diff(q_traj[:, arm_slots], axis=0))
    return bool(steps.max() <= PLAN_MAX_JOINT_STEP) if steps.size else True


def feasible_yaw(
    solver: IKSolver, pos: np.ndarray, side: str, prefer: float = 0.0
) -> float | None:
    """A yaw at which `pos` is reachable top-down, closest to `prefer` first.

    Releasing over the bin does not constrain the wrist at all, and the bin
    centre is only reachable at *some* yaws — measured: the nominal bin position
    solves at 157.5 deg but not at 0. Fixing every waypoint to the grasp yaw
    would therefore fail for a reason that has nothing to do with the task.

    Returns None if no candidate works.
    """
    for yaw in sorted(CANDIDATE_YAWS, key=lambda y: _angle_gap(float(y), float(prefer))):
        if solver.solve_cold(top_down_pose(pos, float(yaw)), side).converged:
            return float(yaw)
    return None


# --- generic executor -------------------------------------------------------


def densify(waypoints: list[Waypoint], rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Waypoints -> a per-cycle `(N, 4, 4)` pose path and `(N,)` gripper channel.

    Poses interpolate (linear position, slerp rotation) between consecutive
    waypoints. The gripper is held at the *incoming* waypoint's value for the
    whole leg and only set at its end, so the fingers never close mid-descent.
    """
    poses: list[np.ndarray] = []
    grips: list[float] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(int(round(b.duration_s * rate_hz)), 2)
        leg = interp_poses(a.pose, b.pose, n)
        poses.append(leg)
        grips.append(np.full(n, a.grip, dtype=np.float64))
    if not poses:
        raise ValueError("need at least two waypoints")
    pose_path = np.concatenate(poses, axis=0)
    grip_path = np.concatenate(grips, axis=0)
    grip_path[-1] = waypoints[-1].grip
    return pose_path, grip_path


def execute_pose_path(
    robot: SimRobot,
    pose_path: np.ndarray,
    grip_path: np.ndarray,
    side: str = "right",
    rate_hz: float | None = None,
    solver: IKSolver | None = None,
    free_yaw: bool | np.ndarray = False,
) -> ExecResult:
    """Solve IK along `pose_path` and stream it. Rejects per conventions section 10.

    `free_yaw` leaves the wrist's rotation about its approach axis to the solver;
    it may be a per-sample mask (see :func:`yaw_locked_mask`).
    """
    cfg = config.skills()["exec"]
    rate_hz = float(cfg["rate_hz"]) if rate_hz is None else float(rate_hz)
    solver = solver if solver is not None else IKSolver()

    q_seed = robot.get_state()["q"]
    q_traj, results = solver.solve_path(pose_path, side, q_seed=q_seed, free_yaw=free_yaw)

    # Carry the commanded gripper channel into the joint trajectory.
    q_traj = q_traj.copy()
    q_traj[:, robot.index.gripper_slot[side]] = grip_path

    # Stretch time until the per-cycle joint step and the tcp speed both fit. The
    # waypoint durations in `config/skills.yaml` are nominal: the tcp path is a
    # straight line but the JOINT path along it is not, so neither rate is known
    # until IK has run. Measured leftovers were small (0.054-0.154 rad against the
    # 0.05 rad limit; 0.274 m/s against 0.25 m/s), and replaying the same geometry
    # more slowly is always valid — this is `exec.time_scale` in section 5.
    q_traj = _fit_rate_limits(q_traj, rate_hz, robot.index, side, robot.tcp_path(q_traj, side))

    pos_err = max(r.pos_err for r in results)
    rot_err = max(r.rot_err_deg for r in results)
    viol = [str(v) for v in safety.check_ik_error(pos_err, rot_err)]
    if viol:
        return ExecResult(False, "ik_error", ik_max_pos_err=pos_err, ik_max_rot_err_deg=rot_err, violations=viol)

    try:
        robot.stream(q_traj, rate_hz)
    except UnsafeTrajectory as exc:
        return ExecResult(
            False,
            "unsafe_trajectory",
            ik_max_pos_err=pos_err,
            ik_max_rot_err_deg=rot_err,
            violations=[str(v) for v in exc.violations],
        )
    return ExecResult(True, "", ik_max_pos_err=pos_err, ik_max_rot_err_deg=rot_err)


# --- the hand-coded put_in_bin ---------------------------------------------


def plan_put_in_bin(
    T_table_cube: np.ndarray,
    side: str = "right",
    solver: IKSolver | None = None,
) -> list[Waypoint]:
    """Waypoints for picking the cube and dropping it in the bin.

    All heights and durations come from `config/skills.yaml` `exec`. The grasp
    yaw follows the cube's own yaw folded into its 90 deg symmetry, so the jaws
    always close on a pair of opposite faces. The *release* yaw is unconstrained
    by the task, so when a `solver` is given it is chosen by reachability (see
    :func:`feasible_yaw`); the gripper rotates to it during the carry.
    """
    cfg = config.skills()["exec"]
    scene = config.skills()["scene"]

    cube_xy = np.asarray(T_table_cube)[:2, 3]
    grasp_yaw = wrap_to_symmetry(float(yaw_of(T_table_cube)))

    # A cube grasp is only legal when the jaws straddle a pair of opposite faces,
    # i.e. yaw == cube_yaw (mod 90 deg). Jaw-line symmetry makes yaw and
    # yaw + 180 the same grasp, so there are exactly TWO distinct choices.
    # Measured: a single fixed choice is reachable for only ~80% of cube poses in
    # the pick zone, so try both.
    grasp_yaw_options = [grasp_yaw, grasp_yaw + np.pi / 2.0]

    # The tcp sits near the fingertips (0.09 m along the hand's +z), so the
    # commanded grasp height is the cube centre plus a tunable offset.
    grasp_z = float(T_table_cube[2, 3]) + float(cfg["grasp_tcp_offset"])
    approach = float(cfg["approach_height"])
    lift = float(cfg["lift_height"])

    bin_centre = np.asarray(scene["bin"]["center"], dtype=np.float64)
    release_z = float(scene["bin"]["height"]) + float(cfg["place_clearance"])

    open_g, close_g = float(cfg["grip_open"]), float(cfg["grip_close"])
    t_move = float(cfg["move_s"])
    t_grip = float(cfg["grip_settle_s"])
    rate_hz = float(cfg["rate_hz"])

    release_pos = np.array([bin_centre[0], bin_centre[1], release_z])

    def build(grasp_y: float, release_y: float, tilt: float = 0.0) -> list[Waypoint]:
        xy = [cube_xy[0], cube_xy[1]]
        above_cube = grasp_pose([*xy, grasp_z + approach], grasp_y, tilt)
        at_cube = grasp_pose([*xy, grasp_z], grasp_y, tilt)
        lifted = grasp_pose([*xy, grasp_z + lift], grasp_y, tilt)
        lift_pos = np.asarray(lifted)[:3, 3]
        return [
            Waypoint(above_cube, open_g, 0.0, "start_above_cube"),
            Waypoint(at_cube, open_g, t_move, "descend"),
            Waypoint(at_cube, close_g, t_grip, "close"),
            Waypoint(lifted, close_g, t_move, "lift"),
            Waypoint(grasp_pose(lift_pos, release_y, tilt), close_g, t_move, "turn"),
            Waypoint(grasp_pose(release_pos, release_y, tilt), close_g, t_move, "carry"),
            Waypoint(grasp_pose(release_pos, release_y, tilt), open_g, t_grip, "release"),
        ]

    if solver is None:  # pure-planning path, used by the tests
        return build(grasp_yaw_options[0], grasp_yaw_options[0])

    # Search (grasp yaw, release yaw) for a plan whose every leg is not just
    # solvable but SMOOTH.
    #
    # Checking endpoints only, or even convergence only, is not enough: both
    # failed here in turn. A leg can solve at every sample and still contain a
    # branch flip — measured 2.3 rad of `joint_1` between adjacent 50 Hz samples,
    # 48x the section 10 step limit. Validating each leg the way it will actually
    # be executed is the only check that correlates with success.
    release_order = sorted(CANDIDATE_YAWS, key=lambda y: _angle_gap(float(y), grasp_yaw))

    # Tilt is the outermost loop and starts at 0, so a strictly vertical grasp is
    # always preferred and the common case exits on the first iteration.
    tilts = [float(np.deg2rad(t)) for t in cfg["grasp_tilt_deg"]]

    fallback = None
    for tilt in tilts:
        for grasp_y in grasp_yaw_options:
            # Skip a grasp yaw whose own pick is impossible, but remember the
            # first candidate that at least *starts* reachable: falling back to
            # an unreachable approach wastes the attempt outright.
            if not _plan_ok(solver, build(grasp_y, grasp_y, tilt)[:4], side, rate_hz):
                if fallback is None and solver.solve_cold(
                    build(grasp_y, grasp_y, tilt)[0].pose, side
                ).converged:
                    fallback = build(grasp_y, grasp_y, tilt)
                continue
            for release_y in release_order:
                plan = build(grasp_y, float(release_y), tilt)
                if _plan_ok(solver, plan, side, rate_hz):
                    return plan
            if fallback is None:
                fallback = build(grasp_y, grasp_y, tilt)

    return fallback if fallback is not None else build(grasp_yaw_options[0], grasp_yaw_options[0])


# --- retargeted source demos (plan tasks 2.P2.3, 4.P2.2) -------------------


def bin_place_target(yaw: float = 0.0, bin_name: str = "bin") -> np.ndarray:
    """The `put_in_bin` place target: bin centre, `place_clearance` above the rim.

    conventions.md section 2: "Place target = `bin` origin + `place_clearance`".
    Only the position and yaw of this pose are ever read (section 8 step 2), so
    `yaw` exists purely to let a caller express a preferred release heading.
    """
    scene = config.skills()["scene"]
    cfg = config.skills()["exec"]
    centre = np.asarray(scene[bin_name]["center"], dtype=np.float64)
    z = float(scene[bin_name]["height"]) + float(cfg["place_clearance"])
    return top_down_pose([centre[0], centre[1], z], float(yaw))


@dataclass
class DemoPlan:
    """A source demo re-anchored onto one scene, densified and ready to stream."""

    pose_path: np.ndarray
    grip_path: np.ndarray
    segments: dict[str, tuple[int, int]]
    heading_delta: float
    release_yaw: float


def _hold(pose: np.ndarray, grip: float, duration_s: float, rate_hz: float):
    n = max(int(round(duration_s * rate_hz)), 1)
    return np.repeat(np.asarray(pose)[None], n, axis=0), np.full(n, float(grip))


def plan_from_demo(
    demo,
    T_table_cube: np.ndarray,
    side: str = "right",
    solver: IKSolver | None = None,
    release_yaw: float | None = None,
) -> DemoPlan:
    """Retarget `demo` onto a cube pose and add the dwells the hardware needs.

    `common.mimic.retarget` produces the geometry of conventions.md section 8.
    Two things have to be added on top of it before it can be streamed:

    * **Finger dwells.** A source demo's `grip` channel flips between samples,
      but real fingers take `exec.grip_settle_s` to travel. The pose is held
      still across each flip, at the grasp and at the release.
    * **The place target's yaw.** conventions.md section 8 step 4 fixes it: "yaw =
      demo release yaw after heading alignment". This is *not* a free choice the
      way it is for the hand-coded planner. There, the waypoints are built at
      whatever yaw :func:`feasible_yaw` likes; here the target's yaw enters the
      rigid transform, so choosing it freely rotates the whole transport segment
      about the bin by up to 180 deg and throws away the step 3 heading alignment
      that was just established. Pass `release_yaw` only to override deliberately.
    """
    cfg = config.skills()["exec"]
    rate_hz = float(cfg["rate_hz"])
    T_cube = np.asarray(T_table_cube, dtype=np.float64)

    if release_yaw is None:
        # The heading alignment depends on the target's POSITION only, so the
        # aligned release yaw can be computed before the target exists — no
        # circularity. Matching it leaves the transport a pure translation of the
        # aligned demo rather than a translation plus a spurious rotation.
        T_grasp, T_release = demo_anchors(demo)
        align = heading_rotation(T_grasp, T_release, T_cube, bin_place_target())
        release_yaw = float(yaw_of(align @ T_release))
    T_tgt = bin_place_target(release_yaw)

    rt = retarget(demo, T_cube, T_tgt, rate_hz)

    # Re-walk the segments, inserting a dwell wherever the gripper command
    # changes. Segment bounds are recomputed as we go so they stay valid indices
    # into the densified output.
    t_grip = float(cfg["grip_settle_s"])
    poses: list[np.ndarray] = []
    grips: list[np.ndarray] = []
    segments: dict[str, tuple[int, int]] = {}

    for label, (i0, i1) in rt.segments.items():
        start = sum(len(p) for p in poses)
        poses.append(rt.T_table_tcp[i0:i1])
        grips.append(rt.grip[i0:i1])
        segments[label] = (start, start + (i1 - i0))

    pose_path = np.concatenate(poses, axis=0)
    grip_path = np.concatenate(grips, axis=0)

    # conventions.md section 11 counts a grasp as missed if the gripper is not
    # closed after the close command, so the dwells are not cosmetic.
    #
    # The dwell goes *after* the first sample that carries the new command, not
    # before it: the fingers have to close on the cube at the grasp pose, not one
    # sample short of it. With the section 7.2 grip channel there are exactly two
    # edges, the close at the end of the approach and the open at the end of the
    # transport, so this inserts one dwell at each.
    out_poses: list[np.ndarray] = []
    out_grips: list[np.ndarray] = []
    shift = np.zeros(len(pose_path) + 1, dtype=int)
    prev = 0
    for e in np.flatnonzero(np.diff(grip_path) != 0.0):
        cut = min(int(e) + 2, len(pose_path))
        out_poses.append(pose_path[prev:cut])
        out_grips.append(grip_path[prev:cut])
        hp, hg = _hold(pose_path[cut - 1], grip_path[cut - 1], t_grip, rate_hz)
        out_poses.append(hp)
        out_grips.append(hg)
        shift[cut:] += len(hp)
        prev = cut
    if prev < len(pose_path):
        out_poses.append(pose_path[prev:])
        out_grips.append(grip_path[prev:])
        # No edge at the very end means the gripper never opened; hold so the
        # fingers have settled before `settle_s` judges the drop.
        hp, hg = _hold(pose_path[-1], grip_path[-1], t_grip, rate_hz)
        out_poses.append(hp)
        out_grips.append(hg)

    return DemoPlan(
        pose_path=np.concatenate(out_poses, axis=0),
        grip_path=np.concatenate(out_grips, axis=0),
        segments={k: (int(i0 + shift[i0]), int(i1 + shift[i1])) for k, (i0, i1) in segments.items()},
        heading_delta=rt.heading_delta,
        release_yaw=float(yaw_of(T_tgt)),
    )


def yaw_locked_mask(grip_path: np.ndarray) -> np.ndarray:
    """Which samples must hold the commanded yaw. True = locked, False = free.

    The wrist's rotation about its own approach axis is only constrained where
    the jaws close on the cube: there the yaw must put them across a pair of
    opposite faces. Once the cube is held, spinning the wrist spins the cube, and
    nothing downstream cares — `cube_in_bin` tests the cube's *centre* and never
    its orientation. Before the grasp the gripper is in free space.

    So the lock covers the closing edge plus the dwell that follows it, and
    nothing else. Freeing the rest is what makes the demos solvable at all:
    0 of 548 failing samples were recoverable at their commanded orientation,
    against 120/120 once the yaw was free.

    Locked slightly *before* the edge too, so the gripper arrives already
    aligned instead of having to swing the wrist in the last cycle, which would
    show up as a joint-step violation.
    """
    grip = np.asarray(grip_path, dtype=np.float64)
    locked = np.zeros(len(grip), dtype=bool)
    lead = max(int(round(float(config.skills()["exec"]["grip_settle_s"]) * 50.0)), 1)
    for e in np.flatnonzero(np.diff(grip) < 0.0):  # closing edges only
        lo = max(int(e) - lead, 0)
        hi = min(int(e) + 2 * lead, len(grip))
        locked[lo:hi] = True
    return locked


# Yaw offsets searched per segment (radians).
#
# The APPROACH is restricted to multiples of 90 deg because the jaws have to
# straddle a pair of opposite cube faces: `retarget` already snapped the grasp
# yaw to a legal one, and a 90 deg multiple keeps it legal. Jaw symmetry makes
# 0/180 and 90/270 the same jaw line but different arm configurations, so all
# four are kept (the same reason `CANDIDATE_YAWS` spans the full circle).
#
# The TRANSPORT is unrestricted: the cube is held, spinning the wrist spins the
# cube, and `cube_in_bin` reads the cube's centre and never its orientation.
APPROACH_YAW_OFFSETS = np.deg2rad([0.0, 90.0, 180.0, 270.0])
TRANSPORT_YAW_OFFSETS = np.deg2rad(np.arange(0.0, 360.0, 45.0))


def _tilt_offsets() -> np.ndarray:
    """Approach elevations to try, from `config/skills.yaml` `exec.demo_tilt_deg`.

    "Top-down" was never a requirement of the task — see
    :func:`common.transforms.tilt_about_jaw`. It was a convention, and an
    expensive one: P7 measured that only **0.59%** of this arm's configurations
    put the approach axis within 10 deg of straight down, against 5.5% within
    30 deg, so insisting on vertical throws away most of the usable workspace and
    is the root cause of the "~25 cm" reachable patch the scene was fitted to.
    """
    return np.deg2rad(np.asarray(config.skills()["exec"]["demo_tilt_deg"], dtype=np.float64))


def _ramped(plan: "DemoPlan", approach: float, transport: float) -> np.ndarray:
    """Per-sample values ramping from an approach to a transport constant.

    Constant-per-segment joined by a linear ramp over the existing section 8
    step 5 mid-bridge. That is what makes these offsets **continuous by
    construction**, which is where the per-sample free-yaw solve failed: it could
    pick a different null-space resolution at every sample, turning 23
    reachability rejects into 23 branch flips.
    """
    n = len(plan.pose_path)
    out = np.empty(n, dtype=np.float64)
    i0, i1 = plan.segments.get("bridge_mid", (n, n))
    out[:i0] = approach
    if i1 > i0:
        out[i0:i1] = np.linspace(approach, transport, i1 - i0)
    out[i1:] = transport
    return out


def apply_yaw_offsets(
    plan: "DemoPlan", approach: float, transport: float, tilt: float = 0.0
) -> np.ndarray:
    """`plan.pose_path` with a per-segment wrist spin and approach tilt.

    Both are post-multiplications, so **positions are untouched** — neither can
    move the grasp off the cube or the release off the bin.

    `tilt` is held constant over the whole path rather than ramped. The jaw axis
    stays horizontal at any tilt, so a constant tilt is legal everywhere, and
    holding it constant keeps the path continuous without spending a second ramp
    on it.
    """
    spun = spin_about_approach(plan.pose_path, _ramped(plan, approach, transport))
    return spun if tilt == 0.0 else tilt_about_jaw(spun, tilt)


def fit_yaw_offsets(
    solver: IKSolver, plan: "DemoPlan", side: str, rate_hz: float
) -> tuple[np.ndarray | None, str]:
    """Search per-segment yaw offsets for a path that is solvable and smooth.

    Staged rather than exhaustive: screen the approach against its 4 offsets
    first, then only the surviving ones against the transport's 8. That is at
    most 4 + 8 screens per demo instead of 32, and the approach is the cheaper
    segment to reject on.

    **The winner is then re-screened at full density.** The staged search runs at
    `PLAN_COARSE_FACTOR` to stay affordable, but execution runs at 50 Hz, so a
    branch flip between two coarse samples passes the search and breaks at run
    time — `unsafe_trajectory` was the largest single failure mode in step 9
    (128 of 496 trials) and this is its known cause. The confirmation costs one
    path, not twelve, and a candidate that fails it does not abort the search:
    the next offset pair gets its turn.

    Returns `(pose_path, "")` on success, or `(None, reason)` with the last
    failure so `sim/feasibility.py` can still report a category.
    """
    i_app = plan.segments["approach"][1]
    reason = "no_yaw_offset_fits"

    # Tilt is the outer loop and is tried in the order the config lists it, so a
    # near-vertical approach is still preferred where one is reachable; the tilt
    # is a fallback that opens up the workspace, not a free-for-all.
    for tilt in _tilt_offsets():
        for a in APPROACH_YAW_OFFSETS:
            trial = apply_yaw_offsets(plan, float(a), float(a), float(tilt))
            ok, why = screen_pose_path(solver, trial[:i_app], side, rate_hz)
            if not ok:
                reason = f"approach: {why}"
                continue
            for t in TRANSPORT_YAW_OFFSETS:
                full = apply_yaw_offsets(plan, float(a), float(a + t), float(tilt))
                ok, why = screen_pose_path(solver, full, side, rate_hz)
                if not ok:
                    reason = why
                    continue
                ok, why = screen_pose_path(solver, full, side, rate_hz, stride=1)
                if ok:
                    return full, ""
                reason = f"coarse_screen_optimistic ({why})"
    return None, reason


def screen_pose_path(
    solver: IKSolver,
    pose_path: np.ndarray,
    side: str,
    rate_hz: float,
    free_yaw: bool | np.ndarray = False,
    stride: int = PLAN_COARSE_FACTOR,
) -> tuple[bool, str]:
    """Is a dense tcp path solvable, smooth, and clear of the table?

    The same screen `_plan_ok` applies to the hand-coded waypoints, but taking a
    pose path directly: a retargeted demo has no waypoints. Returns a reason
    string so `sim/feasibility.py` can report *why* a demo is infeasible rather
    than just how many were.

    `free_yaw` is passed through to the solver and may be a per-sample mask; see
    :func:`yaw_locked_mask`.

    `stride` sub-samples the path. The default screens every 5th sample, which is
    what makes a 12-candidate yaw search affordable; `stride=1` screens at the
    density execution actually runs at. `PLAN_MAX_JOINT_STEP` is a *branch flip*
    detector (~2 rad) rather than the section 10 per-cycle limit (0.05 rad), so
    the same threshold is meaningful at either stride — it just gets stricter
    about where a flip can hide.
    """
    min_z = float(config.skills()["library"]["min_tcp_z"])
    z = pose_path[:, 2, 3]
    if z.min() < min_z:
        return False, f"tcp_below_table (min z {z.min():.3f} < {min_z})"

    coarse = pose_path[::stride]
    mask = np.broadcast_to(np.asarray(free_yaw, dtype=bool), (len(pose_path),))
    coarse_mask = mask[::stride]
    if len(coarse) < 2:
        coarse, coarse_mask = pose_path, mask
    first = solver.solve_cold(coarse[0], side, free_yaw=bool(coarse_mask[0]))
    if not first.converged:
        return False, f"start_unreachable ({first.pos_err * 1000:.0f} mm)"

    q_traj, results = solver.solve_path(coarse, side, q_seed=first.q, free_yaw=coarse_mask)
    worst = max(results, key=lambda r: r.pos_err)
    if not all(r.converged for r in results):
        n_bad = sum(not r.converged for r in results)
        return False, f"ik_diverged ({n_bad}/{len(results)} samples, worst {worst.pos_err * 1000:.0f} mm)"

    steps = np.abs(np.diff(q_traj[:, solver.index.arm_slots[side]], axis=0))
    if steps.size and steps.max() > PLAN_MAX_JOINT_STEP:
        unit = "coarse step" if stride > 1 else "cycle"
        return False, f"branch_flip ({steps.max():.2f} rad over one {unit})"
    return True, ""


def run_demo(
    robot: SimRobot,
    demo,
    cube_xy: np.ndarray,
    cube_yaw: float = 0.0,
    side: str = "right",
    solver: IKSolver | None = None,
) -> ExecResult:
    """One full attempt at `put_in_bin` by replaying a retargeted source demo.

    Same contract as :func:`run_put_in_bin` — reset, place the cube, go to the
    start in joint space, stream, judge — so `sim/validate_library.py` can score
    library entries exactly the way gate G1 scores the hand-coded baseline.
    """
    cfg = config.skills()["exec"]
    cube = config.skills()["scene"]["cube"]
    half = float(cube["size"]) / 2.0

    robot.reset()
    quat = np.array([np.cos(cube_yaw / 2.0), 0.0, 0.0, np.sin(cube_yaw / 2.0)])
    robot.set_free_body_pose("cube_free", [cube_xy[0], cube_xy[1], half], quat)

    solver = solver if solver is not None else IKSolver()
    plan = plan_from_demo(demo, robot.body_pose("cube"), side, solver)

    # Pick the per-segment wrist spin that makes the whole path executable. The
    # offsets leave every position untouched, so this cannot change the task.
    pose_path, why = fit_yaw_offsets(solver, plan, side, float(cfg["rate_hz"]))
    if pose_path is None:
        return ExecResult(False, f"no_feasible_yaw ({why})")

    # HOME -> the demo's first pose in JOINT space, for the reason in
    # `run_put_in_bin`: HOME is tucked and in a different IK branch.
    seed = robot.get_state()["q"]
    ik = solver.solve_cold(pose_path[0], side, extra_seeds=[seed])
    viol = [str(v) for v in safety.check_ik_error(ik.pos_err, ik.rot_err_deg)]
    if viol:
        return ExecResult(
            False,
            "approach_pose_unreachable",
            ik_max_pos_err=ik.pos_err,
            ik_max_rot_err_deg=ik.rot_err_deg,
            violations=viol,
        )

    q_first = ik.q.copy()
    q_first[robot.index.gripper_slot[side]] = plan.grip_path[0]
    try:
        robot.move_to(q_first, float(cfg["bridge_s"]))
    except UnsafeTrajectory as exc:
        return ExecResult(False, "bridge_unsafe", violations=[str(v) for v in exc.violations])

    res = execute_pose_path(robot, pose_path, plan.grip_path, side=side, solver=solver)
    if not res:
        return res

    robot.settle(float(cfg["settle_s"]))
    ok, why = cube_in_bin(robot)
    res.success = ok
    res.reason = "" if ok else why
    res.cube_pose = robot.body_pose("cube")
    return res


def cube_in_bin(robot: SimRobot, cube_body: str = "cube", bin_name: str = "bin") -> tuple[bool, str]:
    """conventions.md section 11: inside the bin interior, below the rim, at rest."""
    scene = config.skills()["scene"]
    bin_centre = np.asarray(scene[bin_name]["center"], dtype=np.float64)
    inner = np.asarray(scene[bin_name]["inner_size"], dtype=np.float64) / 2.0
    rim = float(scene[bin_name]["height"])

    pos = robot.body_pose(cube_body)[:3, 3]
    d = np.abs(pos[:2] - bin_centre)
    if not np.all(d <= inner):
        return False, f"cube at {np.round(pos, 3).tolist()} is outside the bin footprint"
    if pos[2] > rim:
        return False, f"cube z {pos[2]:.3f} is above the rim {rim:.3f}"

    # "nearly at rest" (conventions.md section 11). This has to gate the result,
    # not just the message — an earlier version returned `True, "" if ... else msg`,
    # which granted success no matter how fast the cube was still moving.
    speed = robot.free_body_speed(f"{cube_body}_free")
    at_rest = float(config.skills()["exec"]["at_rest_speed"])
    if speed > at_rest:
        return False, f"cube still moving at {speed:.3f} m/s (limit {at_rest})"
    return True, ""


def run_put_in_bin(
    robot: SimRobot,
    cube_xy: np.ndarray,
    cube_yaw: float = 0.0,
    side: str = "right",
    solver: IKSolver | None = None,
) -> ExecResult:
    """One full hand-coded attempt, from HOME, on a cube placed at `cube_xy`."""
    cfg = config.skills()["exec"]
    cube = config.skills()["scene"]["cube"]
    half = float(cube["size"]) / 2.0

    robot.reset()
    quat = np.array([np.cos(cube_yaw / 2.0), 0.0, 0.0, np.sin(cube_yaw / 2.0)])
    robot.set_free_body_pose("cube_free", [cube_xy[0], cube_xy[1], half], quat)

    solver = solver if solver is not None else IKSolver()
    T_cube = robot.body_pose("cube")
    waypoints = plan_put_in_bin(T_cube, side, solver)

    # Get to the first waypoint in JOINT space, not tcp space.
    #
    # HOME is a tucked pose whose approach axis points up and forward, and it
    # lives in a different IK branch from any top-down grasp (joint1 ~ 0 vs
    # ~ -1.2). Interpolating the *pose* from HOME to above-the-cube walks
    # straight out of the reachable set — measured 24-70 mm IK error on every
    # attempt. A joint-space move has no such problem, and the approach is
    # free space anyway. `SimRobot.stream` still checks the resulting tcp path
    # against the section 10 workspace box.
    first = waypoints[0]
    seed = robot.get_state()["q"]
    ik = solver.solve_cold(first.pose, side, extra_seeds=[seed])
    viol = [str(v) for v in safety.check_ik_error(ik.pos_err, ik.rot_err_deg)]
    if viol:
        return ExecResult(
            False,
            "approach_pose_unreachable",
            ik_max_pos_err=ik.pos_err,
            ik_max_rot_err_deg=ik.rot_err_deg,
            violations=viol,
        )

    q_first = ik.q.copy()
    q_first[robot.index.gripper_slot[side]] = first.grip
    try:
        robot.move_to(q_first, float(cfg["bridge_s"]))
    except UnsafeTrajectory as exc:
        return ExecResult(
            False, "bridge_unsafe", violations=[str(v) for v in exc.violations]
        )

    # Everything from above-the-cube onward is Cartesian: the wrist turns in
    # place at the lift height ("turn"), then translates at constant yaw, so no
    # leg has to rotate and travel at once.
    res = execute_pose_path(
        robot, *densify(waypoints, float(cfg["rate_hz"])), side=side, solver=solver
    )
    if not res:
        return res

    robot.settle(float(cfg["settle_s"]))
    ok, why = cube_in_bin(robot)
    res.success = ok
    res.reason = "" if ok else why
    res.cube_pose = robot.body_pose("cube")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--side", default="right")
    ap.add_argument("--out", type=Path, default=config.path("sim", "plots", "pick_place.json"))
    args = ap.parse_args()

    scene = config.skills()["scene"]
    seed = int(config.skills()["validation"]["seed"]) if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    centre = np.asarray(scene["pick_zone"]["center"], dtype=np.float64)
    size = np.asarray(scene["pick_zone"]["size"], dtype=np.float64)

    robot = SimRobot()
    solver = IKSolver()
    rows, n_ok = [], 0
    for i in range(args.trials):
        xy = centre + rng.uniform(-0.5, 0.5, 2) * size
        yaw = rng.uniform(-np.pi / 4.0, np.pi / 4.0)
        res = run_put_in_bin(robot, xy, yaw, args.side, solver)
        n_ok += bool(res)
        print(
            f"  trial {i + 1:2d}: cube=({xy[0]:.3f},{xy[1]:+.3f}) yaw={np.degrees(yaw):+5.1f} "
            f"-> {'OK ' if res else 'FAIL'} ik={res.ik_max_pos_err * 1000:5.2f}mm "
            f"{res.reason[:70]}",
            flush=True,
        )
        rows.append(
            {
                "trial": i + 1,
                "cube_xy": xy.tolist(),
                "cube_yaw": float(yaw),
                "success": bool(res),
                "reason": res.reason,
                "ik_max_pos_err": res.ik_max_pos_err,
                "ik_max_rot_err_deg": res.ik_max_rot_err_deg,
                "violations": res.violations[:6],
            }
        )

    print(f"\n=== hand-coded put_in_bin: {n_ok}/{args.trials} ===")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "conventions_version": config.CONVENTIONS_VERSION,
                "seed": seed,
                "n_trials": args.trials,
                "n_success": n_ok,
                "trials": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out.relative_to(config.REPO_ROOT)}")

    # Gate G1 (plan section 9) wants 10/10.
    return 0 if n_ok == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
