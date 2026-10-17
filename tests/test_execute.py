"""Tests for the pure planning helpers in `sim/execute.py` (plan task 1.P1.4).

Physics-dependent behaviour (does the gripper actually hold the cube) is measured
by `uv run python sim/execute.py --trials 10`, not asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.transforms import make_T, rot_z, top_down_pose, yaw_of  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.path("sim", "scene.xml").exists(),
    reason="run `uv run python sim/build_models.py` first",
)


@pytest.fixture(scope="module")
def ex():
    from sim import execute

    return execute


# --- wrap_to_symmetry -------------------------------------------------------


def test_wrap_to_symmetry_folds_into_a_quarter_turn(ex):
    """A cube looks identical every 90 deg, so grasp yaw only matters mod 90."""
    for k in range(-4, 5):
        got = ex.wrap_to_symmetry(0.3 + k * np.pi / 2.0)
        assert got == pytest.approx(0.3, abs=1e-9)


def test_wrap_to_symmetry_range(ex):
    for yaw in np.linspace(-2 * np.pi, 2 * np.pi, 97):
        got = ex.wrap_to_symmetry(yaw)
        assert -np.pi / 4.0 - 1e-9 <= got < np.pi / 4.0 + 1e-9


# --- densify ----------------------------------------------------------------


def test_densify_length_and_endpoints(ex):
    a = top_down_pose([0.3, -0.2, 0.20], 0.0)
    b = top_down_pose([0.3, -0.2, 0.05], 0.0)
    wps = [ex.Waypoint(a, 1.0, 0.0, "start"), ex.Waypoint(b, 1.0, 1.0, "down")]
    poses, grips = ex.densify(wps, 50.0)

    assert len(poses) == len(grips) == 50
    np.testing.assert_allclose(poses[0], a, atol=1e-9)
    np.testing.assert_allclose(poses[-1], b, atol=1e-9)


def test_densify_holds_grip_until_the_end_of_a_leg(ex):
    """The fingers must not start closing during the descent."""
    a = top_down_pose([0.3, -0.2, 0.20], 0.0)
    b = top_down_pose([0.3, -0.2, 0.05], 0.0)
    wps = [
        ex.Waypoint(a, 1.0, 0.0, "start"),
        ex.Waypoint(b, 1.0, 1.0, "descend"),
        ex.Waypoint(b, 0.0, 0.5, "close"),
    ]
    poses, grips = ex.densify(wps, 50.0)
    # Open for the whole descent and the close leg, shut only on the final sample.
    assert np.all(grips[:-1] == 1.0)
    assert grips[-1] == 0.0


def test_densify_interpolates_position_monotonically(ex):
    a = top_down_pose([0.3, -0.2, 0.20], 0.0)
    b = top_down_pose([0.3, -0.2, 0.05], 0.0)
    poses, _ = ex.densify([ex.Waypoint(a, 1.0, 0.0), ex.Waypoint(b, 1.0, 1.0)], 50.0)
    z = poses[:, 2, 3]
    assert np.all(np.diff(z) <= 1e-12)


def test_densify_needs_two_waypoints(ex):
    a = top_down_pose([0.3, -0.2, 0.20], 0.0)
    with pytest.raises(ValueError, match="at least two waypoints"):
        ex.densify([ex.Waypoint(a, 1.0, 0.0)], 50.0)


# --- plan_put_in_bin --------------------------------------------------------


def test_plan_is_all_top_down(ex):
    """Every waypoint keeps the approach axis vertical (conventions.md sec 2)."""
    T_cube = make_T(rot_z(0.2), [0.31, -0.22, 0.0225])
    for wp in ex.plan_put_in_bin(T_cube):
        np.testing.assert_allclose(wp.pose[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)


def test_plan_descends_then_lifts_then_releases_over_the_bin(ex):
    cfg = config.skills()["exec"]
    scene = config.skills()["scene"]
    T_cube = make_T(np.eye(3), [0.31, -0.22, 0.0225])
    wps = {w.label: w for w in ex.plan_put_in_bin(T_cube)}

    z = {k: w.pose[2, 3] for k, w in wps.items()}
    assert z["start_above_cube"] == pytest.approx(0.0225 + cfg["approach_height"], abs=1e-9)
    assert z["descend"] == pytest.approx(0.0225, abs=1e-9)
    assert z["lift"] == pytest.approx(0.0225 + cfg["lift_height"], abs=1e-9)

    # Release happens over the bin, clear of the rim.
    np.testing.assert_allclose(wps["release"].pose[:2, 3], scene["bin"]["center"], atol=1e-9)
    assert z["release"] == pytest.approx(scene["bin"]["height"] + cfg["place_clearance"], abs=1e-9)


def test_plan_grip_schedule(ex):
    """Open while approaching, shut from the close leg, open again to release."""
    T_cube = make_T(np.eye(3), [0.31, -0.22, 0.0225])
    wps = ex.plan_put_in_bin(T_cube)
    schedule = [(w.label, w.grip) for w in wps]
    assert schedule[0][1] == 1.0  # approaching, open
    assert dict(schedule)["descend"] == 1.0
    assert dict(schedule)["close"] == 0.0
    assert dict(schedule)["lift"] == 0.0
    assert dict(schedule)["carry"] == 0.0
    assert dict(schedule)["release"] == 1.0


def test_plan_follows_the_cube_yaw(ex):
    """Grasp yaw tracks the cube, folded into the 90 deg symmetry."""
    for yaw in (0.0, 0.3, -0.5):
        T_cube = make_T(rot_z(yaw), [0.31, -0.22, 0.0225])
        wp = ex.plan_put_in_bin(T_cube)[1]
        assert yaw_of(wp.pose) == pytest.approx(ex.wrap_to_symmetry(yaw), abs=1e-6)


# --- cube_in_bin success criterion -----------------------------------------


@pytest.fixture(scope="module")
def robot():
    from sim.sim_robot import SimRobot

    return SimRobot()


def _park_cube_in_bin(robot):
    scene = config.skills()["scene"]
    cx, cy = scene["bin"]["center"]
    robot.set_free_body_pose("cube_free", [cx, cy, scene["cube"]["size"] / 2])


def test_cube_in_bin_accepts_a_resting_cube(ex, robot):
    robot.reset()
    _park_cube_in_bin(robot)
    ok, why = ex.cube_in_bin(robot)
    assert ok, why


def test_cube_in_bin_rejects_a_moving_cube(ex, robot):
    """conventions.md section 11 requires "nearly at rest".

    Regression guard: this check used to be written as
    `return True, "" if speed < 0.05 else msg`, so the ternary bound to the
    message and success was granted no matter how fast the cube was moving.
    """
    import mujoco

    robot.reset()
    _park_cube_in_bin(robot)
    jid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    dadr = int(robot.model.jnt_dofadr[jid])
    robot.data.qvel[dadr : dadr + 3] = [1.0, 0.0, 0.0]

    ok, why = ex.cube_in_bin(robot)
    assert not ok
    assert "still moving" in why


def test_cube_in_bin_rejects_a_cube_outside_the_footprint(ex, robot):
    robot.reset()
    scene = config.skills()["scene"]
    robot.set_free_body_pose("cube_free", [0.30, -0.20, scene["cube"]["size"] / 2])
    ok, why = ex.cube_in_bin(robot)
    assert not ok
    assert "footprint" in why


def test_free_body_speed_reads_the_right_dofs(robot):
    """Regression guard: `qvel[:3]` is the left arm's joints 1-3, not the cube."""
    import mujoco

    robot.reset()
    jid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    dadr = int(robot.model.jnt_dofadr[jid])
    assert dadr >= 3, "cube dofs must not start at 0, or this test proves nothing"

    robot.data.qvel[:] = 0.0
    robot.data.qvel[0:3] = [5.0, 5.0, 5.0]  # left arm moving, cube still
    assert robot.free_body_speed("cube_free") == pytest.approx(0.0)

    robot.data.qvel[dadr : dadr + 3] = [0.0, 3.0, 4.0]
    assert robot.free_body_speed("cube_free") == pytest.approx(5.0)


def test_plan_x_y_track_the_cube(ex):
    T_cube = make_T(np.eye(3), [0.28, -0.31, 0.0225])
    wps = {w.label: w for w in ex.plan_put_in_bin(T_cube)}
    for label in ("start_above_cube", "descend", "lift"):
        np.testing.assert_allclose(wps[label].pose[:2, 3], [0.28, -0.31], atol=1e-9)
