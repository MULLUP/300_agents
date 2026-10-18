"""Tests for `common/ik.py` (plan task 1.P1.3).

The central property is **round-tripping**: the forward kinematics of a valid
configuration is by construction a reachable pose, so IK must recover *some*
configuration that reproduces it to within the `conventions.md` section 10
tolerances. This is what caught both bugs recorded in
`docs/conventions_proposals.md` P6 (a posture weight that left a 5 mm standing
error, and seeding from HOME).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.ik import IKSolver, SeedBank, seed_bank_path  # noqa: E402
from common.transforms import rot_angle, top_down_pose  # noqa: E402

IK_MODEL = config.path("sim", "robot_ik.xml")

pytestmark = pytest.mark.skipif(
    not IK_MODEL.exists(), reason="run `uv run python sim/build_models.py` first"
)


@pytest.fixture(scope="module")
def solver():
    return IKSolver()


# --- forward kinematics -----------------------------------------------------


def test_fk_matches_the_model_at_home(solver):
    """FK of HOME is the pose `config/home.yaml` documents in its own comments.

    HOME is the *tucked* pose (joint4 = 2.40), not the upstream keyframe: the
    keyframe left the tcp 0.08 m above the table and the left hand inside the
    bin. If this assertion fails, check whether `config/home.yaml` moved before
    assuming the solver broke.
    """
    T = solver.fk(config.home_q(), "right")
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(T[:3, 3], [0.2743, -0.1525, 0.3795], atol=1e-3)
    # The folded elbow leaves the approach axis pointing up and forward, well
    # away from top-down -- this is why section 10's HOME -> first waypoint leg
    # has to run in joint space (handoff section 11b).
    assert T[2, 2] > 0.5, "HOME's approach axis should tilt upward, not at the table"


def test_fk_is_deterministic(solver):
    q = config.home_q()
    np.testing.assert_allclose(solver.fk(q, "right"), solver.fk(q, "right"))


def test_fk_sides_are_mirrored_at_home(solver):
    """HOME is symmetric, so the two tcps differ only in the sign of y."""
    left = solver.fk(config.home_q(), "left")[:3, 3]
    right = solver.fk(config.home_q(), "right")[:3, 3]
    np.testing.assert_allclose(left[0], right[0], atol=1e-9)
    np.testing.assert_allclose(left[1], -right[1], atol=1e-9)
    np.testing.assert_allclose(left[2], right[2], atol=1e-9)


# --- top_down_pose ----------------------------------------------------------


def test_top_down_pose_points_down():
    """conventions.md section 2: tcp +z along table -z."""
    for yaw in np.linspace(-np.pi, np.pi, 9):
        T = top_down_pose([0.3, -0.2, 0.05], yaw)
        np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_top_down_pose_yaw_round_trips_through_yaw_of():
    """conventions.md section 3's `yaw_of` must invert the `yaw` argument.

    The section 8 anchor math aligns headings with `yaw_of` and `rot_z`, so a
    `top_down_pose` whose yaw did not round-trip would silently rotate demos.
    """
    from common.transforms import yaw_of

    for yaw in np.linspace(-np.pi / 2, np.pi / 2, 11):
        assert yaw_of(top_down_pose([0.3, -0.2, 0.05], yaw)) == pytest.approx(yaw, abs=1e-9)


def test_rot_y_is_a_proper_rotation_about_y():
    from common.transforms import rot_y

    for theta in np.linspace(-np.pi, np.pi, 9):
        R = rot_y(theta)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-12)


def test_grasp_pose_zero_tilt_is_top_down():
    from common.transforms import grasp_pose

    for yaw in np.linspace(-np.pi, np.pi, 7):
        np.testing.assert_allclose(
            grasp_pose([0.3, -0.25, 0.05], yaw, 0.0),
            top_down_pose([0.3, -0.25, 0.05], yaw),
            atol=1e-15,
        )


def test_grasp_pose_tilt_angle_is_exact():
    """The approach axis leans by exactly `tilt` off `table` -z (proposal P7)."""
    from common.transforms import grasp_pose

    for tilt_deg in (0.0, 7.5, -7.5, 15.0, -15.0, 30.0):
        T = grasp_pose([0.3, -0.25, 0.05], 0.4, np.deg2rad(tilt_deg))
        off = np.degrees(np.arccos(np.clip(-T[2, 2], -1.0, 1.0)))
        assert off == pytest.approx(abs(tilt_deg), abs=1e-9)


def test_grasp_pose_keeps_the_jaw_axis_horizontal():
    """The whole point of tilting about tcp +y: the jaws stay level.

    If the jaw axis left the table plane, one finger would dip toward the table
    and the jaws would catch the cube's edges instead of its faces.
    """
    from common.transforms import grasp_pose

    for tilt_deg in (0.0, 7.5, -7.5, 15.0, -15.0, 30.0):
        for yaw in np.linspace(-np.pi, np.pi, 7):
            T = grasp_pose([0.3, -0.25, 0.05], yaw, np.deg2rad(tilt_deg))
            assert T[2, 1] == pytest.approx(0.0, abs=1e-12)


def test_grasp_pose_preserves_yaw_and_position():
    """Tilt must not disturb what the section 8 anchor math reads."""
    from common.transforms import grasp_pose, yaw_of

    pos = [0.31, -0.22, 0.0225]
    for tilt_deg in (0.0, 7.5, 15.0, -15.0):
        T = grasp_pose(pos, 0.3, np.deg2rad(tilt_deg))
        assert yaw_of(T) == pytest.approx(0.3, abs=1e-9)
        np.testing.assert_allclose(T[:3, 3], pos)
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_top_down_pose_jaw_axis_is_horizontal():
    """The jaw-closing axis (tcp +y) stays in the table plane at any yaw."""
    for yaw in np.linspace(-np.pi, np.pi, 9):
        assert top_down_pose([0, 0, 0], yaw)[2, 1] == pytest.approx(0.0, abs=1e-12)


def test_top_down_pose_carries_position():
    pos = [0.31, -0.22, 0.0225]
    np.testing.assert_allclose(top_down_pose(pos, 0.7)[:3, 3], pos)


# --- inverse kinematics -----------------------------------------------------


def test_ik_recovers_its_own_fk(solver):
    """A converged solve must reproduce the target within the section 10 limits.

    Seeded near the true configuration, so this tests the *solver*, not seeding.
    """
    rng = np.random.default_rng(11)
    lo, hi = solver.index.arm_limits("right")
    slots = solver.index.arm_slots["right"]
    home = config.home_q()

    n_conv = 0
    for _ in range(25):
        q_true = home.copy()
        q_true[slots] = rng.uniform(lo, hi)
        T = solver.fk(q_true, "right")

        seed = q_true.copy()
        seed[slots] = np.clip(seed[slots] + rng.normal(0.0, 0.1, 7), lo, hi)
        res = solver.solve(T, "right", seed)
        if res.converged:
            n_conv += 1
            # A converged result must actually be accurate, independently measured.
            T_got = solver.fk(res.q, "right")
            assert np.linalg.norm(T_got[:3, 3] - T[:3, 3]) <= solver.pos_tol
            assert np.degrees(rot_angle(T_got[:3, :3], T[:3, :3])) <= solver.rot_tol_deg

    # P6: with posture_weight 0.05 this was 6/7; the regression guard is the rate.
    assert n_conv >= 20, f"only {n_conv}/25 near-seeded solves converged"


def test_ik_leaves_the_other_arm_and_grippers_untouched(solver):
    """Only the target arm's 7 slots may change (the `solve` contract)."""
    home = config.home_q()
    T = solver.fk(home, "right").copy()
    T[2, 3] += 0.02
    res = solver.solve(T, "right", home)

    moved = np.flatnonzero(~np.isclose(res.q, home))
    assert set(moved).issubset(set(solver.index.arm_slots["right"]))


def test_ik_result_q_is_always_a_16_vector(solver):
    """Even a failed solve returns a usable vector, per the IKResult docstring."""
    unreachable = top_down_pose([5.0, 5.0, 5.0], 0.0)
    res = solver.solve(unreachable, "right", config.home_q())
    assert res.q.shape == (16,)
    assert not res.converged
    assert np.all(np.isfinite(res.q))


def test_ik_rejects_an_unknown_side(solver):
    with pytest.raises(ValueError, match="side must be"):
        solver.solve(np.eye(4), "middle", config.home_q())


def test_solve_multi_requires_seeds(solver):
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        solver.solve_multi(np.eye(4), "right", [])


# --- seed bank --------------------------------------------------------------


@pytest.mark.skipif(
    not seed_bank_path("right").exists(), reason="run `uv run python sim/reachability.py` first"
)
def test_seed_bank_entries_are_top_down(solver):
    """Every harvested seed really does point its approach axis downward."""
    bank = SeedBank.load("right")
    assert len(bank) > 0
    assert bank.arm_q.shape[1] == 7
    assert bank.tcp_pos.shape == (len(bank), 3)

    slots = solver.index.arm_slots["right"]
    home = config.home_q()
    for arm_q, pos in zip(bank.arm_q[:25], bank.tcp_pos[:25]):
        q = home.copy()
        q[slots] = arm_q
        T = solver.fk(q, "right")
        np.testing.assert_allclose(T[:3, 3], pos, atol=1e-9)
        tilt_deg = np.degrees(np.arccos(np.clip(-T[2, 2], -1.0, 1.0)))
        assert tilt_deg <= 10.0 + 1e-6


@pytest.mark.skipif(
    not seed_bank_path("right").exists(), reason="run `uv run python sim/reachability.py` first"
)
def test_nearest_returns_k_sorted_by_distance(solver):
    bank = SeedBank.load("right")
    target = np.array([0.30, -0.20, 0.05])
    seeds = bank.nearest(target, 5, solver.index)
    assert len(seeds) == 5
    dists = [np.linalg.norm(solver.fk(q, "right")[:3, 3] - target) for q in seeds]
    assert dists == sorted(dists)
