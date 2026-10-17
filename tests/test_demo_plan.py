"""Tests for the demo -> executable-path pipeline (`sim/execute.py plan_from_demo`).

The point of these is **isolation**. `sim/feasibility.py` screens real EgoDex
demos and reports a single feasible count; when that count is low there is no way
to tell a broken transform from genuinely unreachable human geometry. So the
tests here feed in a *synthetic* demo built to be exactly what the nominal scene
asks for. If the pipeline is correct that demo comes out as a clean top-down
pick-and-place, and any failure is the code rather than the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.demos import trim_approach  # noqa: E402
from common.mimic import SourceDemo  # noqa: E402
from common.transforms import (  # noqa: E402
    flatten_top_down,
    inv_T,
    rot_z_T,
    top_down_pose,
    yaw_of,
)
from sim.execute import bin_place_target, plan_from_demo  # noqa: E402
from sim.feasibility import index_offset, nominal_cube_pose, scene_index  # noqa: E402

RATE = 50.0


def synthetic_demo(
    dist_xy: float,
    dz: float,
    heading: float = 0.0,
    demo_yaw: float = 0.0,
    lift: float = 0.10,
) -> SourceDemo:
    """A clean top-down pick-and-place in an arbitrary `demo` frame.

    Descend onto the object, close, lift `lift`, travel `dist_xy` along `heading`,
    end `dz` above the grasp, open. Positions are in a frame whose origin and
    heading are deliberately *not* the table's, because that is what the section 8
    transform exists to absorb.
    """
    origin = np.array([1.7, -0.9, 0.35])  # arbitrary
    d = np.array([np.cos(heading), np.sin(heading), 0.0])

    def leg(a, b, n):
        u = np.linspace(0.0, 1.0, n)[:, None]
        return a[None, :] + u * (b - a)[None, :]

    grasp_p = origin
    above = origin + [0.0, 0.0, 0.10]
    lifted = origin + [0.0, 0.0, lift]
    release_p = origin + d * dist_xy + [0.0, 0.0, dz]
    over = release_p + [0.0, 0.0, max(lift - dz, 0.0)]

    pts = np.concatenate(
        [leg(above, grasp_p, 20), leg(grasp_p, lifted, 20), leg(lifted, over, 30), leg(over, release_p, 15)]
    )
    T = np.stack([top_down_pose(p, demo_yaw) for p in pts])

    # conventions.md section 7.2, exactly as `common/segment.py` writes it:
    # "grip is 1 (open) outside [grasp, release)". So `grip[grasp_idx]` is
    # already closed and `grip[release_idx]` is already open, which is what puts
    # one close edge at the end of the approach and one open edge at the end of
    # the transport.
    grasp_idx, release_idx = 19, len(T) - 1
    grip = np.ones(len(T))
    grip[grasp_idx:release_idx] = 0.0
    return SourceDemo(
        T_demo_tcp=T, grip=grip, grasp_idx=grasp_idx, release_idx=release_idx, arm="right"
    )


def matched_demo(**kw) -> SourceDemo:
    """A synthetic demo whose index is exactly what the nominal scene asks for."""
    q = scene_index(nominal_cube_pose())
    return synthetic_demo(q["dist_xy"], q["dz"], **kw)


# --- the index the screen measures -----------------------------------------


def test_matched_demo_has_zero_index_offset():
    """The guard for the fixtures below: `matched_demo` really is matched."""
    demo = matched_demo()
    assert index_offset(demo.index, scene_index(nominal_cube_pose())) == pytest.approx(0, abs=1e-9)


# --- the geometry plan_from_demo must produce -------------------------------


def test_matched_demo_lands_its_grasp_on_the_cube():
    """Section 8 step 4: the approach's last pose is the detected object pose."""
    T_cube = nominal_cube_pose()
    plan = plan_from_demo(matched_demo(), T_cube)
    i0, i1 = plan.segments["approach"]
    np.testing.assert_allclose(plan.pose_path[i1 - 1][:3, 3], T_cube[:3, 3], atol=1e-9)


def test_matched_demo_lands_its_release_over_the_bin():
    """Section 8 step 4: the transport's last pose is the place target."""
    plan = plan_from_demo(matched_demo(), nominal_cube_pose())
    i0, i1 = plan.segments["transport"]
    np.testing.assert_allclose(
        plan.pose_path[i1 - 1][:3, 3], bin_place_target()[:3, 3], atol=1e-9
    )


def test_a_matched_demo_needs_no_mid_bridge_travel():
    """With a matched index the transport starts *at* the cube.

    This is what the `(dist_xy, dz)` index is for, and it is the property that
    makes the section 8 step 5 mid-bridge a formality rather than a second carry.
    A regression here means the two segment transforms disagree.
    """
    T_cube = nominal_cube_pose()
    plan = plan_from_demo(matched_demo(), T_cube)
    i0, i1 = plan.segments["transport"]
    assert np.linalg.norm(plan.pose_path[i0][:3, 3] - T_cube[:3, 3]) < 1e-9


@pytest.mark.parametrize("heading", np.linspace(-np.pi, np.pi, 7))
def test_the_plan_is_independent_of_the_demo_frame_heading(heading):
    """Section 8 step 3 must absorb the demo frame's arbitrary heading.

    EgoDex's `demo` frame has "arbitrary heading" (conventions.md section 2), so
    two recordings of the same motion facing different ways have to produce the
    same robot trajectory. If they do not, the heading rotation is wrong.
    """
    T_cube = nominal_cube_pose()
    ref = plan_from_demo(matched_demo(heading=0.0), T_cube)
    got = plan_from_demo(matched_demo(heading=float(heading)), T_cube)
    np.testing.assert_allclose(got.pose_path[:, :3, 3], ref.pose_path[:, :3, 3], atol=1e-9)


def test_the_plan_stays_above_the_table():
    plan = plan_from_demo(matched_demo(), nominal_cube_pose())
    min_z = float(config.skills()["library"]["min_tcp_z"])
    assert plan.pose_path[:, 2, 3].min() >= min_z


def test_the_plan_is_top_down_throughout():
    """A top-down source demo must stay top-down after retargeting.

    The section 8 rigid transform is a z-rotation plus a translation, so it cannot
    tilt the approach axis. If this fails, something is composing rotations in the
    wrong order.
    """
    plan = plan_from_demo(matched_demo(demo_yaw=0.6), nominal_cube_pose())
    axis = plan.pose_path[:, :3, 2]
    np.testing.assert_allclose(axis, np.broadcast_to([0.0, 0.0, -1.0], axis.shape), atol=1e-9)


def test_the_gripper_closes_once_and_opens_once():
    plan = plan_from_demo(matched_demo(), nominal_cube_pose())
    edges = np.diff(plan.grip_path)
    assert (edges < 0).sum() == 1, "expected exactly one close"
    assert (edges > 0).sum() == 1, "expected exactly one open"


def test_each_gripper_change_is_held_still_for_grip_settle_s():
    """The fingers need `grip_settle_s` of stationary pose to travel."""
    plan = plan_from_demo(matched_demo(), nominal_cube_pose())
    want = max(int(round(float(config.skills()["exec"]["grip_settle_s"]) * RATE)), 1)
    for e in np.flatnonzero(np.diff(plan.grip_path) != 0.0):
        held = plan.pose_path[e + 1 : e + 1 + want, :3, 3]
        np.testing.assert_allclose(
            held, np.repeat(held[:1], len(held), axis=0), atol=1e-12,
            err_msg=f"pose moves during the gripper dwell at sample {e}",
        )


# --- flatten_top_down -------------------------------------------------------


def test_flatten_leaves_an_already_top_down_pose_alone():
    for yaw in np.linspace(-np.pi, np.pi, 9):
        T = top_down_pose([0.3, -0.25, 0.05], yaw)
        np.testing.assert_allclose(flatten_top_down(T), T, atol=1e-12)


def test_flatten_forces_the_approach_axis_down():
    rng = np.random.default_rng(3)
    for _ in range(20):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0:
            R = R @ np.diag([1.0, 1.0, -1.0])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, rng.normal(size=3)
        out = flatten_top_down(T)
        np.testing.assert_allclose(out[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(out[:3, 3], T[:3, 3], atol=1e-12)
        assert np.linalg.det(out[:3, :3]) == pytest.approx(1.0)


def test_flatten_commutes_with_a_z_rotation():
    """Why flattening before the section 8 transform is safe.

    The retarget rigid transform is a z-rotation plus a translation. Flattening
    only has to commute with the rotation part for the order to be irrelevant.
    """
    rng = np.random.default_rng(5)
    T = np.eye(4)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    T[:3, :3] = R if np.linalg.det(R) > 0 else R @ np.diag([1.0, 1.0, -1.0])
    T[:3, 3] = [0.2, -0.3, 0.1]
    Rz = rot_z_T(0.7)
    np.testing.assert_allclose(flatten_top_down(Rz @ T), Rz @ flatten_top_down(T), atol=1e-12)


# --- trim_approach ----------------------------------------------------------


def test_trim_keeps_the_grasp_and_the_whole_transport():
    demo = matched_demo()
    trimmed = trim_approach(demo, 0.03)
    np.testing.assert_allclose(
        trimmed.T_demo_tcp[trimmed.grasp_idx], demo.T_demo_tcp[demo.grasp_idx], atol=1e-12
    )
    np.testing.assert_allclose(
        trimmed.T_demo_tcp[trimmed.release_idx], demo.T_demo_tcp[demo.release_idx], atol=1e-12
    )
    assert trimmed.release_idx - trimmed.grasp_idx == demo.release_idx - demo.grasp_idx


def test_trim_bounds_the_remaining_approach_path_length():
    demo = matched_demo()
    for budget in (0.01, 0.03, 0.06, 0.12):
        trimmed = trim_approach(demo, budget)
        p = trimmed.T_demo_tcp[: trimmed.grasp_idx + 1, :3, 3]
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        # One sample of slack: the trim keeps the first sample that fits.
        assert length <= budget + 0.01


def test_trim_is_a_no_op_when_the_budget_exceeds_the_approach():
    demo = matched_demo()
    trimmed = trim_approach(demo, 10.0)
    assert trimmed.grasp_idx == demo.grasp_idx
    assert len(trimmed.T_demo_tcp) == len(demo.T_demo_tcp)


def test_trim_never_leaves_fewer_than_two_approach_samples():
    demo = matched_demo()
    trimmed = trim_approach(demo, 0.0)
    assert trimmed.grasp_idx >= 1


# --- the place target's yaw is not free (section 8 step 4) ------------------


def test_place_target_yaw_follows_the_aligned_demo_release_yaw():
    """Choosing it freely would rotate the whole transport about the bin.

    The hand-coded planner picks a release yaw by reachability, which is correct
    *there* because its waypoints are built at that yaw. Here the target's yaw
    enters the rigid transform, so it must match what the demo's own release yaw
    became after heading alignment.
    """
    T_cube = nominal_cube_pose()
    for demo_yaw in (0.0, 0.5, -1.1):
        for heading in (0.0, 1.3, -2.0):
            demo = matched_demo(heading=heading, demo_yaw=demo_yaw)
            plan = plan_from_demo(demo, T_cube)
            i0, i1 = plan.segments["transport"]
            # The transport's final pose IS the place target, so its yaw is the
            # one that was used; it must equal the plan's reported release yaw.
            assert float(yaw_of(plan.pose_path[i1 - 1])) == pytest.approx(
                plan.release_yaw, abs=1e-9
            )


def test_transport_is_a_pure_translation_of_the_aligned_demo():
    """With the step 4 yaw rule, the transport keeps the demo's shape exactly.

    Measured as: every transport sample is the same rigid offset from the
    corresponding demo sample, up to one global z-rotation. A spurious rotation
    about the bin shows up here as a changing offset.
    """
    T_cube = nominal_cube_pose()
    demo = matched_demo(heading=0.9, demo_yaw=0.3)
    plan = plan_from_demo(demo, T_cube)
    src = demo.T_demo_tcp[demo.grasp_idx : demo.release_idx + 1]
    # The transport range also contains the release dwell appended after its last
    # sample, so compare only the samples that came from the demo.
    i0, i1 = plan.segments["transport"]
    got = plan.pose_path[i0:i1][: len(src)]
    assert len(got) == len(src)

    rigid = got[0] @ inv_T(src[0])
    np.testing.assert_allclose(rigid @ src, got, atol=1e-9)
