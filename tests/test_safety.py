"""Tests for `common/safety.py` — the `conventions.md` section 10 limits.

These are the checks that stand between a bad trajectory and the hardware, so
each limit gets both a passing and a failing case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, safety  # noqa: E402
from common.joints import N_Q, JointIndex  # noqa: E402

IK_MODEL = config.path("sim", "robot_ik.xml")

pytestmark = pytest.mark.skipif(
    not IK_MODEL.exists(), reason="run `uv run python sim/build_models.py` first"
)


@pytest.fixture(scope="module")
def index():
    return JointIndex(mujoco.MjModel.from_xml_path(str(IK_MODEL)))


@pytest.fixture(scope="module")
def lim():
    return safety.limits()


# --- limits loading ---------------------------------------------------------


def test_limits_match_conventions_section_10(lim):
    """Hard-coded here on purpose: a silent config edit should fail the suite."""
    assert lim.box_x == (0.15, 0.65)
    assert lim.box_y == (-0.45, 0.45)
    assert lim.box_z == (0.01, 0.45)
    assert lim.max_tcp_speed == 0.25
    assert lim.max_joint_step_50hz == 0.05
    assert lim.max_ik_pos_err == 0.005
    assert lim.max_ik_rot_err_deg == 3.0


def test_joint_step_limit_scales_with_rate(lim):
    """"scale accordingly at 30 Hz" — a slower cycle allows a proportionally bigger step."""
    assert lim.joint_step_limit(50.0) == pytest.approx(0.05)
    assert lim.joint_step_limit(30.0) == pytest.approx(0.05 * 50.0 / 30.0)


# --- workspace box ----------------------------------------------------------


def test_in_box_accepts_the_interior_and_rejects_each_face(lim):
    assert bool(lim.in_box(np.array([0.40, 0.0, 0.20])))
    for bad in (
        [0.10, 0.0, 0.20],  # x low
        [0.70, 0.0, 0.20],  # x high
        [0.40, -0.50, 0.20],  # y low
        [0.40, 0.50, 0.20],  # y high
        [0.40, 0.0, 0.005],  # z low (below the table)
        [0.40, 0.0, 0.50],  # z high
    ):
        assert not bool(lim.in_box(np.array(bad))), bad


def test_check_tcp_path_reports_the_offending_sample():
    path = np.array([[0.40, 0.0, 0.20], [0.40, 0.0, 0.20], [0.90, 0.0, 0.20]])
    viol = safety.check_tcp_path(path, 50.0)
    kinds = {v.kind for v in viol}
    assert "workspace_box" in kinds
    assert any(v.index == 2 for v in viol if v.kind == "workspace_box")


def test_check_tcp_path_catches_excess_speed(lim):
    """0.02 m in one 50 Hz cycle is 1.0 m/s, four times the limit."""
    path = np.array([[0.40, 0.0, 0.20], [0.42, 0.0, 0.20]])
    viol = safety.check_tcp_path(path, 50.0)
    assert any(v.kind == "tcp_speed" for v in viol)

    # The same displacement over a 10x longer cycle is within the limit.
    assert not [v for v in safety.check_tcp_path(path, 5.0) if v.kind == "tcp_speed"]


def test_a_still_path_inside_the_box_is_clean():
    path = np.repeat(np.array([[0.40, 0.0, 0.20]]), 10, axis=0)
    assert safety.check_tcp_path(path, 50.0) == []


# --- joint limits and steps -------------------------------------------------


def test_home_passes_every_check(index):
    """HOME must be safe, or `home()` itself would raise."""
    traj = np.repeat(config.home_q()[None, :], 5, axis=0)
    assert safety.check_shape(traj) == []
    assert safety.check_joint_limits(traj, index) == []
    assert safety.check_joint_steps(traj, 50.0, index) == []


def test_joint_limit_violation_is_detected(index):
    q = config.home_q().copy()
    slot = index.arm_slots["right"][0]
    _, hi = index.arm_limits("right")
    q[slot] = hi[0] + 0.5
    viol = safety.check_joint_limits(q[None, :], index)
    assert [v.kind for v in viol] == ["joint_limit"]
    assert "right_joint_1" in viol[0].detail


def test_gripper_out_of_range_is_detected(index):
    q = config.home_q().copy()
    q[index.gripper_slot["right"]] = 1.4
    viol = safety.check_joint_limits(q[None, :], index)
    assert any(v.kind == "gripper_range" for v in viol)


def test_joint_step_violation_is_detected(index):
    a = config.home_q().copy()
    b = a.copy()
    b[index.arm_slots["right"][2]] += 0.2  # 4x the 0.05 rad limit
    viol = safety.check_joint_steps(np.stack([a, b]), 50.0, index)
    assert [v.kind for v in viol] == ["joint_step"]
    assert viol[0].index == 1


def test_gripper_steps_are_exempt(index):
    """Closing the gripper in one cycle is intended, not a violation."""
    a = config.home_q().copy()
    b = a.copy()
    b[index.gripper_slot["right"]] = 0.0  # full 1.0 -> 0.0 jump
    assert safety.check_joint_steps(np.stack([a, b]), 50.0, index) == []


def test_single_sample_has_no_step_violation(index):
    assert safety.check_joint_steps(config.home_q()[None, :], 50.0, index) == []


# --- shape and IK -----------------------------------------------------------


def test_check_shape_rejects_wrong_width_and_nans():
    assert safety.check_shape(np.zeros((4, N_Q))) == []
    assert [v.kind for v in safety.check_shape(np.zeros((4, 7)))] == ["shape"]
    bad = np.zeros((4, N_Q))
    bad[1, 3] = np.nan
    assert [v.kind for v in safety.check_shape(bad)] == ["shape"]


def test_check_ik_error_thresholds(lim):
    assert safety.check_ik_error(0.004, 2.0) == []
    assert [v.kind for v in safety.check_ik_error(0.006, 2.0)] == ["ik_pos_err"]
    assert [v.kind for v in safety.check_ik_error(0.004, 4.0)] == ["ik_rot_err"]
    assert len(safety.check_ik_error(0.006, 4.0)) == 2


def test_violation_str_mentions_the_sample():
    v = safety.Violation("joint_step", "too far", 7)
    assert "joint_step" in str(v) and "7" in str(v)
    assert "sample" not in str(safety.Violation("shape", "bad"))
