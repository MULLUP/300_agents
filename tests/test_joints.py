"""Tests for `common/joints.py` — the 16-vector <-> MuJoCo mapping (plan task 1.P1.1).

These guard the rule in `conventions.md` section 4 that joint order is never
inferred: if `config/joint_names.yaml` and the model ever disagree, these fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.joints import N_Q, SIDES, JointIndex  # noqa: E402

IK_MODEL = config.path("sim", "robot_ik.xml")
SCENE_MODEL = config.path("sim", "scene.xml")

pytestmark = pytest.mark.skipif(
    not IK_MODEL.exists(), reason="run `uv run python sim/build_models.py` first"
)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(IK_MODEL))


@pytest.fixture(scope="module")
def index(model):
    return JointIndex(model)


def test_order_is_16_and_matches_conventions(index):
    """conventions.md section 4 fixes both the length and the exact names."""
    assert len(index.order) == N_Q
    expected = [f"left_joint_{i}" for i in range(1, 8)] + ["left_gripper"]
    expected += [f"right_joint_{i}" for i in range(1, 8)] + ["right_gripper"]
    assert index.order == expected


def test_every_slot_resolves(index):
    """Every canonical name resolves to a real joint and actuator."""
    assert len(index.qadr) == N_Q
    assert len(index.actuator_id) == N_Q
    assert all(a >= 0 for a in index.actuator_id)
    assert sum(index.is_gripper) == 2


def test_arm_slots_and_dofs(index):
    for side in SIDES:
        assert len(index.arm_slots[side]) == 7
        assert len(index.arm_dofs[side]) == 7
    # The two arms must not share a slot or a dof.
    assert not set(index.arm_slots["left"]) & set(index.arm_slots["right"])
    assert not set(index.arm_dofs["left"]) & set(index.arm_dofs["right"])


def test_qpos_round_trip(index, model):
    """`write_qpos` then `read_qpos` is the identity on a 16-vector."""
    rng = np.random.default_rng(0)
    qpos = np.zeros(model.nq)
    for _ in range(20):
        q = np.empty(N_Q)
        for side in SIDES:
            lo, hi = index.arm_limits(side)
            q[index.arm_slots[side]] = rng.uniform(lo, hi)
            q[index.gripper_slot[side]] = rng.uniform(0.0, 1.0)
        index.write_qpos(q, qpos)
        np.testing.assert_allclose(index.read_qpos(qpos), q, atol=1e-12)


def test_gripper_is_normalized(index, model):
    """A gripper value of 1.0 means `gripper_open_m`, 0.0 means closed."""
    qpos = np.zeros(model.nq)
    q = config.home_q()
    q[index.gripper_slot["right"]] = 1.0
    index.write_qpos(q, qpos)
    driven, mimic = index.qadr[index.gripper_slot["right"]]
    assert qpos[driven] == pytest.approx(index.gripper_open_m)
    # Both coupled finger joints are written, so a reset state is consistent.
    assert qpos[mimic] == pytest.approx(index.gripper_open_m)

    q[index.gripper_slot["right"]] = 0.0
    index.write_qpos(q, qpos)
    assert qpos[driven] == pytest.approx(0.0)


def test_home_is_within_joint_limits(index):
    """HOME must be a legal configuration or every skill starts illegal."""
    q = config.home_q()
    for side in SIDES:
        lo, hi = index.arm_limits(side)
        assert np.all(q[index.arm_slots[side]] >= lo)
        assert np.all(q[index.arm_slots[side]] <= hi)
        assert 0.0 <= q[index.gripper_slot[side]] <= 1.0


def test_write_ctrl_respects_actuator_ranges(index, model):
    """Targets are clipped into each actuator's ctrlrange."""
    ctrl = np.zeros(model.nu)
    q = config.home_q()
    index.write_ctrl(q, ctrl)
    for slot, aid in enumerate(index.actuator_id):
        lo, hi = model.actuator_ctrlrange[aid]
        assert lo - 1e-9 <= ctrl[aid] <= hi + 1e-9


@pytest.mark.skipif(not SCENE_MODEL.exists(), reason="scene.xml not generated")
def test_write_qpos_leaves_the_free_cube_alone():
    """The cube's 7 free-joint values must survive a robot reset (scene.xml)."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_MODEL))
    index = JointIndex(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    assert jid >= 0
    adr = int(model.jnt_qposadr[jid])

    qpos = np.zeros(model.nq)
    sentinel = np.array([0.31, -0.22, 0.0225, 1.0, 0.0, 0.0, 0.0])
    qpos[adr : adr + 7] = sentinel
    index.write_qpos(config.home_q(), qpos)
    np.testing.assert_allclose(qpos[adr : adr + 7], sentinel)
