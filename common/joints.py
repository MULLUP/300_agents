"""The robot-interface 16-vector <-> MuJoCo `qpos`/`ctrl` mapping.

`conventions.md` section 4 forbids inferring joint order from a model or a
dataset, so every index used anywhere comes from `config/joint_names.yaml`
through this module. Build one :class:`JointIndex` per MuJoCo model and reuse it.

The 16-vector is ``[left_joint_1..7, left_gripper, right_joint_1..7, right_gripper]``
with arm joints in radians and grippers normalized (0 = closed, 1 = open).
Each gripper is one interface value but two coupled MJCF joints and one actuator.
"""

from __future__ import annotations

import mujoco
import numpy as np

from common import config

N_Q = 16  # the robot-interface vector length
SIDES = ("left", "right")


class JointIndex:
    """Cached name -> index lookups for one compiled MuJoCo model."""

    def __init__(self, model: mujoco.MjModel):
        names = config.joint_names()
        self.model = model
        self.order: list[str] = list(names["order"])
        if len(self.order) != N_Q:
            raise ValueError(f"joint_names.yaml `order` has {len(self.order)} entries, expected {N_Q}")

        self.gripper_open_m = float(names["gripper_open_m"])
        self.tcp_sites: dict[str, str] = dict(names["tcp_sites"])

        # Per interface slot: the qpos addresses it drives and its actuator id.
        # Arm slots drive one joint; gripper slots drive the driven finger joint
        # plus its mimic (the model's equality constraint keeps them equal, but
        # writing both keeps a reset state consistent before the solver runs).
        self.qadr: list[list[int]] = []
        self.actuator_id: list[int] = []
        self.is_gripper: list[bool] = []

        for canon in self.order:
            if canon in names["joints"]:
                self.qadr.append([self._qadr(names["joints"][canon])])
                self.is_gripper.append(False)
            else:
                self.qadr.append(
                    [
                        self._qadr(names["gripper_joints"][canon]),
                        self._qadr(names["gripper_mimic_joints"][canon]),
                    ]
                )
                self.is_gripper.append(True)
            self.actuator_id.append(self._actuator(names["actuators"][canon]))

        # Interface slots and velocity-space (dof) indices per arm, for IK.
        self.arm_slots: dict[str, list[int]] = {
            side: [i for i, c in enumerate(self.order) if c.startswith(f"{side}_joint_")] for side in SIDES
        }
        self.gripper_slot: dict[str, int] = {
            side: self.order.index(f"{side}_gripper") for side in SIDES
        }
        self.arm_dofs: dict[str, list[int]] = {
            side: [self._dofadr(names["joints"][self.order[i]]) for i in self.arm_slots[side]]
            for side in SIDES
        }

    # --- name lookups -------------------------------------------------------

    def _qadr(self, mjcf_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
        if jid < 0:
            raise KeyError(f"joint {mjcf_name!r} not in the model")
        return int(self.model.jnt_qposadr[jid])

    def _dofadr(self, mjcf_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
        if jid < 0:
            raise KeyError(f"joint {mjcf_name!r} not in the model")
        return int(self.model.jnt_dofadr[jid])

    def _actuator(self, mjcf_name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mjcf_name)
        if aid < 0:
            raise KeyError(f"actuator {mjcf_name!r} not in the model")
        return int(aid)

    def site_id(self, side: str) -> int:
        name = self.tcp_sites[side]
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"site {name!r} not in the model (regenerate with sim/build_models.py)")
        return int(sid)

    # --- conversions --------------------------------------------------------

    def write_qpos(self, q: np.ndarray, qpos: np.ndarray) -> None:
        """Write a 16-vector into `qpos` in place, leaving other dofs untouched.

        Other dofs matter: `sim/scene.xml` has a free cube whose 7 qpos values
        must survive a robot reset.
        """
        q = np.asarray(q, dtype=np.float64).reshape(N_Q)
        for slot, adrs in enumerate(self.qadr):
            value = self.gripper_open_m * q[slot] if self.is_gripper[slot] else q[slot]
            for adr in adrs:
                qpos[adr] = value

    def read_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """`qpos` -> the 16-vector. Gripper values are de-normalized back to [0, 1]."""
        q = np.empty(N_Q, dtype=np.float64)
        for slot, adrs in enumerate(self.qadr):
            value = float(qpos[adrs[0]])
            q[slot] = value / self.gripper_open_m if self.is_gripper[slot] else value
        return q

    def write_ctrl(self, q: np.ndarray, ctrl: np.ndarray) -> None:
        """Write a 16-vector as position-actuator targets, in place.

        Actuator control ranges are clipped to, since the finger actuators accept
        a slightly negative target (-0.005) to let the fingers squeeze.
        """
        q = np.asarray(q, dtype=np.float64).reshape(N_Q)
        for slot, aid in enumerate(self.actuator_id):
            value = self.gripper_open_m * q[slot] if self.is_gripper[slot] else q[slot]
            lo, hi = self.model.actuator_ctrlrange[aid]
            ctrl[aid] = float(np.clip(value, lo, hi))

    def arm_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) joint limits in radians for one arm's 7 joints."""
        names = config.joint_names()
        lo, hi = [], []
        for slot in self.arm_slots[side]:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, names["joints"][self.order[slot]]
            )
            lo.append(self.model.jnt_range[jid][0])
            hi.append(self.model.jnt_range[jid][1])
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
