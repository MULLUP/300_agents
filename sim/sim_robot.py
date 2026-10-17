"""`SimRobot`: the MuJoCo implementation of `RobotInterface` (plan task 1.P1.3).

Owns the one `MjModel`/`MjData` pair for `sim/scene.xml` and is the only place in
`sim/` that writes `data.ctrl` (conventions.md section 4). Position actuators are
already in the upstream model, so commanding a joint means writing its actuator
target and stepping physics.

Safety (conventions.md section 10) is enforced here, through `common/safety.py`,
so `SimRobot` and `RealRobot` reject exactly the same trajectories.

    robot = SimRobot()
    robot.home()
    robot.stream(q_traj, rate_hz=50.0)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, safety  # noqa: E402
from common.joints import SIDES, JointIndex  # noqa: E402
from common.robot_interface import RobotInterface  # noqa: E402

SCENE_PATH = config.path("sim", "scene.xml")

# The same scene with the visual geoms stripped: dynamically identical, which
# `sim/build_models.py` asserts on every build, but 25 MB of model buffer
# instead of 322 MB. That is what lets a 16-worker batch step fit in memory --
# the full model costs ~1.65 GB of RSS per process. Anything that RENDERS must
# use `SCENE_PATH`; the headless scene has no visual geometry to photograph.
SCENE_HEADLESS_PATH = config.path("sim", "scene_headless.xml")


class UnsafeTrajectory(RuntimeError):
    """A commanded trajectory broke a `conventions.md` section 10 limit."""

    def __init__(self, violations: list[safety.Violation]):
        self.violations = violations
        super().__init__("; ".join(str(v) for v in violations[:6]))


class SimRobot(RobotInterface):
    """MuJoCo-backed robot. One instance owns one simulation."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        enforce_safety: bool = True,
        record: bool = False,
        render: bool = True,
    ):
        """`render=False` loads the visual-free scene — use it for batch physics.

        An explicit `model_path` wins over `render`.
        """
        model_path = Path(model_path) if model_path is not None else (
            SCENE_PATH if render else SCENE_HEADLESS_PATH
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} is missing — generate it with `uv run python sim/build_models.py`"
            )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.index = JointIndex(self.model)
        self.enforce_safety = enforce_safety

        # Recorded (t, q) per control cycle, for dataset export and debugging.
        self.record = record
        self.log_t: list[float] = []
        self.log_q: list[np.ndarray] = []

        self.reset()

    # --- state --------------------------------------------------------------

    def reset(self, keep_objects: bool = False) -> None:
        """Reset to HOME. Object poses are reset too unless `keep_objects`."""
        if keep_objects:
            qpos = self.data.qpos.copy()
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = qpos
        else:
            mujoco.mj_resetData(self.model, self.data)
        q = config.home_q()
        self.index.write_qpos(q, self.data.qpos)
        self.index.write_ctrl(q, self.data.ctrl)
        mujoco.mj_forward(self.model, self.data)
        self.log_t.clear()
        self.log_q.clear()

    def get_state(self) -> dict[str, Any]:
        return {"t": float(self.data.time), "q": self.index.read_qpos(self.data.qpos)}

    def tcp_pose(self, side: str) -> np.ndarray:
        """Current `T_table_tcp` (4,4) for one side, from the live simulation."""
        sid = self.index.site_id(self.validate_side(side))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[sid]
        return T

    # --- motion -------------------------------------------------------------

    def move_to(self, q_target: np.ndarray, duration_s: float) -> None:
        """Cosine-eased joint interpolation from the current pose to `q_target`.

        The requested duration is a *floor*: a joint-space move makes the tcp
        follow a curved path whose length is not known in advance, so the
        duration is stretched if needed to keep the tcp under the
        `conventions.md` section 10 speed limit. Picking durations by hand does
        not work here — the same 0.31 m move needed 2.5 s by straight-line
        distance but overshot the limit by 20% in practice.
        """
        q_target = self.validate_q(q_target)
        rate_hz = float(config.skills()["exec"]["rate_hz"])
        q0 = self.get_state()["q"]
        duration_s = max(float(duration_s), self._safe_duration(q0, q_target, rate_hz))

        n = max(int(round(duration_s * rate_hz)), 2)
        # Cosine easing starts and ends at zero velocity, which keeps the first
        # and last joint steps inside the per-cycle limit for any duration that
        # is itself feasible.
        u = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
        traj = q0[None, :] + u[:, None] * (q_target - q0)[None, :]
        self.stream(traj, rate_hz)

    def _safe_duration(
        self, q0: np.ndarray, q_target: np.ndarray, rate_hz: float, margin: float = 0.85
    ) -> float:
        """Shortest duration keeping a cosine-eased move inside the section 10 limits.

        Along the path parameterized by `u` in [0, 1], the tcp speed is
        ``|dp/du| * u'(t)``. Cosine easing has ``max u' = pi / (2 T)``, so

            T >= max_u |dp/du| * pi / (2 * v_max * margin)

        The same reasoning bounds the per-cycle joint step. Both arms are checked,
        since a two-arm move can violate the limit on either.
        """
        lim = safety.limits()
        q0 = np.asarray(q0, dtype=np.float64)
        dq = np.asarray(q_target, dtype=np.float64) - q0

        probe = np.linspace(0.0, 1.0, 64)
        traj = q0[None, :] + probe[:, None] * dq[None, :]

        t_speed = 0.0
        for side in SIDES:
            pos = self.tcp_path(traj, side)
            # |dp/du| from finite differences on the uniform u grid.
            dp_du = np.linalg.norm(np.diff(pos, axis=0), axis=1) / (probe[1] - probe[0])
            peak = float(dp_du.max()) if len(dp_du) else 0.0
            t_speed = max(t_speed, peak * np.pi / (2.0 * lim.max_tcp_speed * margin))

        arm_slots = [s for side in SIDES for s in self.index.arm_slots[side]]
        peak_dq = float(np.abs(dq[arm_slots]).max())
        step_limit = lim.joint_step_limit(rate_hz) * margin
        t_joint = peak_dq * np.pi / (2.0 * step_limit * rate_hz)

        return max(t_speed, t_joint, 2.0 / rate_hz)

    def stream(self, q_traj: np.ndarray, rate_hz: float) -> None:
        """Execute an `(N, 16)` trajectory, holding each sample for one cycle."""
        traj = np.atleast_2d(np.asarray(q_traj, dtype=np.float64))
        if self.enforce_safety:
            self._check(traj, rate_hz)

        steps = max(int(round((1.0 / float(rate_hz)) / self.model.opt.timestep)), 1)
        for q in traj:
            self.index.write_ctrl(q, self.data.ctrl)
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            if self.record:
                self.log_t.append(float(self.data.time))
                self.log_q.append(self.index.read_qpos(self.data.qpos))

    def set_gripper(self, side: str, g: float) -> None:
        """Command one gripper and hold the arms still while it moves."""
        side = self.validate_side(side)
        g = self.validate_gripper(g)
        q = self.get_state()["q"]
        q[self.index.gripper_slot[side]] = g
        # A gripper command is not a joint interpolation: hold the target for a
        # fixed settle so the fingers reach it and contacts stabilize.
        settle_s = float(config.skills()["exec"].get("grip_settle_s", 0.5))
        rate_hz = float(config.skills()["exec"]["rate_hz"])
        n = max(int(round(settle_s * rate_hz)), 1)
        self.stream(np.repeat(q[None, :], n, axis=0), rate_hz)

    def settle(self, seconds: float) -> None:
        """Step physics with the current command held, e.g. before judging success."""
        for _ in range(max(int(round(seconds / self.model.opt.timestep)), 1)):
            mujoco.mj_step(self.model, self.data)

    # --- safety -------------------------------------------------------------

    def _check(self, traj: np.ndarray, rate_hz: float) -> None:
        violations = safety.check_shape(traj)
        if violations:  # shape errors make the rest meaningless
            raise UnsafeTrajectory(violations)
        violations += safety.check_joint_limits(traj, self.index)
        violations += safety.check_joint_steps(traj, rate_hz, self.index)
        violations += safety.check_tcp_path(self.tcp_path(traj, "right"), rate_hz, "tcp_right")
        if violations:
            raise UnsafeTrajectory(violations)

    def tcp_path(self, traj: np.ndarray, side: str) -> np.ndarray:
        """`(N, 3)` tcp positions for a trajectory, without disturbing the live state.

        Uses a scratch `MjData` so a safety check never advances the simulation.
        """
        scratch = getattr(self, "_scratch", None)
        if scratch is None:
            scratch = self._scratch = mujoco.MjData(self.model)
        sid = self.index.site_id(side)
        out = np.empty((len(traj), 3), dtype=np.float64)
        for k, q in enumerate(traj):
            self.index.write_qpos(q, scratch.qpos)
            mujoco.mj_kinematics(self.model, scratch)
            out[k] = scratch.site_xpos[sid]
        return out

    # --- scene (sim only; `RealRobot` has no equivalent) --------------------

    def body_pose(self, name: str) -> np.ndarray:
        """`T_table_<body>` (4,4) for any body, e.g. the cube."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in {self.model.nbody} bodies")
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.data.xpos[bid]
        return T

    def free_body_speed(self, joint_name: str) -> float:
        """Linear speed of a free-jointed body, m/s.

        Reads the joint's own dof block. Indexing `qvel` from 0 instead would
        read the left arm's first three joints, which is what an earlier version
        of the success check did.
        """
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise KeyError(f"free joint {joint_name!r} not in the model")
        dadr = int(self.model.jnt_dofadr[jid])
        return float(np.linalg.norm(self.data.qvel[dadr : dadr + 3]))

    def set_free_body_pose(self, joint_name: str, pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Place a free-jointed body (the cube) and zero its velocity."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise KeyError(f"free joint {joint_name!r} not in the model")
        adr = int(self.model.jnt_qposadr[jid])
        self.data.qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float64)
        self.data.qpos[adr + 3 : adr + 7] = np.asarray(quat_wxyz, dtype=np.float64)
        dadr = int(self.model.jnt_dofadr[jid])
        self.data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
