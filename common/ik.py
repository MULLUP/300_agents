"""Inverse kinematics for one OpenArm arm (plan task 1.P1.3).

Solves for a `tcp_left` / `tcp_right` pose given in the `table` frame, on
`sim/robot_ik.xml` — the robot-only model, so no free object can perturb the
solution (`conventions.md` section 4).

Differential IK via `mink`: each iteration builds a QP from a frame task (the
target pose), a posture task (stay near the seed, weakly) and the model's
configuration limits, then integrates the resulting joint velocity. Tolerances
come from `config/skills.yaml` `ik` and match the `conventions.md` section 10
rejection limits (5 mm, 3 deg).

    solver = IKSolver()
    res = solver.solve(T_table_tcp, side="right", q_seed=config.home_q())
    if res.converged: ...
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mink
import mujoco
import numpy as np

from common import config
from common.joints import JointIndex
from common.transforms import rot_angle

IK_MODEL_PATH = config.path("sim", "robot_ik.xml")


@dataclass(frozen=True)
class IKResult:
    """Outcome of one IK solve.

    `q` is always a full robot-interface 16-vector: the target arm's joints are
    the solution, every other value is passed through from the seed unchanged.
    `q` is returned even when `converged` is False, so callers can inspect the
    best effort — but `conventions.md` section 10 says a trajectory with
    `pos_err > 5 mm` or `rot_err_deg > 3` must be rejected.
    """

    q: np.ndarray
    converged: bool
    pos_err: float
    rot_err_deg: float
    iters: int

    def __bool__(self) -> bool:
        return self.converged


def seed_bank_path(side: str) -> "Path":
    return config.path("sim", f"ik_seeds_{side}.npz")


class SeedBank:
    """Near-top-down configurations indexed by their tcp position.

    Differential IK is local, so a cold query needs a seed that is already close
    in *task* space. A fixed handful of seeds cannot cover the table: the measured
    recovery rate with 12 spread seeds was 30%. This bank instead stores a few
    thousand harvested configurations with their tcp positions, and
    :meth:`nearest` returns the `k` whose tcp is closest to the requested one, so
    the solver always starts a short distance from its target.

    Built by `sim/reachability.py`; stored at `sim/ik_seeds_<side>.npz`.
    """

    def __init__(self, arm_q: np.ndarray, tcp_pos: np.ndarray, side: str):
        self.arm_q = np.asarray(arm_q, dtype=np.float64)
        self.tcp_pos = np.asarray(tcp_pos, dtype=np.float64)
        self.side = side
        if len(self.arm_q) != len(self.tcp_pos):
            raise ValueError("arm_q and tcp_pos must have the same length")

    def __len__(self) -> int:
        return len(self.arm_q)

    @classmethod
    def load(cls, side: str) -> "SeedBank":
        path = seed_bank_path(side)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing — build it with `uv run python sim/reachability.py`"
            )
        with np.load(path) as npz:
            return cls(npz["arm_q"], npz["tcp_pos"], side=str(npz["side"]))

    def save(self) -> "Path":
        path = seed_bank_path(self.side)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            arm_q=self.arm_q,
            tcp_pos=self.tcp_pos,
            side=self.side,
            conventions_version=config.CONVENTIONS_VERSION,
        )
        return path

    def nearest(self, pos: np.ndarray, k: int, index) -> list[np.ndarray]:
        """The `k` seeds whose tcp is closest to `pos`, as full 16-vectors."""
        d = np.linalg.norm(self.tcp_pos - np.asarray(pos, dtype=np.float64)[None, :], axis=1)
        slots = index.arm_slots[self.side]
        home = config.home_q()
        out = []
        for i in np.argsort(d)[: max(int(k), 1)]:
            q = home.copy()
            q[slots] = self.arm_q[i]
            out.append(q)
        return out


@lru_cache(maxsize=1)
def _load_ik_model() -> mujoco.MjModel:
    if not IK_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{IK_MODEL_PATH} is missing — generate it with `uv run python sim/build_models.py`"
        )
    return mujoco.MjModel.from_xml_path(str(IK_MODEL_PATH))


class IKSolver:
    """Reusable IK on the robot-only model. One instance, many solves."""

    def __init__(self, cfg: dict | None = None):
        self.model = _load_ik_model()
        self.index = JointIndex(self.model)
        self.cfg = cfg if cfg is not None else config.skills()["ik"]

        self.pos_tol = float(self.cfg["pos_tol"])
        self.rot_tol_deg = float(self.cfg["rot_tol_deg"])
        self.max_iters = int(self.cfg["max_iters"])
        self.dt = float(self.cfg["dt"])

        self._configuration = mink.Configuration(self.model)
        self._data = mujoco.MjData(self.model)
        # One frame task per side, reused across solves.
        self._frame_task = {
            side: mink.FrameTask(
                frame_name=self.index.tcp_sites[side],
                frame_type="site",
                position_cost=float(self.cfg["position_cost"]),
                orientation_cost=float(self.cfg["orientation_cost"]),
                lm_damping=float(self.cfg.get("lm_damping", 1e-4)),
            )
            for side in self.index.tcp_sites
        }
        # Free-yaw variant: position only, plus an axis-alignment task that pins
        # the approach direction and leaves the spin about it unconstrained.
        #
        # A top-down cube grasp does not constrain the wrist's rotation about its
        # own approach axis except at the moment the jaws close, yet a FrameTask
        # constrains all three rotational dof. Measured consequence: of 548
        # trajectory samples that failed IK, 0 were recoverable at the commanded
        # orientation but **120/120** sampled were reachable once the yaw was
        # free, needing a median of 8 deg and at most 44 deg of yaw change (which
        # is no accident — 45 deg is the cube's yaw symmetry half-period).
        #
        # `AxisAlignTask` is rank two by construction, so this is a genuine
        # relaxation of the constraint rather than a reweighting of it.
        self._pos_task = {
            side: mink.FrameTask(
                frame_name=self.index.tcp_sites[side],
                frame_type="site",
                position_cost=float(self.cfg["position_cost"]),
                orientation_cost=0.0,
                lm_damping=float(self.cfg.get("lm_damping", 1e-4)),
            )
            for side in self.index.tcp_sites
        }
        self._axis_task = {
            side: mink.AxisAlignTask(
                frame_name=self.index.tcp_sites[side],
                frame_type="site",
                axis=(0.0, 0.0, 1.0),  # tcp +z is the approach axis (section 2)
                cost=float(self.cfg["orientation_cost"]),
                lm_damping=float(self.cfg.get("lm_damping", 1e-4)),
            )
            for side in self.index.tcp_sites
        }
        self._posture_task = mink.PostureTask(self.model, cost=float(self.cfg["posture_weight"]))

        # A second, stronger posture task used only for free-yaw solves.
        #
        # Freeing the yaw leaves a 2-dof null space (7 joints - 3 position - 2
        # axis), and *something* has to choose within it. At the normal
        # `posture_weight` of 0.01 the choice is essentially arbitrary, so
        # adjacent trajectory samples get resolved into different IK branches:
        # measured, freeing the yaw cut `ik_diverged` from 29 to 13 demos but
        # pushed `branch_flip` from 6 to 21. Since `solve_path` sets the posture
        # target to the *previous solution*, a stronger posture cost is exactly a
        # continuity preference, and it costs nothing in accuracy because it acts
        # only inside the null space rather than against the position task.
        self._free_posture_task = mink.PostureTask(
            self.model, cost=float(self.cfg.get("free_yaw_posture_weight", 0.5))
        )
        self._limits = [mink.ConfigurationLimit(self.model)]

    # --- forward kinematics -------------------------------------------------

    def fk(self, q: np.ndarray, side: str) -> np.ndarray:
        """16-vector -> `T_table_tcp` (4,4) for one side."""
        self._data.qpos[:] = 0.0
        self.index.write_qpos(q, self._data.qpos)
        mujoco.mj_kinematics(self.model, self._data)
        sid = self.index.site_id(side)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._data.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self._data.site_xpos[sid]
        return T

    # --- inverse kinematics -------------------------------------------------

    def solve(
        self, T_table_tcp: np.ndarray, side: str, q_seed: np.ndarray, free_yaw: bool = False
    ) -> IKResult:
        """Solve one arm for `T_table_tcp`, seeded at `q_seed` (a 16-vector).

        With `free_yaw`, only the position and the **approach axis** of the target
        are constrained; rotation about that axis is left to the solver. The
        target's own yaw is then merely a hint, and `rot_err_deg` reports the
        approach-axis misalignment rather than the full rotation error — which is
        the right error to report, because the yaw was never commanded.

        Use it for every part of a trajectory except where the jaws have to
        straddle an object's faces. See the comment on `_axis_task`.
        """
        if side not in self._frame_task:
            raise ValueError(f"side must be one of {sorted(self._frame_task)}, got {side!r}")

        q_seed = np.asarray(q_seed, dtype=np.float64).reshape(-1)
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        self.index.write_qpos(q_seed, qpos)

        self._configuration.update(qpos)
        self._posture_task.set_target(qpos)
        self._free_posture_task.set_target(qpos)

        T_target = np.asarray(T_table_tcp, dtype=np.float64)
        target = mink.SE3.from_matrix(T_target)
        if free_yaw:
            pos_task = self._pos_task[side]
            pos_task.set_target(target)
            axis_task = self._axis_task[side]
            axis_task.set_target(T_target[:3, 2])
            tasks = [pos_task, axis_task, self._free_posture_task]
        else:
            task = self._frame_task[side]
            task.set_target(target)
            tasks = [task, self._posture_task]

        # Iterate to a *tighter* target than the acceptance threshold. Stopping
        # the instant the error qualifies leaves solutions sitting right on the
        # conventions.md section 10 limit (measured: 2.8-4.8 mm against a 5 mm
        # tolerance), so a small downstream perturbation would fail the check.
        polish = float(self.cfg.get("polish_factor", 0.3))
        iters = 0
        for iters in range(1, self.max_iters + 1):
            vel = mink.solve_ik(
                self._configuration, tasks, self.dt, solver="daqp", damping=1e-12, limits=self._limits
            )
            self._configuration.integrate_inplace(vel, self.dt)
            if self._within_tol(side, T_target, polish, free_yaw):
                break

        # Only the target arm is allowed to move: the other arm and both grippers
        # are branches the frame task cannot reach, but we pass the seed values
        # straight through rather than trusting the posture task to hold them.
        q_out = q_seed.copy()
        q_solved = self.index.read_qpos(self._configuration.q)
        for slot in self.index.arm_slots[side]:
            q_out[slot] = q_solved[slot]

        pos_err, rot_err_deg = self.pose_error(q_out, side, T_target, free_yaw)
        return IKResult(
            q=q_out,
            converged=bool(pos_err <= self.pos_tol and rot_err_deg <= self.rot_tol_deg),
            pos_err=pos_err,
            rot_err_deg=rot_err_deg,
            iters=iters,
        )

    def solve_multi(
        self,
        T_table_tcp: np.ndarray,
        side: str,
        seeds: Sequence[np.ndarray],
        n_restarts: int = 0,
        rng: np.random.Generator | None = None,
        prefer_q: np.ndarray | None = None,
        free_yaw: bool = False,
    ) -> IKResult:
        """Try several seeds, return the first converged result, else the best.

        With `prefer_q`, *all* seeds are tried and the converged solution closest
        to `prefer_q` in joint space is returned instead of the first one. This
        is what keeps a trajectory continuous: a 7-dof arm has many solutions for
        the same pose, and taking an arbitrary one mid-path produced a measured
        2.4 rad jump in `joint_1` in a single 50 Hz cycle — a branch flip that
        violates the `conventions.md` section 10 step limit by 48x.

        Differential IK is strictly local and this arm needs it: a top-down grasp
        lives near ``joint1 = -1.2, joint2 = +2.9`` while HOME is at all-zeros, so
        seeding from HOME lands in a joint-limit corner tens of millimetres from
        the target. `conventions.md` section 10 forbids using such a solution, so
        callers must pass a seed bank — see `config/skills.yaml` `ik.seeds`.

        `n_restarts` adds uniform perturbations of the best seed, which recovers
        poses that no fixed seed happens to be near.
        """
        if not seeds:
            raise ValueError("seeds must be non-empty; see config/skills.yaml ik.seeds")

        slots = self.index.arm_slots[side]

        def closeness(res: IKResult) -> float:
            return float(np.linalg.norm(res.q[slots] - np.asarray(prefer_q)[slots]))

        best: IKResult | None = None
        hits: list[IKResult] = []
        for seed in seeds:
            res = self.solve(T_table_tcp, side, seed, free_yaw)
            if res.converged:
                if prefer_q is None:
                    return res
                hits.append(res)
            if best is None or res.pos_err < best.pos_err:
                best = res

        if n_restarts:
            rng = rng if rng is not None else np.random.default_rng(0)
            lo, hi = self.index.arm_limits(side)
            for _ in range(n_restarts):
                if prefer_q is None or not hits:
                    seed = np.asarray(best.q, dtype=np.float64).copy()
                else:
                    seed = min(hits, key=closeness).q.copy()
                scale = 0.25 * (hi - lo)
                seed[slots] = np.clip(seed[slots] + rng.normal(0.0, scale), lo, hi)
                res = self.solve(T_table_tcp, side, seed, free_yaw)
                if res.converged:
                    if prefer_q is None:
                        return res
                    hits.append(res)
                if res.pos_err < best.pos_err:
                    best = res

        if hits:
            return min(hits, key=closeness)
        return best

    def solve_cold(
        self,
        T_table_tcp: np.ndarray,
        side: str,
        k: int | None = None,
        n_restarts: int | None = None,
        extra_seeds: Sequence[np.ndarray] = (),
        rng: np.random.Generator | None = None,
        prefer_q: np.ndarray | None = None,
        free_yaw: bool = False,
    ) -> IKResult:
        """Solve with no useful prior, using the on-disk seed bank.

        For an isolated target (a reachability probe, the first waypoint of a
        skill). Along a trajectory prefer :meth:`solve_path`, which warm-starts
        each waypoint from the previous solution and is both faster and more
        reliable.
        """
        k = int(self.cfg.get("seed_k", 12)) if k is None else k
        n_restarts = int(self.cfg.get("n_restarts", 12)) if n_restarts is None else n_restarts
        bank = self._bank(side)
        seeds = list(extra_seeds) + bank.nearest(np.asarray(T_table_tcp)[:3, 3], k, self.index)
        return self.solve_multi(
            T_table_tcp, side, seeds, n_restarts=n_restarts, rng=rng, prefer_q=prefer_q,
            free_yaw=free_yaw,
        )

    def solve_path(
        self,
        T_path: np.ndarray,
        side: str,
        q_seed: np.ndarray | None = None,
        cold_k: int | None = None,
        free_yaw: bool | np.ndarray = False,
    ) -> tuple[np.ndarray, list[IKResult]]:
        """Solve an `(N, 4, 4)` pose path, warm-starting each step from the last.

        Consecutive waypoints are close, so the previous solution is the best
        available seed — this is both the fastest and the most reliable mode, and
        it keeps the joint path continuous (no mid-trajectory elbow flips, which
        would break the `conventions.md` section 10 per-cycle step limit).

        Returns `(q_traj (N, 16), per-waypoint results)`. Non-converged waypoints
        are included; the caller decides, since section 10 says reject.
        """
        T_path = np.asarray(T_path, dtype=np.float64).reshape(-1, 4, 4)
        # `free_yaw` may be per-sample: the jaws are only constrained where they
        # close on the object, so yaw is locked there and free everywhere else.
        mask = np.broadcast_to(np.asarray(free_yaw, dtype=bool), (len(T_path),))
        results: list[IKResult] = []
        q_traj = np.empty((len(T_path), 16), dtype=np.float64)

        current = None if q_seed is None else np.asarray(q_seed, dtype=np.float64)
        for k, T in enumerate(T_path):
            fy = bool(mask[k])
            if current is None:
                res = self.solve_cold(T, side, k=cold_k, free_yaw=fy)
            else:
                res = self.solve(T, side, current, free_yaw=fy)
                if not res.converged:
                    # Warm start stalled: fall back to the bank, but ask for the
                    # solution nearest the previous one so the joint path does
                    # not jump to a different branch.
                    res = self.solve_cold(
                        T, side, k=cold_k, extra_seeds=[current], prefer_q=current,
                        free_yaw=fy,
                    )
            results.append(res)
            q_traj[k] = res.q
            current = res.q
        return q_traj, results

    def _bank(self, side: str) -> SeedBank:
        if not hasattr(self, "_banks"):
            self._banks: dict[str, SeedBank] = {}
        if side not in self._banks:
            self._banks[side] = SeedBank.load(side)
        return self._banks[side]

    @staticmethod
    def _axis_error_deg(R: np.ndarray, R_target: np.ndarray) -> float:
        """Angle between two frames' +z axes, in degrees — the free-yaw error."""
        c = float(np.clip(np.dot(R[:3, 2], np.asarray(R_target)[:3, 2]), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def pose_error(
        self, q: np.ndarray, side: str, T_target: np.ndarray, free_yaw: bool = False
    ) -> tuple[float, float]:
        """(position error in m, orientation error in degrees) of `q` against `T_target`.

        With `free_yaw` the orientation error is the approach-axis misalignment
        only, since the rotation about that axis was deliberately not commanded.
        """
        T = self.fk(q, side)
        pos_err = float(np.linalg.norm(T[:3, 3] - np.asarray(T_target)[:3, 3]))
        if free_yaw:
            return pos_err, self._axis_error_deg(T[:3, :3], np.asarray(T_target)[:3, :3])
        rot_err = float(np.degrees(rot_angle(T[:3, :3], np.asarray(T_target)[:3, :3])))
        return pos_err, rot_err

    def _within_tol(
        self, side: str, T_target: np.ndarray, scale: float = 1.0, free_yaw: bool = False
    ) -> bool:
        """Tolerance check on the solver's own configuration, without a round trip."""
        T = self._configuration.get_transform_frame_to_world(
            self.index.tcp_sites[side], "site"
        ).as_matrix()
        pos_err = float(np.linalg.norm(T[:3, 3] - np.asarray(T_target)[:3, 3]))
        if pos_err > self.pos_tol * scale:
            return False
        if free_yaw:
            err = self._axis_error_deg(T[:3, :3], np.asarray(T_target)[:3, :3])
        else:
            err = float(np.degrees(rot_angle(T[:3, :3], np.asarray(T_target)[:3, :3])))
        return err <= self.rot_tol_deg * scale
