"""Trajectory safety checks (conventions.md section 10).

`SimRobot` and `RealRobot` must enforce *the same* limits, so both call into this
module and neither keeps its own copy of a threshold. Every number comes from
`config/safety.yaml`.

A check returns a list of :class:`Violation`; an empty list means the trajectory
is acceptable. Callers decide whether to raise — `sim/execute.py` rejects, while
the data-generation sweep counts rejections as failures instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from common import config
from common.joints import N_Q, SIDES


@dataclass(frozen=True)
class Violation:
    """One broken limit, with enough detail to debug without re-running."""

    kind: str
    detail: str
    index: int = -1  # trajectory sample where it happened, -1 if not per-sample

    def __str__(self) -> str:
        at = f" at sample {self.index}" if self.index >= 0 else ""
        return f"{self.kind}{at}: {self.detail}"


@dataclass(frozen=True)
class SafetyLimits:
    box_x: tuple[float, float]
    box_y: tuple[float, float]
    box_z: tuple[float, float]
    max_tcp_speed: float
    max_joint_step_50hz: float
    max_ik_pos_err: float
    max_ik_rot_err_deg: float
    allow_arm_arm_contact: bool

    def joint_step_limit(self, rate_hz: float) -> float:
        """Per-cycle joint step limit at `rate_hz`, scaled from the 50 Hz figure."""
        return self.max_joint_step_50hz * (50.0 / float(rate_hz))

    def in_box(self, pos: np.ndarray) -> np.ndarray:
        """Elementwise box test for `(..., 3)` positions in the `table` frame."""
        pos = np.asarray(pos, dtype=np.float64)
        return (
            (pos[..., 0] >= self.box_x[0])
            & (pos[..., 0] <= self.box_x[1])
            & (pos[..., 1] >= self.box_y[0])
            & (pos[..., 1] <= self.box_y[1])
            & (pos[..., 2] >= self.box_z[0])
            & (pos[..., 2] <= self.box_z[1])
        )


@lru_cache(maxsize=1)
def limits() -> SafetyLimits:
    cfg = config.load("safety")
    box = cfg["workspace_box"]
    return SafetyLimits(
        box_x=tuple(box["x"]),
        box_y=tuple(box["y"]),
        box_z=tuple(box["z"]),
        max_tcp_speed=float(cfg["max_tcp_speed"]),
        max_joint_step_50hz=float(cfg["max_joint_step_50hz"]),
        max_ik_pos_err=float(cfg["max_ik_pos_err"]),
        max_ik_rot_err_deg=float(cfg["max_ik_rot_err_deg"]),
        allow_arm_arm_contact=bool(cfg["allow_arm_arm_contact"]),
    )


# Joint-limit slack, radians. The IK solver's limit constraint drives solutions
# exactly ONTO a joint stop, where floating-point rounding leaves them a couple
# of ULPs outside it. Without slack a perfectly good trajectory is rejected for
# `right_joint_3 = -1.5708 outside [-1.5708, 1.5708]`. 1e-6 rad is 6e-5 degrees,
# far below anything mechanically meaningful.
JOINT_LIMIT_SLACK = 1e-6


def check_joint_limits(q_traj: np.ndarray, index) -> list[Violation]:
    """Every arm joint within its MJCF range; every gripper value within [0, 1]."""
    out: list[Violation] = []
    q_traj = np.atleast_2d(np.asarray(q_traj, dtype=np.float64))
    for side in SIDES:
        lo, hi = index.arm_limits(side)
        slots = index.arm_slots[side]
        block = q_traj[:, slots]
        bad = (block < lo - JOINT_LIMIT_SLACK) | (block > hi + JOINT_LIMIT_SLACK)
        for k in np.unique(np.nonzero(bad)[0]):
            j = int(np.nonzero(bad[k])[0][0])
            out.append(
                Violation(
                    "joint_limit",
                    f"{index.order[slots[j]]} = {block[k, j]:.4f} outside [{lo[j]:.4f}, {hi[j]:.4f}]",
                    int(k),
                )
            )
        g = q_traj[:, index.gripper_slot[side]]
        bad_g = (g < -1e-9) | (g > 1.0 + 1e-9)
        for k in np.nonzero(bad_g)[0]:
            out.append(
                Violation("gripper_range", f"{side}_gripper = {g[k]:.4f} outside [0, 1]", int(k))
            )
    return out


def check_joint_steps(q_traj: np.ndarray, rate_hz: float, index) -> list[Violation]:
    """Consecutive-sample joint change within the per-cycle limit (arm joints only).

    Grippers are excluded: a gripper command is a normalized open/close, not a
    joint angle, and closing it in one cycle is intended.
    """
    q_traj = np.atleast_2d(np.asarray(q_traj, dtype=np.float64))
    if len(q_traj) < 2:
        return []
    lim = limits().joint_step_limit(rate_hz)
    arm_slots = [s for side in SIDES for s in index.arm_slots[side]]
    step = np.abs(np.diff(q_traj[:, arm_slots], axis=0))
    out: list[Violation] = []
    for k in np.unique(np.nonzero(step > lim)[0]):
        j = int(np.argmax(step[k]))
        out.append(
            Violation(
                "joint_step",
                f"{index.order[arm_slots[j]]} moves {step[k, j]:.4f} rad in one "
                f"{rate_hz:g} Hz cycle, limit {lim:.4f}",
                int(k) + 1,
            )
        )
    return out


def check_tcp_path(
    pos_traj: np.ndarray, rate_hz: float, label: str = "tcp"
) -> list[Violation]:
    """Workspace box and tcp speed along an `(N, 3)` `table`-frame path."""
    lim = limits()
    pos_traj = np.atleast_2d(np.asarray(pos_traj, dtype=np.float64))
    out: list[Violation] = []

    inside = lim.in_box(pos_traj)
    for k in np.nonzero(~inside)[0]:
        out.append(
            Violation(
                "workspace_box",
                f"{label} at {np.round(pos_traj[k], 4).tolist()} is outside "
                f"x{list(lim.box_x)} y{list(lim.box_y)} z{list(lim.box_z)}",
                int(k),
            )
        )

    if len(pos_traj) >= 2:
        speed = np.linalg.norm(np.diff(pos_traj, axis=0), axis=1) * float(rate_hz)
        for k in np.nonzero(speed > lim.max_tcp_speed)[0]:
            out.append(
                Violation(
                    "tcp_speed",
                    f"{label} speed {speed[k]:.3f} m/s exceeds {lim.max_tcp_speed} m/s",
                    int(k) + 1,
                )
            )
    return out


def check_ik_error(pos_err: float, rot_err_deg: float, index_hint: int = -1) -> list[Violation]:
    """One IK solve against the section 10 rejection thresholds."""
    lim = limits()
    out: list[Violation] = []
    if pos_err > lim.max_ik_pos_err:
        out.append(
            Violation("ik_pos_err", f"{pos_err * 1000:.2f} mm exceeds {lim.max_ik_pos_err * 1000:.1f} mm", index_hint)
        )
    if rot_err_deg > lim.max_ik_rot_err_deg:
        out.append(
            Violation("ik_rot_err", f"{rot_err_deg:.2f} deg exceeds {lim.max_ik_rot_err_deg:.1f} deg", index_hint)
        )
    return out


def check_shape(q_traj: np.ndarray) -> list[Violation]:
    """`(N, 16)` and finite."""
    arr = np.asarray(q_traj, dtype=np.float64)
    arr2 = np.atleast_2d(arr)
    if arr2.ndim != 2 or arr2.shape[1] != N_Q:
        return [Violation("shape", f"expected (N, {N_Q}), got {arr.shape}")]
    if not np.all(np.isfinite(arr2)):
        return [Violation("shape", "trajectory contains non-finite values")]
    return []
