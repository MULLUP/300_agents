"""The robot interface both `SimRobot` and `RealRobot` implement (conventions.md section 4).

"Nothing else touches MuJoCo actuators or LeRobot directly." Skills, the
retrieval runtime and the data generator all go through this interface, so a
trajectory that runs in simulation runs on hardware unchanged.

All joint vectors are the 16-value robot-interface vector in the
`config/joint_names.yaml` order: arm joints in radians, grippers normalized to
0.0 = closed, 1.0 = open.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from common import config
from common.joints import N_Q, SIDES

HOME_DURATION_S = 3.0  # conventions.md section 4: `home()` = `move_to(HOME, 3.0)`


class RobotInterface(ABC):
    """Contract from `conventions.md` section 4. Units are section 1's."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """`{"t": seconds, "q": (16,) float64}`."""

    @abstractmethod
    def move_to(self, q_target: np.ndarray, duration_s: float) -> None:
        """Smoothly interpolate from the current state to a 16-vector. Blocks."""

    @abstractmethod
    def stream(self, q_traj: np.ndarray, rate_hz: float) -> None:
        """Execute an `(N, 16)` trajectory at `rate_hz`. Blocks.

        `rate_hz` is 50 for retrieval or 30 for policy execution
        (conventions.md section 5).
        """

    @abstractmethod
    def set_gripper(self, side: str, g: float) -> None:
        """Command one gripper. `side` in {"left", "right"}, `g` in [0, 1]."""

    def home(self) -> None:
        """Move both arms to HOME. Every skill starts and ends here."""
        self.move_to(config.home_q(), HOME_DURATION_S)

    # --- shared validation --------------------------------------------------

    @staticmethod
    def validate_q(q: np.ndarray) -> np.ndarray:
        """Coerce to a `(16,)` float64 vector, or raise."""
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape != (N_Q,):
            raise ValueError(f"expected a {N_Q}-vector, got shape {np.shape(q)}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("joint vector contains non-finite values")
        return arr

    @staticmethod
    def validate_side(side: str) -> str:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        return side

    @staticmethod
    def validate_gripper(g: float) -> float:
        g = float(g)
        if not 0.0 - 1e-9 <= g <= 1.0 + 1e-9:
            raise ValueError(f"gripper command must be in [0, 1] (0 = closed), got {g}")
        return float(np.clip(g, 0.0, 1.0))
