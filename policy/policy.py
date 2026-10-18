"""Minimal inference API for the cube-in-bin SmolVLA adapter.

No simulator dependency: give it an image and a joint state, get a joint
command. This is the entry point for UI or hardware integration; `rollout.py`
and `watch.py` drive the MuJoCo scene and import from here, so every caller
runs the same inference path.

    from policy import Policy
    p = Policy()                       # loads ./pretrained_model
    action = p.act(rgb_hwc_uint8, state8)   # -> 8 floats

Checkpoint is a LoRA adapter, not a standalone model. First load downloads
`lerobot/smolvla_base` and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors

HERE = Path(__file__).parent
CHECKPOINT = HERE / "pretrained_model"
DATASET_META = HERE / "dataset_meta"  # meta/ only; supplies the feature schema

TASK = "put the cube in the bin"
ROBOT_TYPE = "openarm_bimanual"
CAMERA = "overhead"
FPS = 50

# The 8 commanded slots are the right arm's 7 joints (radians) then the gripper
# (0 = closed, 1 = open).
ARM = slice(0, 7)
GRIP = 7

# conventions.md section 10: the arm rejects a command moving any joint more
# than this in one 1/FPS cycle. Measured raw, this policy breaches it on 1.7% of
# steps (peak 0.153 rad); at SMOOTH_ALPHA 0.3 it breaches on 0.0%.
HW_LIMIT = 0.05
SMOOTH_ALPHA = 0.3


class Policy:
    """Loaded adapter plus the action smoothing the arm needs.

    `smooth_alpha` is the EMA weight on each new arm command; 1.0 disables it.
    The gripper is never smoothed — a part-closed gripper grasps nothing.
    """

    def __init__(
        self,
        checkpoint: Path = CHECKPOINT,
        dataset_meta: Path = DATASET_META,
        device: str | None = None,
        smooth_alpha: float = SMOOTH_ALPHA,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = float(smooth_alpha)
        self._state: np.ndarray | None = None

        meta = LeRobotDatasetMetadata("local/egodex_pick_place", root=dataset_meta)
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        cfg.device = self.device
        self.policy = make_policy(cfg, ds_meta=meta)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def reset(self) -> None:
        """Call between episodes: clears the action chunk queue and the filter."""
        self.policy.reset()
        self._state = None

    def act(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        """One step.

        image: (H, W, 3) RGB, uint8 0..255 or float 0..1.
        state: (8,) right arm joints 1-7 in radians, then gripper 0..1.
        returns: (8,) joint position targets in the same layout.
        """
        img = np.asarray(image)
        if img.dtype != np.float32 or img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        obs = {
            f"observation.images.{CAMERA}": img,
            "observation.state": np.asarray(state, dtype=np.float32),
        }
        action = predict_action(
            obs, self.policy, torch.device(self.device), self.pre, self.post,
            use_amp=False, task=TASK, robot_type=ROBOT_TYPE,
        )
        a = action.squeeze(0).float().cpu().numpy().astype(np.float64)
        return self._smooth(a)

    def _smooth(self, a: np.ndarray) -> np.ndarray:
        if self.alpha >= 1.0:
            return a
        if self._state is None:
            self._state = a.copy()
            return a
        self._state[ARM] = self.alpha * a[ARM] + (1.0 - self.alpha) * self._state[ARM]
        self._state[GRIP] = a[GRIP]
        return self._state.copy()
