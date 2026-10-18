"""Offline action error on held-out episodes, arm and gripper reported separately.

`at-eval` can't run here: it hardcodes the torchcodec video backend, which needs
libavutil.so.56 this box doesn't have. This does the same job through
`policy.Policy`, the path the rollouts use.

Needs the FULL dataset (data/ and videos/), not the `dataset_meta/` shipped
here — point `--dataset-root` at it.

Two questions, both raised by the 2/20 rollout:

1. Did the policy learn the mapping at all? If open-loop error is small but
   closed-loop behaviour is bad, the problem is compounding error, not fitting.
2. Is the motion "shaky"? Compares the per-step change the policy commands
   against the per-step change the *ground-truth* actions make. The arm's seven
   joints are reported apart from the gripper, because the gripper is normalised
   0..1 and a legitimate open/close saturates any max over all eight.

Measured on the 10k-step checkpoint: arm MAE 0.0219 rad against a ground-truth
per-step motion of 0.0038 rad — the error is ~6x the signal. The policy moves
3.3x as much per step as its training data and breaches the hardware limit on
1.7% of steps, where the data never does.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from policy import ARM, CAMERA, CHECKPOINT, GRIP, HW_LIMIT, Policy  # noqa: E402

FULL_DATASET = Path(os.environ.get("EGODEX_DATASET", "/mnt/shared_s/model_data/erh/lerobot/egodex_pick_place"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--dataset-root", type=Path, default=FULL_DATASET)
    ap.add_argument("--episodes", type=int, nargs="*", default=[333, 340, 350, 360])
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # smooth_alpha=1.0: this measures the raw policy, not the filtered command.
    policy = Policy(checkpoint=args.checkpoint, device=args.device, smooth_alpha=1.0)
    ds = LeRobotDataset("local/egodex_pick_place", root=args.dataset_root, video_backend="pyav")
    print(f"checkpoint={args.checkpoint}\ndevice={policy.device} episodes={args.episodes}\n")

    all_err, pred_d, true_d = [], [], []
    for ep in args.episodes:
        meta = ds.meta.episodes[ep]
        first, last = meta["dataset_from_index"], meta["dataset_to_index"]
        last = min(last, first + args.max_frames)
        policy.reset()
        preds, trues = [], []
        for idx in range(first, last):
            item = ds[idx]
            img = (item[f"observation.images.{CAMERA}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            preds.append(policy.act(img, item["observation.state"].numpy()))
            trues.append(item["action"].numpy())
        preds, trues = np.stack(preds), np.stack(trues)
        all_err.append(np.abs(preds - trues))
        pred_d.append(np.abs(np.diff(preds, axis=0)))
        true_d.append(np.abs(np.diff(trues, axis=0)))
        e = all_err[-1]
        print(f"  ep{ep}: {len(preds):4d} frames  arm MAE {e[:, ARM].mean():.4f} rad"
              f"  grip MAE {e[:, GRIP].mean():.4f}")

    err = np.concatenate(all_err)
    pd_, td_ = np.concatenate(pred_d), np.concatenate(true_d)

    print("\n--- action error (held-out, open loop) ---")
    print(f"  arm     MAE {err[:, ARM].mean():.4f} rad   max {err[:, ARM].max():.4f} rad")
    print(f"  gripper MAE {err[:, GRIP].mean():.4f}      max {err[:, GRIP].max():.4f}")
    print(f"  per-joint MAE (rad): {np.round(err[:, ARM].mean(axis=0), 4).tolist()}")

    print("\n--- smoothness: per-step change in commanded arm joints ---")
    print(f"  ground truth  mean {td_[:, ARM].mean():.4f}  p99 {np.percentile(td_[:, ARM], 99):.4f}"
          f"  max {td_[:, ARM].max():.4f} rad")
    print(f"  policy        mean {pd_[:, ARM].mean():.4f}  p99 {np.percentile(pd_[:, ARM], 99):.4f}"
          f"  max {pd_[:, ARM].max():.4f} rad")
    print(f"  steps over the {HW_LIMIT} rad/cycle hardware limit:"
          f"  policy {100 * (pd_[:, ARM] > HW_LIMIT).mean():.1f}%"
          f"   ground truth {100 * (td_[:, ARM] > HW_LIMIT).mean():.1f}%")
    print(f"  policy moves {pd_[:, ARM].mean() / max(td_[:, ARM].mean(), 1e-9):.1f}x as much "
          f"per step as the data it was trained on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
