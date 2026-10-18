# egodex_pick_place — demo + integration bundle

SmolVLA LoRA adapter for the OpenArm `put the cube in the bin` task.
Trained 2026-10-18, 10,000 steps, batch 32, final loss 0.039, on 370 episodes
of MuJoCo-rendered MimicGen data derived from EgoDex human demonstrations.

```
policy.py            inference API — no simulator needed. Start here for UI/hardware.
watch.py             live MuJoCo viewer, one episode at a time. The demo view.
rollout.py           batch rollout, scored, writes mp4 per episode + summary.json
offline_check.py     open-loop action error and smoothness diagnostic
pretrained_model/    the LoRA adapter (12 MB) + normalisation stats
dataset_meta/        meta/ only (620 KB) — supplies the feature schema to load the policy
videos/              the two successful episodes from the baseline sweep
```

## UI integration

```python
from policy import Policy

p = Policy()                            # loads ./pretrained_model
action = p.act(rgb_hwc, state8)         # -> 8 joint targets
p.reset()                               # between episodes
```

`Policy` has no MuJoCo dependency. `rollout.py` and `watch.py` import it, so
every caller runs the same inference path.

| | |
|---|---|
| image | `(H, W, 3)` RGB, uint8 0..255 or float 0..1, **HWC** |
| state | `(8,)` — right arm joints 1-7 in **radians**, then gripper |
| gripper | **0 = closed, 1 = open** |
| rate | **50 Hz** |
| task | `put the cube in the bin` |
| output | `(8,)` joint position **targets**, same layout |

This is a LoRA adapter, not a standalone model. First load pulls
`lerobot/smolvla_base` and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
Needs `lerobot==0.6.1` with PEFT.

**Load every file in `pretrained_model/`.** The two `*_processor.safetensors`
carry the dataset normalisation statistics; without them actions come out at
the wrong scale and the arm moves dangerously.

## Demo

```bash
python watch.py --loop          # live viewer, fresh scene each episode
python watch.py --seed 42       # replay one scene
python rollout.py --episodes 20 # scored sweep + videos
```

Both default to smoothing alpha 0.3 and a 0.10 m cube-to-bin clearance.

## Two defaults that are load-bearing

**Action smoothing (`--smooth`, default 0.3).** An EMA over the seven arm
joints; the gripper passes through unfiltered because a part-closed gripper
grasps nothing. Raw output breaches the 0.05 rad/cycle limit
(`conventions.md` section 10) on 1.7% of steps, peaking at 0.153 rad. At 0.3
that becomes 0.0%. **Do not put the unsmoothed policy on real hardware.**

**Cube clearance (`--min-clearance`, default 0.10 m).** `scene.pick_zone`
reaches to y = -0.34 while the bin wall stands at y = -0.352, so the trained
distribution can spawn a cube 12 mm from an 80 mm wall with no room for the
gripper to descend. Every cube inside 0.05 m failed in the baseline (0/7).

This **conditions** the success rate — it restricts to a sub-region of the
trained distribution (never outside it), so a number produced this way is not
comparable to an unfiltered sweep. Quote it with the threshold. `0` restores
the full zone.

## Measured performance

**2/20** unconditioned, held-out cube poses, seed 99001.

Failure modes, measured rather than guessed:

- Open-loop arm MAE **0.0219 rad** against a ground-truth per-step motion of
  **0.0038 rad** — the error is ~6x the signal, so it jitters rather than tracks.
- **11 of 18** failures never moved the cube at all.
- Grasps are biased top-down (`library.orientation: top_down` in the data
  generator), and only 0.59% of this arm's configurations sit within 10 degrees
  of vertical, so it works at the edge of the reachable set.
- Closes the gripper too high above the cube.

## Known environment issues

- `torchcodec` will not load (missing `libavutil.so.56`). Use
  `video_backend="pyav"` anywhere a `LeRobotDataset` is constructed.
- `at-eval` and `at-sim-eval` (entry point B, in `workspace/`) do not run here —
  the first hardcodes torchcodec, the second needs calibration assets this track
  never produced. `offline_check.py` and `rollout.py` replace them.
- Loading the adapter needs `workspace/`'s interpreter: lerobot 0.6.1 with PEFT,
  against 0.4.4 at the repo root. `cd workspace && uv sync`, then run these
  scripts with `workspace/.venv/bin/python`.
- `ERH_ROOT` and `AT_ROOT` env vars override the paths to the repo root and to
  `workspace/` if either moves.
