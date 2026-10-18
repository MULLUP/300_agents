# 300_agents — human video to an OpenArm manipulation policy

One human demonstration becomes many robot trajectories, and those train a
visuomotor policy for `put the cube in the bin` on a 7-DOF OpenArm.

```
EgoDex human video
  -> data/retarget.py          hand tracking -> gripper-pose trajectories at 50 Hz
  -> common/mimic.py           object-centric transform: one demo onto many scenes
  -> sim/execute.py            replay in MuJoCo, physics decides what succeeded
  -> data/generate_dataset.py  -> LeRobot v3.0 dataset (370 episodes, 263k frames)
  -> SmolVLA LoRA              -> policy/pretrained_model
  -> policy/rollout.py         closed-loop rollout in the same scene, scored
```

The multiplication is the point: 179 retargeted source demos x 10 sampled cube
poses = 1790 trials, of which physics kept 26%.

## Layout

| path | |
|---|---|
| `common/` | transforms, IK, the MimicGen object-centric transform, joint indexing |
| `sim/` | MuJoCo scene generation, execution, reachability and feasibility studies |
| `data/` | EgoDex ingest, retargeting, dataset generation, the shard writer |
| `config/` | every tunable, with the measurement that justifies it |
| `policy/` | trained adapter, inference API, live viewer, scored rollout |
| `tools/` | `recover_parquet.py`, rebuilds a LeRobot shard whose writer was killed |
| `docs/` | measurements and figures behind the config values |

`third_party/` and `data/` are gitignored. The 541 MB dataset is not in the repo.

## Quickstart

```bash
uv sync

# Anything under policy/ needs lerobot >= 0.6 with PEFT, to load the LoRA
# adapter. This repo's own .venv is lerobot 0.4.4 with no PEFT and cannot.
export POLICY_PY=/mnt/shared_s/model_data/AdapterTraining/.venv/bin/python

# watch the trained policy, live
$POLICY_PY policy/watch.py --loop

# score it
$POLICY_PY policy/rollout.py --episodes 20

# regenerate the dataset (~2 h on 6 workers)
MUJOCO_GL=egl setsid nohup python data/generate_dataset.py \
    --trials 10 --processes 6 --name egodex_pick_place --root /path/to/output &
```

The `setsid nohup` is not optional — see "Known issues".

## Results

**2/20** in simulation on held-out cube poses, unconditioned.

The policy fits the mapping but not the precision. Measured open loop on
held-out episodes:

| | |
|---|---|
| arm action MAE | **0.0219 rad** |
| ground-truth per-step motion | **0.0038 rad** |
| policy per-step motion | 0.0127 rad (**3.3x** the data) |
| steps over the 0.05 rad/cycle hardware limit | policy 1.7%, data 0.0% |

Its error is ~6x the per-step signal, so it jitters rather than tracks, and that
compounds in closed loop. **11 of 18 failures never moved the cube at all.**

Contributing causes, measured rather than guessed:

- **Top-down bias.** `library.orientation: top_down` forces a vertical approach
  into every source demo, but only 0.59% of this arm's configurations put the
  approach axis within 10 degrees of vertical (5.5% within 30). The policy works
  at the edge of the reachable set.
- **Scene geometry.** `scene.pick_zone` reached to y = -0.34 while the bin wall
  stands at y = -0.352, so a cube could spawn 12 mm from an 80 mm wall. Every
  cube inside 0.05 m failed (0/7). The rollout now enforces 0.10 m clearance by
  default and the bin was lowered 30%; neither is baked into the dataset yet.
- **Grasp height.** The gripper closes above the cube. `exec.grasp_tcp_offset`
  is still 0.0, and the TCP site sits near the fingertips rather than the pad
  midpoint.

Fixing the first three needs one regenerate plus retrain, not a policy change.

Sim-to-real was not attempted: the policy has seen nothing but MuJoCo renders
from a single fixed overhead camera.

## Known issues

- **`generate_dataset.py` loses every shard's parquet footer on SIGHUP.**
  `util.Finalize` covers a graceful pool shutdown but not a closing terminal,
  and LeRobot writes the parquet footer only on `close()`. `info.json` and
  `stats.json` still look current, so the shards appear healthy while being
  unreadable. Run detached. To recover, `tools/recover_parquet.py` rebuilds the
  footer by walking page headers; `--self-test` proves the method against an
  intact file first.
- `meta/episodes` is written through a 10-deep buffer, so a killed run loses up
  to 9 episodes per shard even after footer recovery. That is why the dataset is
  370 episodes and not the 391 that were generated.
- `torchcodec` needs `libavutil.so.56`; where it is missing pass
  `video_backend="pyav"`.

## Conventions

The contract the code obeys: the `table` frame, a 50 Hz control rate, and the
motion limits of 0.05 rad per joint per cycle and 0.25 m/s at the TCP, which
both the generator and `sim/sim_robot.py`'s safety check enforce.

`config/skills.yaml` holds every tunable with the measurement that set it —
read the comment before changing a number. Code comments cite sections of a
`conventions.md` and a `docs/conventions_proposals.md` that have since been
removed; the surviving record of those decisions is the config comments.
