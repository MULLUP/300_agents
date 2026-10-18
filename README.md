# Mixture and Modality — human video to a robot manipulation skill

One pipeline, `ego -> sim -> robot`. A person's demonstration supplies the motion; a
physics simulator replays it into many scenes and keeps the attempts that succeed; the
survivors train a LoRA adapter for an open-weight vision-language-action model. No
teleoperation, and no robot time to collect the data.

Two entry points feed the same three stages. The data source, the robot (a URDF) and the
base model are each replaceable without touching the rest.

| | lives in | input | status |
|---|---|---|---|
| **A** — egocentric factory | the repo root | the public EgoDex dataset | ran end to end; the trained policy is committed under `policy/` |
| **B** — skill from clips | `workspace/` | 1–5 clips filmed at the robot, one command | every stage runs and is gated; validated on synthetic scenes only |

Written for the ERH Physical AI Hackathon. The report released alongside this repo carries
the argument, the related work and the business case; this file covers getting the code to
run and the things that will bite you.

## A — EgoDex to a trained policy

```
EgoDex human video
  -> data/retarget.py          hand tracking -> gripper-pose trajectories at 50 Hz
  -> common/mimic.py           object-centric transform: one demo onto many scenes
  -> sim/execute.py            replay in MuJoCo, physics decides what succeeded
  -> data/generate_dataset.py  -> LeRobot v3.0 dataset (370 episodes, 263k frames)
  -> SmolVLA LoRA              -> policy/pretrained_model
  -> policy/rollout.py         closed-loop rollout in the same scene, scored
```

The multiplication is the point: 179 retargeted source demos x 10 sampled cube poses = 1790
trials, of which physics kept 26%.

| path | |
|---|---|
| `common/` | transforms, IK, the MimicGen object-centric transform, joint indexing |
| `sim/` | MuJoCo scene generation, execution, reachability and feasibility studies |
| `data/` | EgoDex ingest, retargeting, dataset generation, the shard writer |
| `config/` | every tunable, with the measurement that justifies it |
| `policy/` | trained adapter, inference API, live viewer, scored rollout |
| `tools/` | `recover_parquet.py` rebuilds a LeRobot shard whose writer was killed; `mimicgen_clip.py` renders the fan-out |
| `docs/` | measurements and figures behind the config values |

`sim/`, `data/` and `policy/` each carry a README with the measurements behind their
choices — read those before changing anything in them.

`third_party/` and the generated data are gitignored; the 541 MB dataset is not in the repo.

```bash
uv sync
pytest tests/                   # 168 pass with no dataset; one more needs MUJOCO_GL

# Anything under policy/ loads a LoRA adapter, so it needs lerobot >= 0.6 with PEFT.
# This repo's own .venv is lerobot 0.4.4 with neither. Entry point B pins the right one:
cd workspace && uv sync && cd ..
export POLICY_PY=$PWD/workspace/.venv/bin/python

$POLICY_PY policy/watch.py --loop        # watch the trained policy, live
$POLICY_PY policy/rollout.py --episodes 20   # score it

# regenerate the dataset (~2 h on 6 workers)
MUJOCO_GL=egl setsid nohup python data/generate_dataset.py \
    --trials 10 --processes 6 --name egodex_pick_place --root /path/to/output &
```

The `setsid nohup` is not optional — see "Known issues".

## B — a skill from a few clips at the workspace

`workspace/` takes clips filmed at the robot's own workspace, a skill name and a one-line
description, and runs the whole chain with no task-specific code: hands to gripper poses,
objects named and reconstructed in 3D, the task *inferred* from where grasps and releases
fall, DexMimicGen-style generation, a LeRobot dataset, a LoRA adapter, and a closed-loop
simulated test that decides whether the new version is promoted. Every stage has a quality
gate that stops the job with a plain-language reason instead of training on bad data.

```bash
cd workspace
uv sync
uv run at-run --job jobs/example_stack_sponges.json
```

It is a separate `uv` project because it pins `lerobot==0.6.1` with PEFT against the root's
0.4.4 — which is also the interpreter entry point A's `policy/` scripts borrow above.
`workspace/README.md` is the command reference; `workspace/QUICKSTART.md` is the runbook,
including gated model access and running on an H200 via Modal.

Nothing here has been run on real clips or on a GPU: WiLoR, SAM 3, SAM 3D Objects, π0.5 and
GR00T are all wired in but unexercised. The numbers it reports come from synthetic scenes
with known ground truth.

## What came out

**2/20** in simulation on held-out cube poses, unconditioned, and on the real robot the
policy ran but did not complete the task.

The policy fits the mapping but not the precision: its open-loop action error is about six
times the per-step signal, so it jitters rather than tracks, and that compounds in closed
loop. 11 of 18 failures never moved the cube at all. `policy/README.md` has the measured
breakdown and the three causes that are fixable by one regenerate plus retrain — a top-down
grasp bias inherited from the data generator, a pick zone that can spawn a cube 12 mm from
the bin wall, and a gripper that closes too high. The report argues what that means for the
approach.

Sim-to-real was not attempted beyond that single run: the policy has seen nothing but
MuJoCo renders from one fixed overhead camera.

## Known issues

- **`generate_dataset.py` loses every shard's parquet footer on SIGHUP.** `util.Finalize`
  covers a graceful pool shutdown but not a closing terminal, and LeRobot writes the parquet
  footer only on `close()`. `info.json` and `stats.json` still look current, so the shards
  appear healthy while being unreadable. Run detached. To recover,
  `tools/recover_parquet.py` rebuilds the footer by walking page headers; `--self-test`
  proves the method against an intact file first.
- `meta/episodes` is written through a 10-deep buffer, so a killed run loses up to 9
  episodes per shard even after footer recovery. That is why the dataset is 370 episodes
  and not the 391 that were generated.
- `torchcodec` needs `libavutil.so.56`; where it is missing pass `video_backend="pyav"`.

## Conventions

The contract the code obeys: the `table` frame, a 50 Hz control rate, and the motion limits
of 0.05 rad per joint per cycle and 0.25 m/s at the TCP, which both the generator and
`sim/sim_robot.py`'s safety check enforce.

`config/skills.yaml` holds every tunable with the measurement that set it — read the comment
before changing a number. Code comments cite sections of a `conventions.md` and a
`docs/conventions_proposals.md` that have since been removed; the surviving record of those
decisions is the config comments.
