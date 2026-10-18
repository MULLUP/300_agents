"""Watch the policy drive the arm live in the MuJoCo viewer. The demo view.

The batch sibling, `rollout.py`, renders offscreen to mp4 and scores many
episodes. This opens an interactive window so you can orbit the camera while
the policy runs. Both drive `policy.Policy`, so what you watch is the same
inference path that produces the scored numbers.

The viewer needs a windowed GL context, so MUJOCO_GL is forced to `glx` BEFORE
importing `rollout` (which otherwise defaults it to `egl`). The policy's own
observations still come from an offscreen `mujoco.Renderer` of the overhead
camera, which glx also serves.

    python watch.py                 # random scene, closes when you close the window
    python watch.py --loop          # keep drawing new scenes
    python watch.py --seed 42       # pin one scene to replay it
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from pathlib import Path

# Must precede the `rollout` import: it does setdefault("MUJOCO_GL", "egl"),
# and an EGL context cannot open a window.
os.environ.setdefault("MUJOCO_GL", "glx")

sys.path.insert(0, str(Path(__file__).parent))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

import rollout as R  # noqa: E402
from common import config  # noqa: E402
from data.dataset_writer import state_slots  # noqa: E402
from policy import ARM, CAMERA, CHECKPOINT, HW_LIMIT, SMOOTH_ALPHA, Policy  # noqa: E402
from sim.execute import cube_in_bin  # noqa: E402
from sim.sim_robot import SimRobot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--seed", type=int, default=None, help="default: a fresh random scene each run")
    ap.add_argument("--max-steps", type=int, default=1200, help="~24 s at 50 Hz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time; lower is slower")
    ap.add_argument("--free-camera", action="store_true", help="orbit freely instead of overhead")
    ap.add_argument("--loop", action="store_true", help="keep drawing scenes until the window closes")
    ap.add_argument("--smooth", type=float, default=SMOOTH_ALPHA,
                    help=f"EMA weight on the new arm command (default {SMOOTH_ALPHA}); 1.0 = off")
    ap.add_argument("--min-clearance", type=float, default=R.DEFAULT_CLEARANCE,
                    help=f"metres the cube must clear the bin wall by (default "
                         f"{R.DEFAULT_CLEARANCE}); 0 for the full trained zone")
    args = ap.parse_args()

    cfg = config.skills()["exec"]
    rate = float(cfg["rate_hz"])
    settle_s = float(cfg["settle_s"])
    half = float(config.skills()["scene"]["cube"]["size"]) / 2.0
    period = 1.0 / (rate * max(args.speed, 1e-6))

    policy = Policy(checkpoint=args.checkpoint, device=args.device, smooth_alpha=args.smooth)
    print(f"checkpoint={args.checkpoint}  device={policy.device}  smooth={args.smooth}")

    robot = SimRobot(render=True, enforce_safety=False)  # see rollout.py on why safety is off
    slots = state_slots(robot.index, "right")
    renderer = robot.renderer()

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        if not args.free_camera:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = robot.model.camera(CAMERA).id

        episode = 0
        while viewer.is_running():
            seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)
            (cube_xy, cube_yaw) = R.filtered_scenes(1, seed, args.min_clearance)[0]
            rng = np.random.default_rng(seed)

            robot.reset()
            robot.randomize_appearance(rng)
            quat = np.array([np.cos(cube_yaw / 2.0), 0.0, 0.0, np.sin(cube_yaw / 2.0)])
            robot.set_free_body_pose("cube_free", [cube_xy[0], cube_xy[1], half], quat)
            policy.reset()
            viewer.sync()

            print(f"\nepisode {episode}  seed={seed}  "
                  f"cube=({cube_xy[0]:.3f}, {cube_xy[1]:.3f})  yaw={np.degrees(cube_yaw):+.1f} deg  "
                  f"clearance={R.bin_clearance(cube_xy):.3f} m")

            ok, step, max_dq, prev = False, 0, 0.0, None
            while viewer.is_running() and step < args.max_steps:
                tick = time.perf_counter()

                renderer.update_scene(robot.data, camera=CAMERA)
                img = renderer.render()
                q16 = np.asarray(robot.index.read_qpos(robot.data.qpos), dtype=np.float64)
                a8 = policy.act(img, q16[slots])

                cmd = q16.copy()
                cmd[slots] = a8
                # Arm joints only; the gripper's 0..1 toggle would pin this at 1.0.
                if prev is not None:
                    max_dq = max(max_dq, float(np.abs(a8[ARM] - prev).max()))
                prev = a8[ARM].copy()

                try:
                    robot.stream(cmd[None, :], rate)
                except Exception as e:
                    print(f"  sim error at step {step}: {type(e).__name__}: {str(e)[:60]}")
                    break

                viewer.sync()
                step += 1
                ok, _ = cube_in_bin(robot)
                if ok:
                    break
                if step % 50 == 0:
                    print(f"  step {step:4d}  max_dq={max_dq:.3f} rad", end="\r", flush=True)

                # Hold real time when inference is faster than the control period.
                time.sleep(max(0.0, period - (time.perf_counter() - tick)))

            if viewer.is_running():
                robot.settle(settle_s)
                viewer.sync()
                ok, reason = cube_in_bin(robot)
                flag = "" if max_dq <= HW_LIMIT else f"  OVER the {HW_LIMIT} rad/cycle hardware limit"
                print(f"  {'SUCCESS' if ok else f'fail — {reason}'}\n"
                      f"  {step} steps, max_dq={max_dq:.3f} rad{flag}")
                time.sleep(1.5)  # let the result sit on screen

            episode += 1
            if not args.loop or args.seed is not None:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
