"""Closed-loop rollout of the policy in the ERH MuJoCo scene, scored, with video.

Batch sibling of `watch.py`. Both drive `policy.Policy`, so the scored numbers
and the live demo run the same inference path.

`at-sim-eval` is not used: it needs the AdapterTraining setup (`--calib`,
`--background`, `sources.hdf5`) this project never produced. ERH's scene is the
stricter target anyway, being the exact environment the training frames were
rendered in.

Closes the loop the way the dataset was recorded (`sim/sim_robot.Episode`):
`state[i]` and `image[i]` are observed *before* the command, `action[i]` is the
command then sent. Each iteration renders, asks the policy, streams one control
cycle at `exec.rate_hz`.

Scenes come from `generate_dataset.sample_scenes`, the draw the training data
used, seeded differently so the cube poses are held out.

    python rollout.py --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ERH = Path(os.environ.get("ERH_ROOT", HERE.parent))
AT = Path(os.environ.get("AT_ROOT", ERH / "workspace"))
for _p in (HERE, ERH, AT / "src", AT / "plugins/lerobot_robot_openarm_overlay"):
    sys.path.insert(0, str(_p))

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen; watch.py sets glx first

import av  # noqa: E402

from common import config  # noqa: E402
from data.dataset_writer import state_slots  # noqa: E402
from data.generate_dataset import sample_scenes  # noqa: E402
from policy import ARM, CAMERA, CHECKPOINT, HW_LIMIT, SMOOTH_ALPHA, Policy  # noqa: E402
from sim.execute import cube_in_bin  # noqa: E402
from sim.sim_robot import SimRobot  # noqa: E402

# Metres the cube must clear the bin's outer wall by, by default.
#
# `scene.pick_zone` reaches to y = -0.34 while the bin wall stands at y = -0.352,
# so the trained distribution can spawn a cube 12 mm from the wall, leaving the
# gripper nowhere to descend beside it. Measured on the first 20-episode sweep,
# every cube inside 0.05 m failed (0/7).
#
# This CONDITIONS the success rate. It restricts to a sub-region of the
# distribution the policy trained on and never samples outside it, but a number
# produced this way is not comparable to an unfiltered sweep and must be quoted
# with the threshold. Pass `--min-clearance 0` for the full zone.
DEFAULT_CLEARANCE = 0.10


def bin_clearance(cube_xy) -> float:
    """Distance from the cube centre to the bin's outer rectangle, metres (0 = touching)."""
    s = config.skills()["scene"]
    centre = np.asarray(s["bin"]["center"], dtype=np.float64)
    half = np.asarray(s["bin"]["inner_size"], dtype=np.float64) / 2.0 + float(
        s["bin"]["wall_thickness"]
    )
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(cube_xy) - centre) - half, 0.0)))


def filtered_scenes(n: int, seed: int, min_clearance: float) -> list:
    """`n` scenes from the training draw, keeping only those clear of the bin."""
    if min_clearance <= 0.0:
        return sample_scenes(n, seed)
    kept, batch = [], max(n * 8, 64)
    for cand in sample_scenes(batch, seed):
        if bin_clearance(cand[0]) >= min_clearance:
            kept.append(cand)
        if len(kept) == n:
            return kept
    raise SystemExit(
        f"only {len(kept)} of {batch} sampled scenes clear {min_clearance} m; lower the threshold"
    )


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "threads": "2"}
        for f in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        container.mux(stream.encode())


def rollout(robot, policy: Policy, cube_xy, cube_yaw, rng, max_steps):
    """One episode. Returns (success, reason, frames, diagnostics)."""
    cfg = config.skills()["exec"]
    rate = float(cfg["rate_hz"])
    half = float(config.skills()["scene"]["cube"]["size"]) / 2.0
    slots = state_slots(robot.index, "right")

    robot.reset()
    robot.randomize_appearance(rng)  # the per-episode jitter the training data had
    quat = np.array([np.cos(cube_yaw / 2.0), 0.0, 0.0, np.sin(cube_yaw / 2.0)])
    robot.set_free_body_pose("cube_free", [cube_xy[0], cube_xy[1], half], quat)
    policy.reset()

    renderer = robot.renderer()
    frames: list[np.ndarray] = []
    arm_steps: list[float] = []
    prev_arm = None

    def diag():
        a = np.asarray(arm_steps) if arm_steps else np.zeros(1)
        return {
            "max_dq_arm": float(a.max()),
            "mean_dq_arm": float(a.mean()),
            "frac_over_limit": float((a > HW_LIMIT).mean()),
        }

    for _ in range(max_steps):
        renderer.update_scene(robot.data, camera=CAMERA)
        img = renderer.render()  # HWC uint8
        frames.append(img.copy())

        q16 = np.asarray(robot.index.read_qpos(robot.data.qpos), dtype=np.float64)
        a8 = policy.act(img, q16[slots])

        cmd = q16.copy()
        cmd[slots] = a8
        # Arm joints only: the gripper is normalised 0..1, and a legitimate
        # open/close is a step of 1.0 that would saturate any max over all eight.
        if prev_arm is not None:
            arm_steps.append(float(np.abs(a8[ARM] - prev_arm).max()))
        prev_arm = a8[ARM].copy()

        try:
            robot.stream(cmd[None, :], rate)
        except Exception as e:  # a diverged policy can destabilise the physics
            return False, f"sim error: {type(e).__name__}: {str(e)[:60]}", frames, diag()

        ok, _ = cube_in_bin(robot)
        if ok:
            break

    robot.settle(float(cfg["settle_s"]))  # conventions.md section 11 judges at rest
    ok, reason = cube_in_bin(robot)
    return ok, ("" if ok else reason), frames, diag()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1200, help="~24 s at 50 Hz; training mean 711")
    ap.add_argument("--seed", type=int, default=99001, help="NOT the training seed; scenes held out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "rollouts")
    ap.add_argument("--save-failures", action="store_true")
    ap.add_argument("--smooth", type=float, default=SMOOTH_ALPHA,
                    help=f"EMA weight on the new arm command (default {SMOOTH_ALPHA}); 1.0 = off")
    ap.add_argument("--min-clearance", type=float, default=DEFAULT_CLEARANCE,
                    help=f"metres the cube must clear the bin wall by (default {DEFAULT_CLEARANCE}); "
                         f"conditions the result, so quote it. 0 for the full zone.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fps = int(float(config.skills()["exec"]["rate_hz"]))
    policy = Policy(checkpoint=args.checkpoint, device=args.device, smooth_alpha=args.smooth)

    # enforce_safety=False: the conventions.md section 10 per-cycle limits are a
    # hardware guard. A policy that trips one should be measured, not crash the
    # run; `frac_over_limit` reports what it would have cost on the real arm.
    robot = SimRobot(render=True, enforce_safety=False)
    scenes = filtered_scenes(args.episodes, args.seed, args.min_clearance)

    print(f"checkpoint={args.checkpoint}\ndevice={policy.device} episodes={args.episodes} "
          f"seed={args.seed} smooth={args.smooth} min_clearance={args.min_clearance} m")
    results, t0 = [], time.time()
    for i, (cube_xy, cube_yaw) in enumerate(scenes):
        rng = np.random.default_rng(args.seed + i)
        ok, reason, frames, diag = rollout(robot, policy, cube_xy, cube_yaw, rng, args.max_steps)
        path = ""
        if ok or args.save_failures:
            path = str(args.out / f"ep{i:02d}_{'ok' if ok else 'fail'}.mp4")
            write_video(Path(path), frames, fps)
        results.append(
            {"episode": i, "success": ok, "reason": reason, "frames": len(frames), "video": path,
             **{k: round(v, 4) for k, v in diag.items()},
             "cube_xy": [round(c, 4) for c in cube_xy], "cube_yaw": round(cube_yaw, 4),
             "bin_clearance": round(bin_clearance(cube_xy), 4)}
        )
        print(f"  ep{i:02d} {'SUCCESS' if ok else 'fail':8s} {len(frames):4d} frames  "
              f"arm_dq max={diag['max_dq_arm']:.3f} mean={diag['mean_dq_arm']:.4f} "
              f"over_limit={100 * diag['frac_over_limit']:.1f}%  {reason[:38]}")

    n_ok = sum(r["success"] for r in results)
    summary = {
        "checkpoint": str(args.checkpoint), "episodes": args.episodes, "successes": n_ok,
        "success_rate": n_ok / max(args.episodes, 1), "seed": args.seed,
        "smooth_alpha": args.smooth, "min_clearance": args.min_clearance,
        "mean_dq_arm": round(float(np.mean([r["mean_dq_arm"] for r in results])), 5),
        "frac_over_limit": round(float(np.mean([r["frac_over_limit"] for r in results])), 4),
        "elapsed_s": round(time.time() - t0, 1), "results": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n=== {n_ok}/{args.episodes} succeeded ({100 * n_ok / max(args.episodes, 1):.0f}%) "
          f"in {summary['elapsed_s']}s -> {args.out} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
