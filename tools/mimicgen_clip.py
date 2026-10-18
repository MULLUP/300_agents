"""Render a short clip of MimicGen fanning one human demo into many trajectories.

For the slide deck. Shows the thing that makes the pipeline work: a single
retargeted EgoDex demonstration re-anchored by `common.mimic.retarget` onto
dozens of sampled cube poses, each becoming its own feasible robot trajectory.

The trajectories are drawn as 3D line strips straight into the MuJoCo scene
(`mjv_connector`), so they are projected by the same camera that renders the
robot. No physics and no IK are run — this visualises the object-centric
transform itself, which is what makes generating tens of trajectories take
well under a second.

    python tools/mimicgen_clip.py --trajectories 40 --out docs/mimicgen.mp4
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import av  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from common import config, demos  # noqa: E402
from common.mimic import demo_anchors, heading_rotation, retarget  # noqa: E402
from common.transforms import yaw_of  # noqa: E402
from data.generate_dataset import sample_scenes  # noqa: E402
from sim.execute import bin_place_target  # noqa: E402
from sim.sim_robot import SimRobot  # noqa: E402

TRAIL_RADIUS = 0.0035  # m, drawn capsule radius
CUBE_MARKER = 0.0225  # m, half the cube edge


def tcp_paths(demo, scenes, rate_hz: float) -> list[np.ndarray]:
    """One (N, 3) TCP position path per sampled scene, straight from `retarget`."""
    half = float(config.skills()["scene"]["cube"]["size"]) / 2.0
    offset = float(config.skills()["exec"]["grasp_tcp_offset"])
    paths = []
    for cube_xy, cube_yaw in scenes:
        c, s = np.cos(cube_yaw), np.sin(cube_yaw)
        T_cube = np.eye(4)
        T_cube[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T_cube[:3, 3] = [cube_xy[0], cube_xy[1], half + offset]

        # Same two-step target construction `sim/execute.run_demo` uses: the
        # heading alignment depends on the target's position only, so the
        # aligned release yaw can be computed before the target exists.
        T_grasp, T_release = demo_anchors(demo)
        align = heading_rotation(T_grasp, T_release, T_cube, bin_place_target())
        T_tgt = bin_place_target(float(yaw_of(align @ T_release)))

        rt = retarget(demo, T_cube, T_tgt, rate_hz)
        paths.append(rt.T_table_tcp[:, :3, 3].copy())
    return paths


def add_strip(scene, pts: np.ndarray, rgba, radius: float = TRAIL_RADIUS) -> None:
    """Draw a polyline into the live MjvScene as connected capsules."""
    for a, b in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        scene.ngeom += 1


def add_cube(scene, xy, yaw: float, rgba) -> None:
    """A box marker where this trajectory's cube was sampled."""
    if scene.ngeom >= scene.maxgeom:
        return
    c, s = np.cos(yaw), np.sin(yaw)
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_BOX,
        np.full(3, CUBE_MARKER),
        np.array([xy[0], xy[1], CUBE_MARKER]),
        np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text((16, 14), text, fill=(255, 255, 255))
    if sub:
        d.text((16, 30), sub, fill=(190, 190, 190))
    return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trajectories", type=int, default=40)
    ap.add_argument("--demo", type=Path, default=None, help="source .npz (default: the first)")
    ap.add_argument("--camera", default="overhead", help="overhead / front_camera / side_camera")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds holding the full fan at the end")
    ap.add_argument("--subsample", type=int, default=6, help="keep every Nth TCP sample when drawing")
    ap.add_argument("--out", type=Path, default=Path("docs/mimicgen.mp4"))
    args = ap.parse_args()

    rate_hz = float(config.skills()["exec"]["rate_hz"])
    src = args.demo or sorted(config.path("data", "retargeted").glob("*.npz"))[0]
    demo = demos.load_source_demo(src)
    scenes = sample_scenes(args.trajectories, seed=7)

    print(f"source demo: {src.name}")
    paths = tcp_paths(demo, scenes, rate_hz)
    print(f"retargeted onto {len(paths)} scenes")

    colours = [
        (*colorsys.hsv_to_rgb(i / max(len(paths), 1), 0.85, 1.0), 0.95) for i in range(len(paths))
    ]

    robot = SimRobot(render=True, enforce_safety=False)
    robot.reset()
    size = int(config.skills()["dataset"]["image_size"])
    renderer = mujoco.Renderer(robot.model, height=size, width=size, max_geom=20000)

    reveal = max(args.seconds - args.hold, 0.1)
    n_reveal = int(reveal * args.fps)
    n_hold = int(args.hold * args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(args.out), mode="w") as container:
        stream = container.add_stream("h264", rate=args.fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "threads": "2"}

        for f in range(n_reveal + n_hold):
            # How many trajectories are on screen; eased so the fan accelerates.
            t = min(f / max(n_reveal - 1, 1), 1.0)
            shown = len(paths) if f >= n_reveal else int(round(len(paths) * t * t))

            renderer.update_scene(robot.data, camera=args.camera)
            for i in range(shown):
                add_cube(renderer.scene, scenes[i][0], scenes[i][1], colours[i])
                add_strip(renderer.scene, paths[i][:: args.subsample], colours[i])

            frame = label(
                renderer.render(),
                f"MimicGen: 1 human demo -> {shown} robot trajectories",
                f"{src.name}  |  object-centric retarget, no physics",
            )
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())

    print(f"wrote {args.out}  ({n_reveal + n_hold} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
