"""Where can the right arm grasp top-down? (plan task 1.P1.2 support)

Answers two questions the scene layout depends on, and that `config/skills.yaml`
cannot be guessed for:

1. **What shoulder height** (`T_table_robot.z + 0.697`) maximizes the usable
   top-down tabletop area?
2. **Which (x, y) cells** are actually reachable, so `scene.pick_zone`,
   `scene.bin` and `scene.spot` can be placed inside them rather than guessed.

It also builds the **IK seed bank**. Differential IK is local, and this arm's
top-down configurations sit near `joint1 = -1.2, joint2 = +2.9` while HOME is at
all zeros — seeding from HOME lands in a joint-limit corner tens of millimetres
away. The bank is found by rejection-sampling the joint box for configurations
whose approach axis is near-vertical, then clustering them.

    uv run python sim/reachability.py                  # sweep + plot + seeds
    uv run python sim/reachability.py --no-height-sweep # just the current height

Writes `sim/plots/reachability_right.png` and prints a `ik.seeds` block for
`config/skills.yaml`. Does not edit any config itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.ik import IKSolver, SeedBank, seed_bank_path as bank_path  # noqa: E402
from common.joints import JointIndex  # noqa: E402
from common.transforms import top_down_pose  # noqa: E402

# The shoulder height baked into the current sim/robot_ik.xml. Shifting an IK
# target in z is equivalent to moving the base the other way, so one compiled
# model covers every candidate height.
H_MODEL = 0.697 + float(config.calibration()["T_table_robot"]["pos"][2])

GRASP_Z = 0.0225  # cube centre height = half of scene.cube.size
TILT_TOL_DEG = 10.0  # "near-vertical" for seed harvesting
# Candidate grasp yaws. The gripper's jaw axis is a line, so yaw == yaw + 180;
# a cube is symmetric every 90 deg, so a grasp at yaw and at yaw + 90 both hold
# the same cube. Distinct *arm* poses therefore span [0, 180), and trying all of
# them is free reachability — a cell counts if any one of them solves.
YAWS_DEG = tuple(np.arange(0.0, 180.0, 22.5))


def harvest_seeds(n_samples: int, side: str, seed: int) -> "SeedBank":
    """Rejection-sample the joint box for near-top-down configurations.

    Keeps *every* hit together with its tcp position, so `SeedBank.nearest` can
    warm-start a query from a configuration that is already close in task space.
    A small fixed seed set does not work here: measured cold-start IK recovery
    was 30% with 12 spread seeds.
    """
    model = mujoco.MjModel.from_xml_path(str(config.path("sim", "robot_ik.xml")))
    data = mujoco.MjData(model)
    index = JointIndex(model)
    lo, hi = index.arm_limits(side)
    sid = index.site_id(side)
    slots = index.arm_slots[side]

    rng = np.random.default_rng(seed)
    samples = rng.uniform(lo, hi, size=(n_samples, 7))
    q = config.home_q().copy()

    keep_q, keep_pos = [], []
    cos_tol = -np.cos(np.deg2rad(TILT_TOL_DEG))
    for row in samples:
        q[slots] = row
        index.write_qpos(q, data.qpos)
        mujoco.mj_kinematics(model, data)
        if data.site_xmat[sid].reshape(3, 3)[2, 2] < cos_tol:
            pos = data.site_xpos[sid].copy()
            # Only the half-space in front of the robot is useful: the table is there.
            if pos[0] > 0.10:
                keep_q.append(row.copy())
                keep_pos.append(pos)

    if not keep_q:
        raise RuntimeError("no near-top-down configurations found — check the model")

    return SeedBank(np.asarray(keep_q), np.asarray(keep_pos), side=side)


def reachable_map(
    solver: IKSolver,
    side: str,
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray | float,
    seed_k: int,
    n_restarts: int,
) -> np.ndarray:
    """Boolean `(len(ys), len(xs))` grid: can the tcp reach (x, y, z) top-down?

    A cell counts as reachable if IK converges for **any** yaw in `YAWS_DEG` — a
    cube has 90 deg yaw symmetry, so the grasp yaw is free (conventions.md sec 2).
    """
    grid = np.zeros((len(ys), len(xs)), dtype=bool)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            for yaw in np.deg2rad(YAWS_DEG):
                res = solver.solve_cold(
                    top_down_pose([x, y, float(z)], yaw), side, k=seed_k, n_restarts=n_restarts
                )
                if res.converged:
                    grid[iy, ix] = True
                    break
    return grid


def sweep_heights(solver: IKSolver, side: str, heights, xs, ys, seed_k, n_restarts) -> list[dict]:
    """Reachable-cell count at the grasp height for each candidate shoulder height."""
    rows = []
    for h in heights:
        # Target at table z corresponds to model z + (H_MODEL - h).
        z_model = GRASP_Z + (H_MODEL - h)
        grid = reachable_map(solver, side, xs, ys, z_model, seed_k, n_restarts)
        rows.append({"shoulder_height": float(h), "n_cells": int(grid.sum()), "grid": grid})
        print(
            f"  shoulder_height={h:.3f} -> {grid.sum():4d}/{grid.size} cells reachable top-down",
            flush=True,
        )
    return rows


def plot(rows: list[dict], xs, ys, best: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    ncol = min(n, 5)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), squeeze=False)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    for ax, row in zip(axes.ravel(), rows):
        ax.imshow(row["grid"], origin="lower", extent=extent, aspect="equal", cmap="Greens", vmin=0, vmax=1)
        mark = " <- chosen" if row is best else ""
        ax.set_title(f"h={row['shoulder_height']:.2f} m, {row['n_cells']} cells{mark}", fontsize=8)
        ax.set_xlabel("table x [m]", fontsize=7)
        ax.set_ylabel("table y [m]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Right arm, top-down reachability at tcp z = {GRASP_Z} m (cube centre)\n"
        f"green = IK converged for some yaw within {config.skills()['ik']['pos_tol'] * 1000:.0f} mm / "
        f"{config.skills()['ik']['rot_tol_deg']:.0f} deg",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--side", default="right")
    ap.add_argument("--n-samples", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--step", type=float, default=0.05, help="grid step, m")
    ap.add_argument("--x-range", type=float, nargs=2, default=(0.10, 0.70), metavar=("LO", "HI"))
    ap.add_argument("--y-range", type=float, nargs=2, default=(-0.50, 0.35), metavar=("LO", "HI"))
    ap.add_argument("--seed-k", type=int, default=12, help="nearest seeds tried per target")
    ap.add_argument("--n-restarts", type=int, default=12)
    ap.add_argument("--no-height-sweep", action="store_true")
    ap.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=None,
        help="candidate shoulder heights, m (default: 0.30 .. 0.60)",
    )
    ap.add_argument(
        "--reuse-seeds", action="store_true", help="load sim/ik_seeds_<side>.npz instead of harvesting"
    )
    args = ap.parse_args()

    xs = np.arange(args.x_range[0], args.x_range[1] + 1e-9, args.step)
    ys = np.arange(args.y_range[0], args.y_range[1] + 1e-9, args.step)

    if args.reuse_seeds:
        bank = SeedBank.load(args.side)
        print(f"loaded {len(bank)} seeds from {bank_path(args.side)}")
    else:
        print(f"harvesting top-down seeds from {args.n_samples} samples ...", flush=True)
        bank = harvest_seeds(args.n_samples, args.side, args.seed)
        path = bank.save()
        print(f"  kept {len(bank)} seeds -> {path.relative_to(config.REPO_ROOT)}")

    solver = IKSolver()
    if args.no_height_sweep:
        heights = [H_MODEL]
    elif args.heights:
        heights = list(args.heights)
    else:
        heights = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    print(
        f"sweeping {len(heights)} shoulder height(s) over a {len(xs)}x{len(ys)} grid "
        f"(seed_k={args.seed_k}, restarts={args.n_restarts}) ...",
        flush=True,
    )
    rows = sweep_heights(solver, args.side, heights, xs, ys, args.seed_k, args.n_restarts)
    best = max(rows, key=lambda r: r["n_cells"])
    print(f"\nbest shoulder_height = {best['shoulder_height']:.3f} m ({best['n_cells']} cells)")
    print(f"  -> config/calibration.yaml T_table_robot.pos.z = {best['shoulder_height'] - 0.697:+.3f}")

    out_png = config.path("sim", "plots", "reachability_right.png")
    plot(rows, xs, ys, best, out_png)
    print(f"wrote {out_png.relative_to(config.REPO_ROOT)}")

    # Reachable bounding box at the chosen height, to place the zones inside.
    grid = best["grid"]
    iy, ix = np.nonzero(grid)
    if len(ix):
        print(
            f"\nreachable extent at h={best['shoulder_height']:.2f}: "
            f"x [{xs[ix.min()]:.2f}, {xs[ix.max()]:.2f}]  y [{ys[iy.min()]:+.2f}, {ys[iy.max()]:+.2f}]"
        )

    out_json = config.path("sim", "plots", "reachability_right.json")
    out_json.write_text(
        json.dumps(
            {
                "conventions_version": config.CONVENTIONS_VERSION,
                "side": args.side,
                "grasp_z": GRASP_Z,
                "yaws_deg": list(YAWS_DEG),
                "model_shoulder_height": H_MODEL,
                "grid": {"xs": xs.tolist(), "ys": ys.tolist()},
                "heights": [
                    {"shoulder_height": r["shoulder_height"], "n_cells": r["n_cells"]} for r in rows
                ],
                "best_shoulder_height": best["shoulder_height"],
                # The reachable cell map at the chosen height, row-major over
                # (ys, xs), so the zones in config/skills.yaml can be placed
                # inside it without re-running the sweep.
                "best_grid": best["grid"].astype(int).tolist(),
                "seed_k": args.seed_k,
                "n_restarts": args.n_restarts,
                "n_seeds_in_bank": len(bank),
            },
            indent=2,
        )
    )
    print(f"wrote {out_json.relative_to(config.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
