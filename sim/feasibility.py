"""IK feasibility of the retargeted demos at a nominal cube pose (plan task 2.P2.3).

A source demo that passed the step 4 quality filters is a *human* trajectory: it
can still be impossible for this arm, because the arm's usable top-down workspace
is only ~25 cm across (`AGENT_HANDOFF.md` section 11a) while the kept demos span
7-47 cm of transport. This screen measures how many survive geometry, before any
physics runs, so step 7 only builds library candidates out of demos that can move.

No simulation stepping and no rendering: it is pure IK, so it is cheap enough to
run over every demo.

    uv run python sim/feasibility.py
    uv run python sim/feasibility.py --orientation replay   # the P8 comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, demos  # noqa: E402
from common.ik import IKSolver  # noqa: E402
from common.transforms import top_down_pose  # noqa: E402
from sim.execute import (  # noqa: E402
    bin_place_target,
    plan_from_demo,
    screen_pose_path,
    yaw_locked_mask,
)


def nominal_cube_pose() -> np.ndarray:
    """The pick-zone centre at cube-centre height, at `library.nominal_cube_yaw`.

    One fixed scene, so the feasible count is comparable between runs and between
    orientation modes. Step 9 is what samples the zone.
    """
    scene = config.skills()["scene"]
    centre = np.asarray(scene["pick_zone"]["center"], dtype=np.float64)
    half = float(scene["cube"]["size"]) / 2.0
    yaw = float(config.skills()["library"]["nominal_cube_yaw"])
    return top_down_pose([centre[0], centre[1], half], yaw)


def scene_index(T_table_cube: np.ndarray) -> dict[str, float]:
    """What `(dist_xy, dz)` this scene asks of a source demo (section 7.3).

    The retrieval index is defined on the demo's own grasp -> release motion, so
    the scene's cube -> place-target motion is the query to match it against.
    """
    T_tgt = bin_place_target()
    delta = T_tgt[:3, 3] - np.asarray(T_table_cube, dtype=np.float64)[:3, 3]
    return {"dist_xy": float(np.linalg.norm(delta[:2])), "dz": float(delta[2])}


def index_offset(demo_index: dict[str, float], query: dict[str, float]) -> float:
    """Euclidean distance in the `(dist_xy, dz)` index, metres.

    This *is* the distance between the cube and where the transport segment
    starts: both coordinates enter the section 8 step 4 transform linearly.
    """
    return float(
        np.hypot(query["dist_xy"] - demo_index["dist_xy"], query["dz"] - demo_index["dz"])
    )


def screen_demo(
    solver: IKSolver, path: Path, T_table_cube: np.ndarray, side: str, orientation: str
) -> dict:
    """Retarget one demo onto `T_table_cube` and report whether it is executable.

    The `(dist_xy, dz)` pre-screen runs first because it is free and explains
    most rejections on its own; IK only runs on demos that survive it.
    """
    demo = demos.load_source_demo(path, orientation)
    meta = demos.load_metadata(path)
    lib = config.skills()["library"]
    rate_hz = float(config.skills()["exec"]["rate_hz"])

    offset = index_offset(demo.index, scene_index(T_table_cube))
    row = {
        "id": meta["id"],
        "source_path": meta["source_path"],
        "index": demo.index,
        "index_offset": offset,
    }

    max_offset = float(lib["max_index_offset"])
    if offset > max_offset:
        return {
            **row,
            "feasible": False,
            "reason": f"index_mismatch (offset {offset:.3f} > {max_offset})",
            "category": "index_mismatch",
        }

    plan = plan_from_demo(demo, T_table_cube, side, solver)
    # Yaw is only constrained where the jaws close; everywhere else the solver
    # picks it. See `sim/execute.py yaw_locked_mask`.
    ok, why = screen_pose_path(
        solver, plan.pose_path, side, rate_hz, free_yaw=~yaw_locked_mask(plan.grip_path)
    )

    z = plan.pose_path[:, 2, 3]
    return {
        **row,
        "feasible": bool(ok),
        "reason": why,
        # `reason` carries measurements inline; `category` is the groupable part.
        "category": "ok" if ok else why.split(" ", 1)[0],
        "n_samples": int(len(plan.pose_path)),
        "heading_delta_deg": float(np.degrees(plan.heading_delta)),
        "release_yaw_deg": float(np.degrees(plan.release_yaw)),
        "tcp_z_min": float(z.min()),
        "tcp_z_max": float(z.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="egodex")
    ap.add_argument("--side", default="right")
    ap.add_argument(
        "--orientation",
        default=None,
        choices=demos.ORIENTATION_MODES,
        help="default: config/skills.yaml library.orientation",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    orientation = demos.orientation_mode() if args.orientation is None else args.orientation
    out = args.out or config.path("sim", "plots", f"feasibility_{args.source}_{orientation}.json")

    paths = demos.demo_paths(args.source)[: args.limit]
    if not paths:
        print(f"no {args.source}_*.npz in {demos.demo_dir()}; run data/retarget.py first")
        return 1

    T_cube = nominal_cube_pose()
    solver = IKSolver()
    print(
        f"screening {len(paths)} {args.source} demos, orientation={orientation}, "
        f"cube at ({T_cube[0, 3]:.3f},{T_cube[1, 3]:+.3f})"
    )

    rows = []
    for i, p in enumerate(paths):
        row = screen_demo(solver, p, T_cube, args.side, orientation)
        rows.append(row)
        print(
            f"  {i + 1:3d}/{len(paths)} {row['id']:<14s} "
            f"dist_xy={row['index']['dist_xy']:.3f} dz={row['index']['dz']:+.3f} "
            f"off={row['index_offset']:.3f} "
            f"-> {'OK  ' if row['feasible'] else 'FAIL'} {row['reason'][:60]}",
            flush=True,
        )

    n_ok = sum(r["feasible"] for r in rows)
    reasons = Counter(r["category"] for r in rows)
    print(f"\n=== feasible {n_ok}/{len(rows)} ({100.0 * n_ok / len(rows):.1f}%) ===")
    for reason, count in reasons.most_common():
        print(f"  {reason:<22s} {count}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "conventions_version": config.CONVENTIONS_VERSION,
                "source": args.source,
                "side": args.side,
                "orientation": orientation,
                "nominal_cube_pose": T_cube.tolist(),
                "n_demos": len(rows),
                "n_feasible": n_ok,
                "reasons": dict(reasons),
                "demos": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {out.relative_to(config.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
