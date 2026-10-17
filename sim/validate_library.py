"""Score library candidates in simulation and drop the ones that fail (step 9).

Each candidate from `data/build_library.py` is replayed over `validation.n_trials`
randomly placed cubes, with success judged by the true object state
(`conventions.md` section 11: cube in the bin, below the rim, at rest after
`settle_s`). Entries reaching `validation.min_success_rate` are written back with
their `sim` block filled in; the rest are recorded as dropped, with their failure
reasons, so a low yield can be diagnosed without re-running.

**Every candidate sees the same scenes.** The cube poses are drawn once from
`validation.seed` and reused for all entries, so success rates are comparable and
a difference between two entries is the entry rather than the draw.

    uv run python sim/validate_library.py
    uv run python sim/validate_library.py --trials 8 --processes 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, demos  # noqa: E402
from common.ik import IKSolver  # noqa: E402
from data.build_library import library_path  # noqa: E402
from sim.execute import run_demo  # noqa: E402
from sim.sim_robot import SimRobot  # noqa: E402

_WORKER: dict = {}


def scene_draw(n_trials: int, seed: int) -> list[tuple[np.ndarray, float]]:
    """`n_trials` (cube_xy, cube_yaw) scenes from the pick zone.

    Yaw spans +/- 45 deg only: a cube's yaw is 90 deg-symmetric
    (`conventions.md` section 2), so that covers every distinct orientation and a
    wider range would just re-test the same geometry.
    """
    scene = config.skills()["scene"]
    centre = np.asarray(scene["pick_zone"]["center"], dtype=np.float64)
    size = np.asarray(scene["pick_zone"]["size"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return [
        (centre + rng.uniform(-0.5, 0.5, 2) * size, float(rng.uniform(-np.pi / 4, np.pi / 4)))
        for _ in range(n_trials)
    ]


def _init_worker() -> None:
    """One robot and one solver per process — both are expensive to build."""
    _WORKER["robot"] = SimRobot()
    _WORKER["solver"] = IKSolver()


def _reason_key(reason: str) -> str:
    """Collapse a failure reason to something countable."""
    if not reason:
        return "ok"
    if "outside the bin footprint" in reason:
        return "cube_not_in_bin"
    if "above the rim" in reason:
        return "cube_above_rim"
    if "still moving" in reason:
        return "cube_not_at_rest"
    return reason.split(" ")[0].split("(")[0]


def score_entry(entry: dict, scenes: list, side: str) -> dict:
    """Replay one candidate over every scene and count successes."""
    robot, solver = _WORKER["robot"], _WORKER["solver"]
    demo = demos.load_source_demo(config.path("data", "retargeted", f"{entry['id']}.npz"))

    n_ok = 0
    failures: Counter = Counter()
    for xy, yaw in scenes:
        res = run_demo(robot, demo, xy, yaw, side, solver)
        if res:
            n_ok += 1
        else:
            failures[_reason_key(res.reason)] += 1
    return {
        "id": entry["id"],
        "n_trials": len(scenes),
        "n_success": n_ok,
        "success_rate": n_ok / max(len(scenes), 1),
        "failures": dict(failures),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="egodex")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--side", default="right")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N candidates")
    args = ap.parse_args()

    vcfg = config.skills()["validation"]
    n_trials = int(vcfg["n_trials"]) if args.trials is None else args.trials
    n_proc = int(vcfg["n_processes"]) if args.processes is None else args.processes
    seed = int(vcfg["seed"])
    threshold = float(vcfg["min_success_rate"])

    src = library_path(args.source, args.version)
    if not src.exists():
        print(f"{src} is missing — run `uv run python data/build_library.py` first")
        return 1
    lib = json.loads(src.read_text())
    entries = lib["entries"][: args.limit]
    if not entries:
        print("no candidates to score")
        return 1

    scenes = scene_draw(n_trials, seed)
    print(
        f"scoring {len(entries)} candidates x {n_trials} scenes "
        f"on {n_proc} processes (seed {seed}, threshold {threshold})"
    )

    work = partial(score_entry, scenes=scenes, side=args.side)
    results = []
    with Pool(processes=n_proc, initializer=_init_worker) as pool:
        for i, row in enumerate(pool.imap_unordered(work, entries)):
            results.append(row)
            print(
                f"  {i + 1:3d}/{len(entries)} {row['id']:<16s} "
                f"{row['n_success']:2d}/{row['n_trials']} "
                f"({row['success_rate']:.2f}) {row['failures']}",
                flush=True,
            )

    by_id = {r["id"]: r for r in results}
    kept = []
    for entry in entries:
        row = by_id.get(entry["id"])
        if row is None:
            continue
        entry = dict(entry)
        entry["sim"] = {
            "n_trials": row["n_trials"],
            "success_rate": row["success_rate"],
            "seed": seed,
            "failures": row["failures"],
        }
        if row["success_rate"] >= threshold:
            kept.append(entry)

    rates = np.array([r["success_rate"] for r in results], dtype=np.float64)
    all_failures: Counter = Counter()
    for r in results:
        all_failures.update(r["failures"])

    print(f"\n=== {len(kept)}/{len(results)} candidates at success_rate >= {threshold} ===")
    print(
        f"  success_rate: mean {rates.mean():.2f}  median {np.median(rates):.2f}  "
        f"max {rates.max():.2f}"
    )
    for t in (1.0, 0.9, 0.8, 0.6, 0.4, 0.2):
        print(f"    >= {t:.1f}: {int((rates >= t).sum()):3d}")
    print("  failure reasons across all trials:")
    for reason, count in all_failures.most_common():
        print(f"    {reason:<22s} {count}")

    out = src.with_name(src.name.replace(".json", "_validated.json"))
    out.write_text(
        json.dumps(
            {
                "conventions_version": config.CONVENTIONS_VERSION,
                "source": args.source,
                "version": args.version,
                "orientation": lib.get("orientation"),
                "validated": True,
                "n_candidates": len(results),
                "n_kept": len(kept),
                "min_success_rate": threshold,
                "seed": seed,
                "scores": results,
                "entries": kept,
            },
            indent=2,
        )
    )
    print(f"wrote {out.relative_to(config.REPO_ROOT)}")

    # Handoff escalation rule: fewer than 20 surviving entries needs a human.
    return 0 if len(kept) >= 20 else 2


if __name__ == "__main__":
    raise SystemExit(main())
