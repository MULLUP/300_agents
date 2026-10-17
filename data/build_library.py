"""Turn feasible source demos into library candidates (plan task 4.P2.2, step 7).

Writes `data/library/{source}_v{n}.json` in the `conventions.md` section 7.3
schema, with `sim` left null — `sim/validate_library.py` is what fills it in and
drops the entries that do not hold up. So this script is deliberately dumb: it
carries provenance, segments, anchors and the retrieval index across, and makes
no judgement beyond the feasibility screen's.

    uv run python data/build_library.py
    uv run python data/build_library.py --all   # skip the feasibility filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, demos  # noqa: E402
from common.mimic import demo_anchors  # noqa: E402
from common.transforms import pose_to_dict  # noqa: E402


def library_path(source: str, version: int, left: bool = False) -> Path:
    """`data/library/{source}[_left]_v{n}.json` (conventions.md section 7.3)."""
    suffix = "_left" if left else ""
    return config.path("data", "library", f"{source}{suffix}_v{version}.json")


def feasible_ids(source: str, orientation: str) -> set[str] | None:
    """Ids the step 6 screen passed, or None if it has not been run.

    Keyed off `sim/plots/feasibility_*.json` rather than re-screening, so the
    library is always built from a recorded measurement that can be inspected.
    """
    path = config.path("sim", "plots", f"feasibility_{source}_{orientation}.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {r["id"] for r in data["demos"] if r["feasible"]}


def entry_for(path: Path) -> dict:
    """One section 7.3 entry for a source demo, with `sim` unfilled."""
    demo = demos.load_source_demo(path)
    meta = demos.load_metadata(path)
    T_grasp, T_release = demo_anchors(demo)

    n = len(demo.T_demo_tcp)
    return {
        "id": meta["id"],
        "source": meta["source"],
        "source_path": meta["source_path"],
        "arm": demo.arm,
        "mirrored_from": None,
        "anchor_type": demo.anchor_type,
        # Section 7.3's segments are indices into the SOURCE demo, not into any
        # retargeted path: the transform is re-run per scene, so storing output
        # indices would be meaningless.
        "segments": {
            "approach": [0, demo.grasp_idx],
            "transport": [demo.grasp_idx, demo.release_idx],
        },
        "anchors": {
            "T_demo_grasp": pose_to_dict(T_grasp),
            "T_demo_release": pose_to_dict(T_release),
        },
        "index": demo.index,
        "n_samples": n,
        "conf_min": meta["conf_min"],
        "sim": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="egodex")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument(
        "--all",
        action="store_true",
        help="include demos the step 6 screen rejected (for measuring the screen)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    orientation = demos.orientation_mode()
    keep = None if args.all else feasible_ids(args.source, orientation)
    if keep is None and not args.all:
        print(
            f"no feasibility screen for {args.source}/{orientation} — "
            "run `uv run python sim/feasibility.py` first, or pass --all"
        )
        return 1

    paths = demos.demo_paths(args.source)
    if not paths:
        print(f"no {args.source}_*.npz in {demos.demo_dir()}")
        return 1

    entries, skipped = [], 0
    for p in paths:
        if keep is not None and demos.demo_id(p) not in keep:
            skipped += 1
            continue
        entries.append(entry_for(p))

    out = args.out or library_path(args.source, args.version)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "conventions_version": config.CONVENTIONS_VERSION,
                "source": args.source,
                "version": args.version,
                "orientation": orientation,
                "validated": False,
                "entries": entries,
            },
            indent=2,
        )
    )
    print(f"{len(entries)} candidates ({skipped} skipped by the step 6 screen)")
    print(f"wrote {out.relative_to(config.REPO_ROOT)}")
    print("next: uv run python sim/validate_library.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
