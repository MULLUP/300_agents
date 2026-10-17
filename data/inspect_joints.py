"""Identify the thumb and index MCP joints in EgoDex's ARKit hand skeleton.

ARKit does not label MCPs, so we infer them from the chain geometry: the MCP is
the second joint out from the wrist along each finger, and the two MCPs should
sit at a comparable distance from the wrist (they are both knuckles on the same
hand). Prints the median wrist-to-joint distance per joint over a few episodes
and saves a labelled 3D scatter of one frame as evidence.

    uv run python data/inspect_joints.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.transforms import arkit_to_demo  # noqa: E402
from data.egodex_io import (  # noqa: E402
    HAND_JOINTS,
    TCP_JOINT_ROLES,
    episode_paths,
    load_episode,
)

RAW_DIR = Path(__file__).resolve().parent / "egodex_raw"
PLOT_DIR = Path(__file__).resolve().parent / "retargeted" / "plots"

# Which chain each joint belongs to, in order out from the wrist.
THUMB_CHAIN = ["ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"]
INDEX_CHAIN = [
    "IndexFingerMetacarpal",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
]


def chain_distances(episode: dict) -> dict[str, float]:
    """Median distance from the wrist to every other hand joint, in meters."""
    side = episode["arm"]
    wrist = episode["transforms"][f"{side}Hand"][:, :3, 3]
    return {
        j: float(np.median(np.linalg.norm(episode["transforms"][f"{side}{j}"][:, :3, 3] - wrist, axis=1)))
        for j in HAND_JOINTS
        if j != "Hand"
    }


def is_monotonic(distances: dict[str, float], chain: list[str]) -> bool:
    """True if the wrist distance grows along the chain (no joint out of order)."""
    d = [distances[j] for j in chain]
    return all(b > a for a, b in zip(d, d[1:]))


def plot_frame(episode: dict, frame: int, out_path: Path) -> None:
    """Labelled 3D scatter of one frame's hand joints in the `demo` frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    side = episode["arm"]
    pts, labels = {}, []
    for j in HAND_JOINTS:
        T = arkit_to_demo(episode["transforms"][f"{side}{j}"][frame])
        pts[j] = T[:3, 3]
        labels.append(j)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for chain, color, name in ((THUMB_CHAIN, "tab:orange", "thumb"), (INDEX_CHAIN, "tab:blue", "index")):
        path = np.array([pts["Hand"]] + [pts[j] for j in chain])
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "-o", color=color, label=name, ms=4)

    mcp_names = {TCP_JOINT_ROLES["thumb_mcp"], TCP_JOINT_ROLES["index_mcp"]}
    for j in labels:
        p = pts[j]
        is_mcp = j in mcp_names
        ax.scatter(*p, s=120 if is_mcp else 20, c="red" if is_mcp else "gray", depthshade=False)
        ax.text(*p, f"  {j}", fontsize=7, color="red" if is_mcp else "black")

    ax.scatter(*pts["Hand"], s=140, c="black", depthshade=False)
    ax.set_title(
        f"{Path(episode['source_path']).name} frame {frame} ({side} hand, demo frame)\n"
        f"red = inferred MCPs, black = wrist"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    _equal_aspect_3d(ax, np.array(list(pts.values())))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def _equal_aspect_3d(ax, pts: np.ndarray) -> None:
    """Equal axis scaling, so the hand is not visually distorted."""
    center = pts.mean(axis=0)
    radius = max(float(np.ptp(pts, axis=0).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--side", default="right")
    ap.add_argument("--plot", type=Path, default=PLOT_DIR / "joint_identification.png")
    args = ap.parse_args()

    paths = episode_paths(args.raw_dir)[: args.n_episodes]
    if not paths:
        print(f"no .hdf5 episodes in {args.raw_dir}", file=sys.stderr)
        return 1

    all_distances, ok = [], True
    for path in paths:
        ep = load_episode(path, args.side)
        dist = chain_distances(ep)
        all_distances.append(dist)

        mono_thumb = is_monotonic(dist, THUMB_CHAIN)
        mono_index = is_monotonic(dist, INDEX_CHAIN)
        ok = ok and mono_thumb and mono_index

        print(f"\n{path.name}  N={ep['N']}  ({ep['N'] / ep['fps']:.1f} s)")
        print(f"  {ep['llm_description']}")
        for name, chain, mono in (("thumb", THUMB_CHAIN, mono_thumb), ("index", INDEX_CHAIN, mono_index)):
            print(f"  {name} chain (monotonic from wrist: {mono}):")
            for j in chain:
                print(f"    wrist -> {j:<34} {dist[j] * 100:6.2f} cm")

    mean = {j: float(np.mean([d[j] for d in all_distances])) for j in all_distances[0]}
    print(f"\n=== mean over {len(paths)} episodes ===")
    print(f"{'joint':<36} {'wrist dist':>10}  {'chain position':>14}")
    for chain in (THUMB_CHAIN, INDEX_CHAIN):
        for i, j in enumerate(chain, start=1):
            print(f"{j:<36} {mean[j] * 100:8.2f} cm  {i:>10} from wrist")

    thumb_mcp, index_mcp = TCP_JOINT_ROLES["thumb_mcp"], TCP_JOINT_ROLES["index_mcp"]
    print("\n=== MCP inference ===")
    print(f"index MCP = {index_mcp} (2nd in the index chain, {mean[index_mcp] * 100:.2f} cm from wrist)")
    print(f"thumb MCP = {thumb_mcp} (2nd in the thumb chain, {mean[thumb_mcp] * 100:.2f} cm from wrist)")
    print(
        f"  the two MCPs differ by {abs(mean[thumb_mcp] - mean[index_mcp]) * 100:.2f} cm; "
        f"the first joints of each chain ({THUMB_CHAIN[0]}, {INDEX_CHAIN[0]}) sit at "
        f"{mean[THUMB_CHAIN[0]] * 100:.2f} / {mean[INDEX_CHAIN[0]] * 100:.2f} cm, i.e. at the "
        f"hand base rather than the knuckle line"
    )
    print(f"\nall chains monotonic from the wrist: {ok}")

    plot_frame(load_episode(paths[0], args.side), frame=0, out_path=args.plot)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
