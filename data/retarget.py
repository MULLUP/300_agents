"""EgoDex episodes -> source demos in the `demo` frame (plan tasks 2.P2.1, 2.P2.2).

Per episode: load -> `arkit` to `demo` -> HumanEgo hand-to-`tcp` -> smooth ->
resample to 50 Hz -> detect the grasp/release pair -> apply the quality filters
-> write `data/retargeted/egodex_<idx>.npz` (conventions.md section 7.2).

    uv run python data/retarget.py                 # all episodes
    uv run python data/retarget.py --n-plots 5     # with evidence plots
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.segment import Segmentation, segment, smooth_positions  # noqa: E402
from common.transforms import (  # noqa: E402
    arkit_to_demo,
    gripper_frame,
    make_T,
    orthonormalize,
    resample_poses,
)
from data.egodex_io import (  # noqa: E402
    episode_index,
    episode_paths,
    load_episode,
    tcp_conf_min,
    tcp_conf_track,
    tcp_inputs,
)

OUT_DIR = config.path("data", "retargeted")


def retarget_episode(path: Path, cfg: dict, side: str = "right") -> dict:
    """One episode -> the `demo`-frame trajectory plus its segmentation.

    Returns a dict with the resampled ``t``, ``T_demo_tcp``, ``aperture``, the
    :class:`Segmentation`, and provenance. Does not write anything.
    """
    ep = load_episode(path, side)

    # conventions.md section 2: T_demo_x = T_zup @ T_arkit_x, applied to the full
    # 4x4 so the hand's orientation comes along with its position.
    ep["transforms"] = {k: arkit_to_demo(v) for k, v in ep["transforms"].items()}
    pts = tcp_inputs(ep)

    T_tcp, aperture = gripper_frame(
        pts["wrist"], pts["thumb_mcp"], pts["index_mcp"], pts["thumb_tip"], pts["index_tip"]
    )

    # Smooth at the native 30 Hz, before resampling. Averaging rotation matrices
    # elementwise leaves them slightly non-orthogonal, so project back via SVD.
    window = int(cfg["smooth_window"])
    pos_s = smooth_positions(T_tcp[:, :3, 3], window)
    R_s = orthonormalize(smooth_positions(T_tcp[:, :3, :3], window))
    T_smooth = make_T(R_s, pos_s)
    aperture_s = smooth_positions(aperture, window)

    # Resample by time to the output rate: linear position, slerp rotation.
    fps_out = float(cfg["fps_out"])
    t_src = ep["t"]
    n_out = max(int(np.floor(t_src[-1] * fps_out)) + 1, 2)
    t_out = np.arange(n_out, dtype=np.float64) / fps_out
    T_out = resample_poses(t_src, T_smooth, t_out)
    aperture_out = np.interp(t_out, t_src, aperture_s)

    # Per-frame confidence, resampled alongside the poses so the segmenter can
    # judge it over only the frames we keep.
    conf_track = tcp_conf_track(ep)
    conf_out = float("nan") if conf_track is None else np.interp(t_out, t_src, conf_track)

    seg = segment(T_out, aperture_out, t_out, cfg, conf_out)

    return {
        "idx": episode_index(path),
        "t": t_out,
        "T_demo_tcp": T_out,
        "aperture": aperture_out,
        "seg": seg,
        "conf_min": tcp_conf_min(ep),
        "llm_description": ep["llm_description"],
        "source_path": str(path),
        "arm": side,
        "N_raw": ep["N"],
    }


def save_demo(demo: dict, out_dir: Path) -> Path:
    """Write one kept demo as `data/retargeted/egodex_<idx>.npz` (section 7.2)."""
    seg: Segmentation = demo["seg"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"egodex_{demo['idx']}.npz"
    np.savez_compressed(
        out_path,
        t=demo["t"],
        T_demo_tcp=demo["T_demo_tcp"],
        aperture=demo["aperture"],
        grip=seg.grip,
        grasp_idx=seg.grasp_idx,
        release_idx=seg.release_idx,
        conf_min=demo["conf_min"],
        source="egodex",
        source_path=demo["source_path"],
        arm=demo["arm"],
        llm_description=demo["llm_description"],
        conventions_version=config.CONVENTIONS_VERSION,
    )
    return out_path


def plot_demo(demo: dict, out_dir: Path) -> Path:
    """Aperture over time with grasp/release markers, plus the 3D gripper path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seg: Segmentation = demo["seg"]
    t, pos = demo["t"], demo["T_demo_tcp"][:, :3, 3]

    fig = plt.figure(figsize=(13, 5))

    ax = fig.add_subplot(121)
    ax.plot(t, demo["aperture"] * 100, lw=1.5, label="aperture")
    ax.axhline(seg.thr * 100, color="gray", ls="--", lw=1, label=f"thr = {seg.thr * 100:.1f} cm")
    ax.axhline(seg.a_open * 100, color="lightgray", ls=":", lw=1, label="p90 / p10")
    ax.axhline(seg.a_min * 100, color="lightgray", ls=":", lw=1)
    ax.axvline(t[seg.grasp_idx], color="tab:green", lw=2, label=f"grasp @ {t[seg.grasp_idx]:.2f} s")
    ax.axvline(t[seg.release_idx], color="tab:red", lw=2, label=f"release @ {t[seg.release_idx]:.2f} s")
    ax.axvspan(t[seg.grasp_idx], t[seg.release_idx], color="tab:green", alpha=0.08)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("aperture [cm]")
    ax.set_title(f"episode {demo['idx']}: grip closed between the markers")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    ax3 = fig.add_subplot(122, projection="3d")
    closed = slice(seg.grasp_idx, seg.release_idx)
    ax3.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="lightsteelblue", lw=1, label="open")
    ax3.plot(pos[closed, 0], pos[closed, 1], pos[closed, 2], color="tab:green", lw=2.5, label="carrying")
    ax3.scatter(*pos[seg.grasp_idx], c="tab:green", s=70, label="grasp", depthshade=False)
    ax3.scatter(*pos[seg.release_idx], c="tab:red", s=70, label="release", depthshade=False)
    ax3.scatter(*pos[0], c="black", s=40, marker="^", label="start", depthshade=False)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    ax3.set_title(
        f"tcp path, demo frame\ndist_xy = {seg.metrics['dist_xy'] * 100:.1f} cm, "
        f"dz = {seg.metrics['dz'] * 100:+.1f} cm"
    )
    ax3.legend(fontsize=7)
    _equal_aspect_3d(ax3, pos)

    fig.suptitle(demo["llm_description"][:110], fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"egodex_{demo['idx']}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _equal_aspect_3d(ax, pts: np.ndarray) -> None:
    center = pts.mean(axis=0)
    radius = max(float(np.ptp(pts, axis=0).max()) / 2.0, 1e-3)
    for lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        lim(c - radius, c + radius)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=config.path("data", "egodex_raw"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--side", default="right")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N episodes")
    ap.add_argument("--n-plots", type=int, default=5, help="evidence plots for the first N kept")
    args = ap.parse_args()

    cfg = config.skills()["retarget"]
    paths = episode_paths(args.raw_dir)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no episodes in {args.raw_dir}", file=sys.stderr)
        return 1

    reasons: Counter[str] = Counter()
    kept, errors, missing_conf, plotted = [], [], 0, 0

    for path in paths:
        try:
            demo = retarget_episode(path, cfg, args.side)
        except Exception as exc:  # a malformed episode must not stop the batch
            errors.append({"source_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            reasons["load_error"] += 1
            continue

        if np.isnan(demo["conf_min"]):
            missing_conf += 1

        seg: Segmentation = demo["seg"]
        if not seg.kept:
            reasons[seg.reject] += 1
            continue

        reasons["kept"] += 1
        save_demo(demo, args.out_dir)
        kept.append(
            {
                "idx": demo["idx"],
                "source_path": demo["source_path"],
                "grasp_idx": seg.grasp_idx,
                "release_idx": seg.release_idx,
                "n_frames": len(demo["t"]),
                "conf_min": None if np.isnan(demo["conf_min"]) else demo["conf_min"],
                "llm_description": demo["llm_description"],
                **seg.metrics,
            }
        )
        if plotted < args.n_plots:
            plot_demo(demo, args.out_dir / "plots")
            plotted += 1

    n_kept = len(kept)
    summary = {
        "conventions_version": config.CONVENTIONS_VERSION,
        "source": "egodex",
        "arm": args.side,
        "raw_dir": str(args.raw_dir),
        "n_episodes": len(paths),
        "n_kept": n_kept,
        "keep_rate": n_kept / len(paths),
        "reasons": dict(reasons.most_common()),
        "n_missing_confidences": missing_conf,
        "config_retarget": cfg,
        "errors": errors,
        "kept": kept,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== retargeting: {n_kept}/{len(paths)} kept ({n_kept / len(paths):.1%}) ===")
    for reason, count in reasons.most_common():
        print(f"  {reason:<32} {count:4d}")
    print(f"  (episodes missing confidences: {missing_conf})")
    if kept:
        dist = np.array([k["dist_xy"] for k in kept])
        dz = np.array([k["dz"] for k in kept])
        dur = np.array([k["duration_s"] for k in kept])
        print(f"\n  dist_xy    {dist.min() * 100:5.1f} - {dist.max() * 100:5.1f} cm (median {np.median(dist) * 100:.1f})")
        print(f"  dz         {dz.min() * 100:+5.1f} - {dz.max() * 100:+5.1f} cm (median {np.median(dz) * 100:+.1f})")
        print(f"  duration   {dur.min():5.1f} - {dur.max():5.1f} s (median {np.median(dur):.1f})")
    print(f"\nwrote {n_kept} .npz + summary.json to {args.out_dir}")
    print(f"plots: {plotted} in {args.out_dir / 'plots'}")

    # Gate G2: at least 100 kept, escalate below 40.
    if n_kept < 40:
        print(f"\nESCALATE: only {n_kept} demos kept, below the 40 minimum (gate G2)", file=sys.stderr)
        return 1
    if n_kept < 100:
        print(f"\nWARNING: {n_kept} demos kept, below the gate G2 target of 100", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
