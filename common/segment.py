"""Grasp/release detection and quality filters for source demos (handoff step 4).

EgoDex gives us no gripper command, so the grasp and release have to be read off
the hand's own aperture signal. Both are pure functions of the resampled
trajectory, so they are testable without touching HDF5 or MuJoCo.

**The aperture is not monotonic in grasp state.** A *relaxed* human hand sits at
a small aperture, so the signal over an episode is

    relaxed (small) -> open to grasp (large) -> close on object (small)
    -> carry (small) -> open to release (large) -> relaxed (small)

Measured over all 277 test-split episodes, 79% start below the closed threshold
and 46% end below it. A naive "first sustained close" detector therefore fires on
frame 0, and the post-release relax looks like a second grasp. We handle this by
discarding below-threshold runs that touch either end of the episode; see
`docs/conventions_proposals.md` (P1) for the full write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segmentation:
    """Where the grasp and release happen, and whether the demo is usable."""

    grasp_idx: int | None
    release_idx: int | None
    grip: np.ndarray | None  # (N,) 1 = open, 0 = closed
    thr: float
    a_open: float
    a_min: float
    n_candidates: int  # flanked closed runs found; we require exactly one
    reject: str | None = None  # None = kept, otherwise the first failing filter
    metrics: dict = field(default_factory=dict)

    @property
    def kept(self) -> bool:
        return self.reject is None


def closed_runs(closed: np.ndarray, hold: int) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` runs of ``closed`` lasting at least ``hold`` frames.

    Runs shorter than ``hold`` are dropped, which rejects aperture flicker from
    tracking noise.
    """
    padded = np.concatenate(([False], closed, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2]) if b - a >= hold]


def flanked_runs(closed: np.ndarray, hold: int) -> list[tuple[int, int]]:
    """Closed runs with an open frame on *both* sides.

    A run touching frame 0 or the final frame cannot be distinguished from a
    relaxed hand (see the module docstring), so it is not a grasp candidate.
    """
    n = len(closed)
    return [(s, e) for s, e in closed_runs(closed, hold) if s > 0 and e < n]


def segment(
    T_tcp: np.ndarray,
    aperture: np.ndarray,
    t: np.ndarray,
    cfg: dict,
    conf: np.ndarray | float = float("nan"),
) -> Segmentation:
    """Detect the grasp/release pair and apply the step 4 filters.

    ``T_tcp`` ``(N,4,4)`` and ``aperture`` ``(N,)`` must already be resampled to
    the output rate, because ``hold_frames`` and the speed test are expressed in
    output frames. ``cfg`` is ``config/skills.yaml``'s ``retarget`` block.
    ``conf`` is the per-frame minimum confidence over the five tcp joints, as an
    ``(N,)`` array, or ``nan`` when the episode has no confidences.
    """
    g, fil = cfg["grasp"], cfg["filters"]
    n = len(aperture)

    a_open = float(np.percentile(aperture, g["open_percentile"]))
    a_min = float(np.percentile(aperture, g["min_percentile"]))
    thr = a_min + g["thr_fraction"] * (a_open - a_min)
    hold = int(g["hold_frames"])

    pos = T_tcp[:, :3, 3]
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = np.concatenate(([0.0], step)) / dt
    max_jump = float(step.max()) if n > 1 else 0.0

    runs = flanked_runs(aperture < thr, hold)
    n_candidates = len(runs)

    def fail(reason: str, **extra) -> Segmentation:
        return Segmentation(
            None, None, None, thr, a_open, a_min, n_candidates,
            reject=reason,
            metrics={"max_frame_jump": max_jump, "aperture_range": a_open - a_min, **extra},
        )

    # A usable demo closes the hand exactly once: one grasp, one release.
    if n_candidates != 1:
        return fail("not_one_grasp_release_pair", n_candidates=n_candidates)

    grasp_idx, release_idx = runs[0]

    # The hand must be nearly still at the grasp; a fast-moving closed frame is a
    # pass-through, not a pick.
    if speed[grasp_idx] >= g["max_speed"]:
        return fail("grasp_speed_too_high", grasp_speed=float(speed[grasp_idx]))

    # Both events need room at the ends, so the approach and transport segments
    # are non-degenerate.
    edge = int(round(fil["edge_margin_s"] / dt))
    if grasp_idx < edge or release_idx > n - 1 - edge:
        return fail("event_too_close_to_end", grasp_idx=grasp_idx, release_idx=release_idx, edge=edge)

    dist_xy = float(np.linalg.norm(pos[release_idx, :2] - pos[grasp_idx, :2]))
    if dist_xy < fil["min_transport_xy"]:
        return fail("transport_too_short", dist_xy=dist_xy)

    # A single large jump means the hand tracking lost and reacquired the hand.
    if max_jump > fil["max_frame_jump"]:
        return fail("position_jump")

    # Confidence is judged only over the frames the library actually uses --
    # [0, release + tail] -- not the whole episode. Tracking often degrades after
    # the hand retracts, and we discard those frames anyway (handoff step 7).
    conf_min = _used_conf_min(conf, n, release_idx, dt, fil.get("conf_tail_s", 0.5))
    if not np.isnan(conf_min) and conf_min < fil["min_confidence"]:
        return fail("low_confidence", conf_min=float(conf_min))

    # conventions.md section 7.2: grip is 1 (open) outside [grasp, release).
    grip = np.ones(n, dtype=np.float64)
    grip[grasp_idx:release_idx] = 0.0

    return Segmentation(
        grasp_idx=int(grasp_idx),
        release_idx=int(release_idx),
        grip=grip,
        thr=thr,
        a_open=a_open,
        a_min=a_min,
        n_candidates=n_candidates,
        metrics={
            "dist_xy": dist_xy,
            "dz": float(pos[release_idx, 2] - pos[grasp_idx, 2]),
            "max_frame_jump": max_jump,
            "aperture_range": a_open - a_min,
            "grasp_speed": float(speed[grasp_idx]),
            "duration_s": float(t[-1] - t[0]),
            "conf_min_used": None if np.isnan(conf_min) else float(conf_min),
        },
    )


def _used_conf_min(conf: np.ndarray | float, n: int, release_idx: int, dt: float, tail_s: float) -> float:
    """Minimum confidence over the frames the library keeps, or ``nan``."""
    if np.isscalar(conf) or conf is None:
        return float(conf) if np.isscalar(conf) else float("nan")
    conf = np.asarray(conf, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    end = min(n, release_idx + int(round(tail_s / dt)) + 1)
    return float(np.min(conf[:end]))


def smooth_positions(pos: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over ``(N,...)``, with the ends held.

    ``np.convolve`` would shrink or taper the ends; edge-padding keeps the first
    and last poses where the demo actually started and finished.
    """
    if window <= 1 or len(pos) < 2:
        return pos.astype(np.float64, copy=True)
    half = window // 2
    pad = [(half, half)] + [(0, 0)] * (pos.ndim - 1)
    padded = np.pad(pos.astype(np.float64), pad, mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 0, padded)
