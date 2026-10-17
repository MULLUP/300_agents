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

**One episode can hold several demos.** A human who re-grips, or who picks up
three objects in a row, closes the hand more than once, and each closure is a
usable pick-and-place motion. :func:`segment_all` returns one
:class:`Segmentation` per closure, each judged on its own slice of the episode —
`[previous release, this release + conf_tail_s]`. Requiring *exactly one* closure
instead rejected 572 of 1077 episodes, over half the dataset, and threw away 142
motions that pass every quality filter; see `docs/conventions_proposals.md` P12.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segmentation:
    """One candidate demo: where its grasp and release are, and whether it is usable.

    ``grasp_idx``, ``release_idx`` and ``grip`` are relative to ``span``, the
    half-open slice of the episode this candidate owns. ``thr``, ``a_open`` and
    ``a_min`` are episode-level — the aperture's open/closed range can only be
    read off the whole episode.
    """

    grasp_idx: int | None
    release_idx: int | None
    grip: np.ndarray | None  # (span length,) 1 = open, 0 = closed
    thr: float
    a_open: float
    a_min: float
    n_candidates: int  # flanked closed runs in the episode
    candidate: int = 0  # which of them this is, in time order
    span: tuple[int, int] | None = None  # half-open slice of the episode
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


def descent_window_end(grasp_idx: int, release_idx: int, window_s: float, rate_hz: float) -> int:
    """Exclusive end of the window in which the descent onto the object bottoms out.

    The aperture crosses its threshold mid-descent, so the hand keeps going down
    after ``grasp_idx`` and the object is really at the bottom of that descent.
    The window has to cover the descent and **not** the carry — widening it picks
    up dips that belong to the carry and anchors the grasp below the object
    (`docs/conventions_proposals.md` P9).

    Shared by :func:`descent_bottom` here and
    :func:`common.demos.with_grasp_anchor_z`, so the speed filter and the anchor
    height cannot disagree about where the grasp is.
    """
    frames = max(int(round(float(window_s) * float(rate_hz))), 1)
    return min(grasp_idx + frames, release_idx) + 1


def descent_bottom(z: np.ndarray, grasp_idx: int, end: int) -> int:
    """Frame of the lowest point of the descent, in ``[grasp_idx, end)``.

    This, not ``grasp_idx``, is where the hand has the object: the aperture
    crossing is mid-descent (P9), so ``grasp_idx`` lands on a hand that is still
    falling. ``end`` comes from :func:`descent_window_end`.
    """
    return grasp_idx + int(np.argmin(z[grasp_idx:end]))


def segment_all(
    T_tcp: np.ndarray,
    aperture: np.ndarray,
    t: np.ndarray,
    cfg: dict,
    conf: np.ndarray | float = float("nan"),
    *,
    descent_window_s: float,
) -> list[Segmentation]:
    """Every candidate demo in one episode, each with the step 4 filters applied.

    ``T_tcp`` ``(N,4,4)`` and ``aperture`` ``(N,)`` must already be resampled to
    the output rate, because ``hold_frames`` and the speed test are expressed in
    output frames. ``cfg`` is ``config/skills.yaml``'s ``retarget`` block.
    ``conf`` is the per-frame minimum confidence over the five tcp joints, as an
    ``(N,)`` array, or ``nan`` when the episode has no confidences.
    ``descent_window_s`` is ``library.grasp_window_s`` — the same quantity P9
    uses for the anchor height, passed in rather than duplicated in this section
    of the config.

    Returns one :class:`Segmentation` per flanked closed run, in time order, and
    a single rejected one when the episode has no candidates at all. Callers
    should keep every entry whose :attr:`Segmentation.kept` is true.
    """
    g, fil = cfg["grasp"], cfg["filters"]
    n = len(aperture)

    a_open = float(np.percentile(aperture, g["open_percentile"]))
    a_min = float(np.percentile(aperture, g["min_percentile"]))
    thr = a_min + g["thr_fraction"] * (a_open - a_min)

    runs = flanked_runs(aperture < thr, int(g["hold_frames"]))
    if not runs:
        return [
            Segmentation(
                None, None, None, thr, a_open, a_min, 0,
                reject="no_grasp_release_pair",
                metrics={"aperture_range": a_open - a_min},
            )
        ]

    pos = T_tcp[:, :3, 3]
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = np.concatenate(([0.0], step)) / dt
    # Horizontal speed, which is what the grasp filter tests -- see `evaluate`.
    speed_xy = np.concatenate(([0.0], np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))) / dt
    rate_hz = 1.0 / dt
    # Frames of post-release tail each candidate carries: enough for the
    # confidence judgement, and the boundary to the next candidate's lead-in.
    tail = int(round(fil.get("conf_tail_s", 0.5) / dt))
    edge = int(round(fil["edge_margin_s"] / dt))

    def evaluate(k: int, grasp: int, release: int) -> Segmentation:
        """Apply the filters to candidate ``k``. Indices in absolute, out relative."""
        # This candidate owns everything since the previous release, plus a tail
        # past its own: the lead-in is the approach, and the tail is what the
        # confidence judgement needs.
        s, e = (runs[k - 1][1] if k else 0), min(n, release + tail + 1)
        m = e - s
        gi, ri = grasp - s, release - s

        # A single large jump means the hand tracking lost and reacquired the
        # hand. Judged over this candidate's own frames: a glitch elsewhere in
        # the episode says nothing about this motion.
        max_jump = float(step[s : e - 1].max()) if m > 1 else 0.0

        # What this filter is for is rejecting a pass-through: a hand that sweeps
        # across the table without stopping to pick anything up. So it tests the
        # HORIZONTAL speed at the BOTTOM of the descent.
        #
        # Both halves of that matter. The aperture crossing is mid-descent (P9),
        # where the hand is still falling at a median 0.204 m/s, so testing there
        # rejected 149 episodes for descending briskly. And at the bottom the
        # residual vertical speed still exceeds the horizontal in 39% of
        # candidates, so a 3D speed there is partly the descent too — which P9
        # already established is normal and not disqualifying.
        window = descent_window_end(grasp, release, descent_window_s, rate_hz)
        bottom = descent_bottom(pos[:, 2], grasp, window)
        grasp_speed = float(speed_xy[bottom])

        dist_xy = float(np.linalg.norm(pos[release, :2] - pos[grasp, :2]))
        conf_min = _span_conf_min(conf, (s, e))

        def result(reason: str | None) -> Segmentation:
            grip = None
            if reason is None:
                # conventions.md section 7.2: grip is 1 (open) outside [grasp, release).
                grip = np.ones(m, dtype=np.float64)
                grip[gi:ri] = 0.0
            return Segmentation(
                grasp_idx=gi if reason is None else None,
                release_idx=ri if reason is None else None,
                grip=grip,
                thr=thr,
                a_open=a_open,
                a_min=a_min,
                n_candidates=len(runs),
                candidate=k,
                span=(s, e),
                reject=reason,
                metrics={
                    "dist_xy": dist_xy,
                    "dz": float(pos[release, 2] - pos[grasp, 2]),
                    "max_frame_jump": max_jump,
                    "aperture_range": a_open - a_min,
                    "grasp_speed": grasp_speed,  # horizontal, at the descent bottom
                    # What the filter used to test, kept so the two are comparable.
                    "grasp_speed_3d_at_crossing": float(speed[grasp]),
                    "descent_after_crossing": float(pos[grasp, 2] - pos[bottom, 2]),
                    "duration_s": float((m - 1) * dt),
                    "conf_min_used": None if np.isnan(conf_min) else float(conf_min),
                },
            )

        if grasp_speed >= g["max_speed"]:
            return result("grasp_speed_too_high")
        # Both events need room at the ends, so the approach and transport
        # segments are non-degenerate.
        if gi < edge or ri > m - 1 - edge:
            return result("event_too_close_to_end")
        if dist_xy < fil["min_transport_xy"]:
            return result("transport_too_short")
        if max_jump > fil["max_frame_jump"]:
            return result("position_jump")
        # Confidence is judged only over the frames this candidate owns.
        # Tracking often degrades after the hand retracts, and the next
        # candidate's frames are somebody else's problem.
        if not np.isnan(conf_min) and conf_min < fil["min_confidence"]:
            return result("low_confidence")
        return result(None)

    return [evaluate(k, grasp, release) for k, (grasp, release) in enumerate(runs)]


def _span_conf_min(conf: np.ndarray | float, span: tuple[int, int]) -> float:
    """Minimum confidence over the frames one candidate owns, or ``nan``."""
    if conf is None or np.isscalar(conf):
        return float("nan") if conf is None else float(conf)
    conf = np.asarray(conf, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    return float(np.min(conf[span[0] : span[1]]))


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
