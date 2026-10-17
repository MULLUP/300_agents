"""Tests for grasp/release detection and the step 4 filters (`common/segment.py`).

Same principle as `tests/test_demo_plan.py`: the episodes here are **synthetic**,
built so that what the detector should find is known by construction. Real EgoDex
rejection counts cannot distinguish a detector bug from genuine human behaviour —
two of the three findings this file pins down (P1's relaxed-hand artifact and
P9/section 4b's mid-descent grasp frame) were originally mistaken for the latter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.segment import (  # noqa: E402
    closed_runs,
    descent_bottom,
    descent_window_end,
    flanked_runs,
    segment_all,
    smooth_positions,
)
from data.retarget import demo_from_candidate  # noqa: E402

FPS = 50.0
OPEN, CLOSED = 0.09, 0.01  # aperture, metres; thr lands at ~0.042 between them
WINDOW_S = 0.3  # library.grasp_window_s

# Pinned here rather than read from config/skills.yaml: these tests fix
# behaviour, and a threshold change should not be able to silently rewrite them.
CFG = {
    "grasp": {
        "open_percentile": 90,
        "min_percentile": 10,
        "thr_fraction": 0.4,
        "hold_frames": 8,
        "max_speed": 0.15,
    },
    "filters": {
        "edge_margin_s": 0.2,
        "max_frame_jump": 0.05,
        "min_transport_xy": 0.07,
        "min_confidence": 0.3,
        "conf_tail_s": 0.5,
    },
}


def track(phases: list[tuple[int, float, tuple, tuple]]):
    """Build an episode from ``(n_frames, aperture, p_start, p_end)`` phases.

    Positions ramp linearly across each phase with the end excluded, so phases
    join without a duplicated frame and z is strictly monotonic where intended.
    """
    pos, ap = [], []
    for n, a, p0, p1 in phases:
        pos.append(np.linspace(p0, p1, n, endpoint=False))
        ap.append(np.full(n, a))
    pos = np.concatenate(pos)
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, 3] = pos
    return T, np.concatenate(ap), np.arange(len(pos)) / FPS


def cycle(*, dist: float = 0.20, carry: int = 200, release_hold: int = 40):
    """One pick-and-place: hover, descend, close mid-descent, carry, release, retract.

    Two properties this fixture has to have, both learned the hard way:

    * The hand closes **during** the descent, which is what EgoDex actually
      shows (P9), so the aperture crossing and the descent bottom are different
      frames. The two descent phases share a per-frame step of 0.005 m
      (0.25 m/s) so the speed across their boundary is the descent and not a
      discontinuity in the fixture.
    * It **ends where it began**, so cycles can be concatenated into a
      multi-closure episode without a position jump at the seam.

    Cycles are told apart by `dist`, not by position, for that second reason.
    """
    return [
        (40, OPEN, (0.0, 0.0, 0.20), (0.0, 0.0, 0.20)),            # hover, still
        (20, OPEN, (0.0, 0.0, 0.20), (0.0, 0.0, 0.10)),            # descend, open
        (10, CLOSED, (0.0, 0.0, 0.10), (0.0, 0.0, 0.05)),          # close, still falling
        (carry, CLOSED, (0.0, 0.0, 0.05), (dist, 0.0, 0.05)),      # carry
        (release_hold, OPEN, (dist, 0.0, 0.05), (dist, 0.0, 0.05)),  # release, still
        (60, OPEN, (dist, 0.0, 0.05), (0.0, 0.0, 0.20)),           # retract to hover
    ]


def only(segs):
    assert len(segs) == 1
    return segs[0]


def run(phases, conf=float("nan")):
    T, ap, t = track(phases)
    return T, ap, t, segment_all(T, ap, t, CFG, conf, descent_window_s=WINDOW_S)


# --- run detection ----------------------------------------------------------


def test_closed_runs_drops_flicker_shorter_than_hold():
    closed = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0], dtype=bool)
    assert closed_runs(closed, hold=4) == [(4, 9)]
    assert closed_runs(closed, hold=2) == [(1, 3), (4, 9)]


def test_flanked_runs_rejects_the_relaxed_hand_at_either_end():
    """P1: a run touching frame 0 or the last frame is a relaxed hand, not a grasp.

    The handoff's original detector fired on frame 0 in 79% of real episodes for
    exactly this reason.
    """
    hold = 3
    n = 12
    both = np.zeros(n, dtype=bool)
    both[:4] = True  # relaxed at the start
    both[-4:] = True  # relaxed at the end
    assert closed_runs(both, hold) == [(0, 4), (8, 12)]
    assert flanked_runs(both, hold) == []

    interior = np.zeros(n, dtype=bool)
    interior[4:8] = True
    assert flanked_runs(interior, hold) == [(4, 8)]


# --- the descent window (shared with common/demos.with_grasp_anchor_z) ------


def test_descent_window_end_is_clamped_by_the_release():
    # 0.3 s at 50 Hz is 15 frames past the grasp, and the end is exclusive of
    # the frame after that, so the window covers grasp .. grasp + 15 inclusive.
    assert descent_window_end(100, 400, 0.3, 50.0) == 116
    # A carry shorter than the window must not let the window reach into it:
    # picking up a dip that belongs to the carry anchors the grasp below the
    # object and the fingers close on the table (P9).
    assert descent_window_end(100, 105, 0.3, 50.0) == 106
    # Never empty, even for a degenerate window.
    assert descent_window_end(100, 400, 0.0, 50.0) == 102


def test_descent_bottom_finds_the_low_point_inside_the_window_only():
    z = np.array([0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.01])
    # The 0.01 at index 6 belongs to the carry and must not be picked up.
    assert descent_bottom(z, grasp_idx=1, end=5) == 3
    assert descent_bottom(z, grasp_idx=0, end=len(z)) == 6


# --- section 4b: the speed filter measures the right frame ------------------


def test_grasp_speed_is_horizontal_and_at_the_descent_bottom():
    """Section 4b. This demo descends briskly and then carries, which is normal
    human behaviour (P9) and must not be read as a pass-through."""
    seg = only(run(cycle())[3])

    crossing = seg.metrics["grasp_speed_3d_at_crossing"]
    # Descending at 0.005 m/frame = 0.25 m/s. The descent is vertical, so the
    # horizontal speed where the hand takes the object is zero: it is directly
    # above the object and has not started carrying yet.
    assert crossing == pytest.approx(0.25, abs=1e-9)
    assert seg.metrics["grasp_speed"] == pytest.approx(0.0, abs=1e-9)
    # The whole point: the old metric rejects this demo, the right one keeps it.
    assert crossing > CFG["grasp"]["max_speed"]
    assert seg.kept, seg.reject
    assert seg.metrics["descent_after_crossing"] == pytest.approx(0.05, abs=1e-9)


def test_a_genuine_pass_through_is_still_rejected():
    """The filter must keep catching what it is for: a hand that sweeps across
    the table, closing as it goes, and never settles on anything."""
    sweep = [
        (40, OPEN, (0.0, 0.0, 0.20), (0.0, 0.0, 0.20)),
        (20, OPEN, (0.0, 0.0, 0.20), (0.2, 0.0, 0.10)),      # travelling as it descends
        (10, CLOSED, (0.2, 0.0, 0.10), (0.3, 0.0, 0.05)),    # closes mid-sweep
        (60, CLOSED, (0.3, 0.0, 0.05), (0.9, 0.0, 0.05)),    # 0.01 m/frame = 0.5 m/s
        (40, OPEN, (0.9, 0.0, 0.05), (0.9, 0.0, 0.05)),
    ]
    seg = only(run(sweep)[3])
    assert seg.reject == "grasp_speed_too_high"
    assert seg.metrics["grasp_speed"] == pytest.approx(0.5, abs=1e-9)


def test_a_brisk_descent_alone_does_not_count_as_motion():
    """A purely vertical arrival is settled horizontally, however fast it fell."""
    plunge = cycle()
    # Fall 0.15 m in 5 frames at 0.03 m/frame: 1.5 m/s, ten times the limit.
    plunge[1] = (3, OPEN, (0.0, 0.0, 0.20), (0.0, 0.0, 0.11))
    plunge[2] = (2, CLOSED, (0.0, 0.0, 0.11), (0.0, 0.0, 0.05))
    seg = only(run(plunge)[3])
    assert seg.metrics["grasp_speed_3d_at_crossing"] == pytest.approx(1.5, rel=1e-6)
    assert seg.kept, seg.reject


# --- section 4a: one episode, several demos ---------------------------------


def test_a_single_clean_cycle_yields_one_demo():
    T, ap, t, segs = run(cycle())
    seg = only(segs)
    assert seg.kept, seg.reject
    assert (seg.n_candidates, seg.candidate) == (1, 0)
    assert seg.metrics["dist_xy"] == pytest.approx(0.20, abs=1e-9)
    # grasp and release index the SPAN, and the aperture there is below thr.
    s, e = seg.span
    assert ap[s + seg.grasp_idx] < seg.thr
    assert ap[s + seg.release_idx] > seg.thr


def test_an_episode_with_no_closure_is_rejected_once():
    """Nothing to segment is one episode-level reject, not a per-candidate one."""
    seg = only(run([(200, OPEN, (0.0, 0.0, 0.2), (0.3, 0.0, 0.2))])[3])
    assert seg.reject == "no_grasp_release_pair"
    assert (seg.n_candidates, seg.span, seg.grasp_idx) == (0, None, None)


def test_three_closures_yield_three_demos_with_tiling_spans():
    """Section 4a: re-gripping gives us MORE motions, not zero.

    Requiring exactly one closure rejected 572 of 1077 real episodes.
    """
    dists = (0.20, 0.30, 0.25)
    phases = [p for d in dists for p in cycle(dist=d)]
    T, ap, t, segs = run(phases)
    assert len(segs) == 3
    assert [s.candidate for s in segs] == [0, 1, 2]
    assert all(s.n_candidates == 3 for s in segs)
    assert all(s.kept for s in segs), [s.reject for s in segs]

    # Each demo owns [previous release, its own release + tail]: the spans butt
    # up against each other, so no frame is the carry of two different demos.
    assert segs[0].span[0] == 0
    for a, b in zip(segs, segs[1:]):
        assert a.span[0] < b.span[0] < a.span[1] < b.span[1]
        assert b.span[0] == a.span[0] + a.release_idx  # previous release
    # Each span ends `conf_tail_s` (0.5 s = 25 frames) past its own release, and
    # the last one stops there rather than running to the end of the episode.
    for s in segs:
        assert s.span[1] == s.span[0] + s.release_idx + 26
    assert segs[-1].span[1] < len(ap)

    # Each one is a full pick-and-place in its own right, and they are distinct.
    assert [s.metrics["dist_xy"] for s in segs] == pytest.approx(dists, abs=1e-9)


def test_filters_are_applied_per_candidate_not_to_the_whole_episode():
    """A bad closure must not take a good one down with it, or vice versa."""
    phases = cycle(dist=0.01) + cycle(dist=0.20)  # first transport too short
    segs = run(phases)[3]
    assert [s.reject for s in segs] == ["transport_too_short", None]


def test_grip_is_open_outside_the_closed_span():
    """conventions.md section 7.2: grip is 1 (open) outside [grasp, release)."""
    seg = only(run(cycle())[3])
    gi, ri = seg.grasp_idx, seg.release_idx
    assert len(seg.grip) == seg.span[1] - seg.span[0]
    assert np.all(seg.grip[:gi] == 1.0)
    assert np.all(seg.grip[gi:ri] == 0.0)
    assert np.all(seg.grip[ri:] == 1.0)


def test_confidence_is_judged_over_the_candidates_own_frames():
    """A confidence collapse in one closure must not reject the others."""
    T, ap, t = track(cycle() + cycle(dist=0.30))
    segs = segment_all(T, ap, t, CFG, float("nan"), descent_window_s=WINDOW_S)
    conf = np.ones(len(t))
    conf[: segs[0].release_idx - 5] = 0.1  # ruins only the first candidate's span

    segs = segment_all(T, ap, t, CFG, conf, descent_window_s=WINDOW_S)
    assert [s.reject for s in segs] == ["low_confidence", None]
    assert segs[1].metrics["conf_min_used"] == pytest.approx(1.0)


def test_a_tracking_glitch_is_judged_per_candidate_too():
    T, ap, t = track(cycle() + cycle(dist=0.30))
    T[5, 0, 3] += 0.5  # a jump in the first candidate's lead-in only
    segs = segment_all(T, ap, t, CFG, float("nan"), descent_window_s=WINDOW_S)
    assert [s.reject for s in segs] == ["position_jump", None]


def test_a_closure_without_room_to_approach_is_rejected():
    """A re-grip straight after a release has no approach to contribute.

    Measured on real data: this costs 55 later closures, 44 of which would
    survive a 2-sample lead-in requirement. Not taken — see P12.
    """
    # Written out rather than spliced from `cycle`, because the hand never
    # retracts between the two closures and the track has to stay continuous.
    phases = [
        (40, OPEN, (0.0, 0.0, 0.20), (0.0, 0.0, 0.20)),
        (20, OPEN, (0.0, 0.0, 0.20), (0.0, 0.0, 0.10)),
        (10, CLOSED, (0.0, 0.0, 0.10), (0.0, 0.0, 0.05)),
        (200, CLOSED, (0.0, 0.0, 0.05), (0.20, 0.0, 0.05)),
        (4, OPEN, (0.20, 0.0, 0.05), (0.20, 0.0, 0.05)),    # release, 4 frames
        (10, CLOSED, (0.20, 0.0, 0.05), (0.20, 0.0, 0.05)),  # straight back on
        (200, CLOSED, (0.20, 0.0, 0.05), (0.40, 0.0, 0.05)),
        (40, OPEN, (0.40, 0.0, 0.05), (0.40, 0.0, 0.05)),
    ]
    segs = run(phases)[3]
    assert len(segs) == 2
    assert segs[0].kept, segs[0].reject
    assert segs[1].reject == "event_too_close_to_end"
    assert segs[1].grasp_idx is None  # rejected candidates carry no indices


# --- the npz slice is self-contained (section 7.2) --------------------------


def test_demo_from_candidate_slices_a_self_contained_demo():
    T, ap, t, segs = run(cycle() + cycle(dist=0.30))
    ep = {
        "idx": "7", "t": t, "T_demo_tcp": T, "aperture": ap, "segs": segs,
        "llm_description": "", "source_path": "x.hdf5", "arm": "right",
    }

    second = demo_from_candidate(ep, segs[1])
    assert second["id"] == "7_1"
    # Section 7.2: `t` is seconds from THIS demo's start.
    assert second["t"][0] == 0.0
    n = len(second["t"])
    assert n == len(second["T_demo_tcp"]) == len(second["aperture"]) == len(segs[1].grip)
    assert n < len(t)  # a slice, not the whole episode
    # The grasp index addresses the sliced arrays, not the episode.
    assert second["aperture"][segs[1].grasp_idx] < segs[1].thr
    assert 0 < segs[1].grasp_idx < segs[1].release_idx < n
    # It really is the second cycle: that one carries 0.30 m, the first 0.20 m.
    release_x = second["T_demo_tcp"][segs[1].release_idx, 0, 3]
    assert release_x == pytest.approx(0.30, abs=1e-9)


# --- smoothing --------------------------------------------------------------


def test_smooth_positions_holds_the_ends_and_averages_the_middle():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = smooth_positions(pos, window=3)
    assert out.shape == pos.shape
    # Edge-padded, so the first and last stay where the demo began and ended.
    assert out[0, 0] == pytest.approx(1.0 / 3.0)
    assert out[1, 0] == pytest.approx(1.0)
    assert out[2, 0] == pytest.approx(2.0)
    assert smooth_positions(pos, window=1) is not pos
    assert np.array_equal(smooth_positions(pos, window=1), pos)
