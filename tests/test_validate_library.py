"""Tests for the pure parts of step 9's scoring (`sim/validate_library.py`).

Scoring a real candidate needs MuJoCo and a minute of IK per trial, so what is
tested here is the bookkeeping either side of it: the failure-reason collapsing
that the diagnosis in `AGENT_HANDOFF.md` section 4c is read off, and the tally
that decides whether an entry clears `validation.min_success_rate`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.validate_library import _reason_key, scene_draw, tally  # noqa: E402


def test_reason_key_collapses_the_physics_failures():
    """These four are the step 9 diagnosis categories; they must not drift."""
    assert _reason_key("") == "ok"
    assert _reason_key("cube at (0.3,-0.2) outside the bin footprint") == "cube_not_in_bin"
    assert _reason_key("cube above the rim by 0.02 m") == "cube_above_rim"
    assert _reason_key("cube still moving at 0.08 m/s") == "cube_not_at_rest"


def test_reason_key_keeps_the_leading_token_for_everything_else():
    assert _reason_key("unsafe_trajectory (tcp_speed 0.27 m/s)") == "unsafe_trajectory"
    assert _reason_key("no_feasible_yaw (approach: ik_diverged)") == "no_feasible_yaw"
    assert _reason_key("ik_error") == "ik_error"


def test_tally_counts_successes_and_groups_failures():
    rows = [
        ("a", True, "ok"),
        ("a", False, "cube_not_in_bin"),
        ("a", True, "ok"),
        ("a", False, "cube_not_in_bin"),
        ("a", False, "unsafe_trajectory"),
    ]
    out = tally("a", rows, n_trials=5)
    assert out["id"] == "a"
    assert (out["n_success"], out["n_trials"]) == (2, 5)
    assert out["success_rate"] == pytest.approx(0.4)
    assert out["failures"] == {"cube_not_in_bin": 2, "unsafe_trajectory": 1}
    # Successes must not leak into the failure histogram.
    assert "ok" not in out["failures"]


def test_tally_of_a_clean_sweep_has_no_failures():
    out = tally("b", [("b", True, "ok")] * 3, n_trials=3)
    assert (out["success_rate"], out["failures"]) == (1.0, {})


def test_scene_draw_is_reproducible_and_inside_the_pick_zone():
    """Every candidate must see the same scenes, or success rates are not
    comparable between entries -- which is the whole point of the fixed seed."""
    from common import config

    a = scene_draw(12, seed=99)
    assert len(a) == 12
    assert [(tuple(xy), yaw) for xy, yaw in a] == [
        (tuple(xy), yaw) for xy, yaw in scene_draw(12, seed=99)
    ]
    assert scene_draw(12, seed=100)[0][0].tolist() != a[0][0].tolist()

    zone = config.skills()["scene"]["pick_zone"]
    for xy, yaw in a:
        for v, c, s in zip(xy, zone["center"], zone["size"]):
            assert c - s / 2 <= v <= c + s / 2
        # A cube's yaw is 90 deg-symmetric, so +/- 45 deg covers every pose.
        assert -3.15 / 4 <= yaw <= 3.15 / 4


def test_scene_draw_is_a_prefix_so_trial_counts_stay_comparable():
    """An 8-trial run must be the first 8 scenes of a 20-trial run, or section 0b's
    "re-measure at 20 trials" would be comparing different draws."""
    short, long = scene_draw(8, seed=7), scene_draw(20, seed=7)
    assert [(tuple(xy), yaw) for xy, yaw in short] == [
        (tuple(xy), yaw) for xy, yaw in long[:8]
    ]
