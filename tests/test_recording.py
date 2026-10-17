"""Tests for `SimRobot`'s episode recorder (`sim/sim_robot.py`).

The alignment between `state`, `action` and the image is the whole contract: a
one-cycle shift is invisible in every summary statistic — frame counts, shapes,
dtypes and file sizes all stay correct — and silently teaches a policy to output
the command it has already executed. So it is asserted directly, against a
trajectory whose commanded values are distinguishable per cycle.

Rendering is covered separately and skipped where no GL context is available:
the joint-only path is what carries the alignment contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.path("sim", "scene_headless.xml").exists(),
    reason="run `uv run python sim/build_models.py` first",
)


@pytest.fixture(scope="module")
def robot():
    from sim.sim_robot import SimRobot

    # Headless: these tests never photograph anything, and the full visual model
    # costs ~1.65 GB of RSS to load.
    return SimRobot(render=False)


@pytest.fixture
def ramp(robot):
    """A short trajectory that moves one joint by a distinguishable amount per cycle.

    Resets first: `robot` is module-scoped, so reading `get_state` without a
    reset builds the ramp from wherever the previous test left the arm, and the
    accumulated drift walked the tcp out of the section 10 workspace box.
    """
    robot.reset()
    q0 = robot.get_state()["q"]
    slot = robot.index.arm_slots["right"][0]
    traj = np.repeat(q0[None, :], 6, axis=0)
    # Well inside the section 10 per-cycle limit of 0.05 rad, and small enough
    # in total that the tcp stays inside the workspace box from HOME.
    traj[:, slot] = q0[slot] + np.arange(6) * 0.01
    return traj, slot


# --- shape and lifecycle ----------------------------------------------------


def test_episode_has_one_frame_per_commanded_cycle(robot, ramp):
    traj, _ = ramp
    robot.reset()
    robot.start_episode()
    robot.stream(traj, 50.0)
    ep = robot.end_episode()
    assert len(ep) == len(traj)
    assert ep.state.shape == traj.shape
    assert ep.action.shape == traj.shape
    assert ep.t.shape == (len(traj),)


def test_an_episode_that_never_streamed_is_empty_not_an_error(robot):
    """A trial can fail before it commands a single cycle.

    `run_demo` returns early on an unreachable approach pose or no feasible
    wrist angle, having streamed nothing. The recording must still close into a
    well-shaped empty episode — inferring the column count from an empty list
    raised `cannot reshape array of size 0` and killed the whole generator pool.
    """
    robot.reset()
    robot.start_episode()
    ep = robot.end_episode()
    assert len(ep) == 0
    assert ep.state.shape == (0, 16)
    assert ep.action.shape == (0, 16)
    assert ep.t.shape == (0,)


def test_end_episode_without_start_is_an_error(robot):
    robot.reset()
    with pytest.raises(RuntimeError, match="without a matching start_episode"):
        robot.end_episode()


def test_streaming_outside_an_episode_records_nothing(robot, ramp):
    """The batch physics path must not pay for, or accumulate, frames."""
    traj, _ = ramp
    robot.reset()
    robot.stream(traj, 50.0)
    assert robot._capture is None


def test_a_second_episode_does_not_inherit_the_first(robot, ramp):
    traj, _ = ramp
    robot.reset()
    robot.start_episode()
    robot.stream(traj, 50.0)
    robot.end_episode()

    robot.reset()
    robot.start_episode()
    robot.stream(traj[:3], 50.0)
    assert len(robot.end_episode()) == 3


def test_an_episode_spans_several_streams(robot, ramp):
    """`run_demo` streams the bridge, the path and the gripper holds separately."""
    traj, _ = ramp
    robot.reset()
    robot.start_episode()
    robot.stream(traj, 50.0)
    robot.stream(traj, 50.0)
    assert len(robot.end_episode()) == 2 * len(traj)


# --- the alignment contract -------------------------------------------------


def test_action_is_the_command_issued_at_that_frame(robot, ramp):
    """`action[i]` is exactly what was commanded on cycle `i`."""
    traj, _ = ramp
    robot.reset()
    robot.start_episode()
    robot.stream(traj, 50.0)
    ep = robot.end_episode()
    assert np.allclose(ep.action, traj)


def test_state_is_observed_before_the_command_not_after(robot, ramp):
    """The first state is the pre-command pose, not the result of commanding it.

    This is the shift that a frame-count check cannot see. `state[0]` must be
    the HOME pose the robot was reset to — if the recorder stepped first, it
    would instead hold the pose produced by `action[0]`.
    """
    traj, slot = ramp
    robot.reset()
    home = robot.get_state()["q"]
    robot.start_episode()
    robot.stream(traj, 50.0)
    ep = robot.end_episode()

    assert np.allclose(ep.state[0], home)
    # And the recorded time of the first frame is before any stepping happened.
    assert ep.t[0] == pytest.approx(0.0, abs=1e-12)
    # Later states must trail the commands rather than match them: a position
    # actuator needs cycles to converge, so state[i] lags action[i].
    assert abs(ep.state[-1][slot] - ep.action[-1][slot]) > 0


def test_state_tracks_the_previous_command(robot, ramp):
    """`state[i + 1]` is closer to `action[i]` than `state[i]` was.

    The weaker, physics-independent half of the contract: whatever the actuator
    dynamics, commanding a joint upward must move the next observed state toward
    that command.
    """
    traj, slot = ramp
    robot.reset()
    robot.start_episode()
    robot.stream(traj, 50.0)
    ep = robot.end_episode()

    for i in range(len(ep) - 1):
        before = abs(ep.state[i][slot] - ep.action[i][slot])
        after = abs(ep.state[i + 1][slot] - ep.action[i][slot])
        assert after <= before + 1e-12, f"frame {i} moved away from its command"


def test_frame_times_advance_by_one_control_cycle(robot, ramp):
    traj, _ = ramp
    robot.reset()
    robot.start_episode()
    robot.stream(traj, 50.0)
    ep = robot.end_episode()
    assert np.allclose(np.diff(ep.t), 1.0 / 50.0)


# --- images -----------------------------------------------------------------


def test_no_camera_means_no_frames(robot, ramp):
    traj, _ = ramp
    robot.reset()
    robot.start_episode(camera=None)
    robot.stream(traj, 50.0)
    ep = robot.end_episode()
    assert ep.jpegs is None
    assert ep.frames() is None


def test_headless_robot_refuses_to_render(robot):
    """The headless scene has no visual geometry, so it must fail loudly."""
    robot.reset()
    with pytest.raises(RuntimeError, match="headless scene"):
        robot.start_episode(camera="overhead")


@pytest.mark.skipif(
    not os.environ.get("MUJOCO_GL"), reason="needs MUJOCO_GL (egl/osmesa) for offscreen rendering"
)
def test_recorded_frames_decode_to_the_configured_size(ramp):
    from sim.sim_robot import SimRobot

    size = int(config.skills()["dataset"]["image_size"])
    rig = SimRobot(render=True)
    q0 = rig.get_state()["q"]
    traj = np.repeat(q0[None, :], 3, axis=0)

    rig.start_episode(camera=config.skills()["dataset"]["camera"])
    rig.stream(traj, 50.0)
    ep = rig.end_episode()

    assert len(ep.jpegs) == 3
    frames = ep.frames()
    assert frames.shape == (3, size, size, 3)
    assert frames.dtype == np.uint8
    # A real render, not a blank buffer.
    assert frames[0].std() > 1.0
