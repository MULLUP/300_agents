"""Tests for `common/transforms.py` (handoff step 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.transforms import (  # noqa: E402
    R_ZUP,
    arkit_to_demo,
    gripper_frame,
    interp_poses,
    inv_T,
    make_T,
    mat_to_quat_wxyz,
    mirror_lr,
    pos_yaw_pose,
    quat_wxyz_to_mat,
    resample_poses,
    rot_z,
    slerp,
    yaw_of,
)
from data.egodex_io import episode_paths, load_episode, tcp_inputs  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "egodex_raw"
APERTURE_MAX = 0.15  # meters; handoff step 3 acceptance


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def random_rotations(rng, n):
    """``(n,3,3)`` uniformly random proper rotations, via QR with a sign fix."""
    Q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    Q[np.linalg.det(Q) < 0, :, 0] *= -1.0
    return Q


def random_poses(rng, n):
    return make_T(random_rotations(rng, n), rng.normal(size=(n, 3)))


def test_inv_T_round_trip(rng):
    T = random_poses(rng, 32)
    eye = np.broadcast_to(np.eye(4), T.shape)
    np.testing.assert_allclose(inv_T(T) @ T, eye, atol=1e-12)
    np.testing.assert_allclose(T @ inv_T(T), eye, atol=1e-12)


def test_inv_T_single_pose(rng):
    T = random_poses(rng, 1)[0]
    assert T.shape == (4, 4)
    np.testing.assert_allclose(inv_T(T) @ T, np.eye(4), atol=1e-12)


def test_quat_round_trip(rng):
    R = random_rotations(rng, 64)
    np.testing.assert_allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(R)), R, atol=1e-12)


def test_quat_round_trip_from_quat(rng):
    q = rng.normal(size=(64, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(q[:, 0:1] < 0, -1.0, 1.0)  # canonical w >= 0, as the module returns
    np.testing.assert_allclose(mat_to_quat_wxyz(quat_wxyz_to_mat(q)), q, atol=1e-12)


def test_quat_is_wxyz_not_xyzw():
    """A 90 deg rotation about +z must give w = cos(45 deg) in slot 0."""
    q = mat_to_quat_wxyz(rot_z(np.pi / 2))
    np.testing.assert_allclose(q, [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-12)


def test_arkit_to_demo_maps_y_up_to_z_up():
    """ARKit's up axis (0,1,0) must become the demo frame's up axis (0,0,1)."""
    T = make_T(np.eye(3), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(arkit_to_demo(T)[:3, 3], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(R_ZUP @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)


def test_arkit_to_demo_batched(rng):
    T = random_poses(rng, 8)
    batched = arkit_to_demo(T)
    assert batched.shape == (8, 4, 4)
    for i in range(8):
        np.testing.assert_allclose(batched[i], arkit_to_demo(T[i]), atol=1e-12)


def test_arkit_to_demo_preserves_rotation_properness(rng):
    T = arkit_to_demo(random_poses(rng, 16))
    np.testing.assert_allclose(np.linalg.det(T[:, :3, :3]), 1.0, atol=1e-12)


def test_yaw_of_matches_rot_z():
    for theta in np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 25):
        T = make_T(rot_z(theta), np.zeros(3))
        assert yaw_of(T) == pytest.approx(theta, abs=1e-12)


def test_pos_yaw_pose_keeps_position_and_yaw_only(rng):
    T = random_poses(rng, 16)
    reduced = pos_yaw_pose(T)
    np.testing.assert_allclose(reduced[:, :3, 3], T[:, :3, 3], atol=1e-12)
    np.testing.assert_allclose(yaw_of(reduced), yaw_of(T), atol=1e-12)
    # Roll and pitch gone: the reduced z-axis is the table normal.
    np.testing.assert_allclose(reduced[:, :3, 2], np.tile([0.0, 0.0, 1.0], (16, 1)), atol=1e-12)


def test_mirror_lr_is_a_proper_rotation(rng):
    T = random_poses(rng, 16)
    mirrored = mirror_lr(T)
    np.testing.assert_allclose(np.linalg.det(mirrored[:, :3, :3]), 1.0, atol=1e-12)
    np.testing.assert_allclose(mirrored[:, 1, 3], -T[:, 1, 3], atol=1e-12)
    np.testing.assert_allclose(mirror_lr(mirrored), T, atol=1e-12)  # involution


def test_slerp_endpoints_and_midpoint(rng):
    R0, R1 = random_rotations(rng, 8), random_rotations(rng, 8)
    np.testing.assert_allclose(slerp(R0, R1, np.zeros(8)), R0, atol=1e-10)
    np.testing.assert_allclose(slerp(R0, R1, np.ones(8)), R1, atol=1e-10)
    half = slerp(R0, R1, np.full(8, 0.5))
    np.testing.assert_allclose(np.linalg.det(half), 1.0, atol=1e-10)
    # The midpoint is equidistant from both ends.
    from common.transforms import rot_angle

    np.testing.assert_allclose(rot_angle(R0, half), rot_angle(half, R1), atol=1e-10)


def test_slerp_identical_rotations_is_stable(rng):
    """Equal endpoints hit the sin(theta) ~ 0 branch; it must not produce NaN."""
    R = random_rotations(rng, 8)
    out = slerp(R, R.copy(), np.linspace(0, 1, 8))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, R, atol=1e-10)


def test_interp_poses_endpoints(rng):
    T0, T1 = random_poses(rng, 2)
    path = interp_poses(T0, T1, 25)
    assert path.shape == (25, 4, 4)
    np.testing.assert_allclose(path[0], T0, atol=1e-10)
    np.testing.assert_allclose(path[-1], T1, atol=1e-10)


def test_resample_poses_recovers_source_times(rng):
    t = np.arange(20) / 30.0
    T = random_poses(rng, 20)
    np.testing.assert_allclose(resample_poses(t, T, t), T, atol=1e-10)


def test_resample_poses_linear_positions():
    """A straight constant-speed path must resample to exact linear positions."""
    t = np.arange(11) / 10.0
    p = np.stack([t, 2 * t, -t], axis=1)
    T = make_T(np.broadcast_to(np.eye(3), (11, 3, 3)), p)
    t_dst = np.linspace(0.0, 1.0, 51)
    out = resample_poses(t, T, t_dst)
    np.testing.assert_allclose(out[:, :3, 3], np.stack([t_dst, 2 * t_dst, -t_dst], axis=1), atol=1e-12)


def test_gripper_frame_construction_on_synthetic_hand():
    """A hand laid out on the axes must give the documented tcp axes exactly."""
    wrist = np.array([[0.0, 0.0, 0.0]])
    thumb_mcp = np.array([[0.0, -0.02, 0.08]])
    index_mcp = np.array([[0.0, 0.02, 0.08]])
    thumb_tip = np.array([[0.0, -0.03, 0.14]])
    index_tip = np.array([[0.0, 0.03, 0.14]])

    T, aperture = gripper_frame(wrist, thumb_mcp, index_mcp, thumb_tip, index_tip)

    np.testing.assert_allclose(T[0, :3, 3], [0.0, 0.0, 0.14], atol=1e-12)  # tip midpoint
    assert aperture[0] == pytest.approx(0.06, abs=1e-12)  # tip separation
    np.testing.assert_allclose(T[0, :3, 2], [0.0, 0.0, 1.0], atol=1e-12)  # z = approach
    np.testing.assert_allclose(T[0, :3, 1], [0.0, 1.0, 0.0], atol=1e-12)  # y = jaw
    np.testing.assert_allclose(T[0, :3, 0], np.cross(T[0, :3, 1], T[0, :3, 2]), atol=1e-12)


def test_gripper_frame_jaw_is_orthogonalized_against_approach():
    """A jaw axis deliberately tilted into the approach direction gets projected out."""
    wrist = np.array([[0.0, 0.0, 0.0]])
    thumb_mcp = np.array([[0.0, -0.02, 0.06]])
    index_mcp = np.array([[0.0, 0.02, 0.10]])  # not perpendicular to the approach
    thumb_tip = np.array([[0.0, -0.03, 0.14]])
    index_tip = np.array([[0.0, 0.03, 0.14]])

    T, _ = gripper_frame(wrist, thumb_mcp, index_mcp, thumb_tip, index_tip)
    assert np.dot(T[0, :3, 1], T[0, :3, 2]) == pytest.approx(0.0, abs=1e-12)


def episode_files(n):
    paths = episode_paths(RAW_DIR)
    if len(paths) < n:
        pytest.skip(f"need {n} episodes in {RAW_DIR}, found {len(paths)}")
    return paths[:n]


@pytest.mark.parametrize("path", episode_files(3) if RAW_DIR.exists() else [])
def test_gripper_frame_on_real_episode(path):
    """Rotations orthonormal with det = +1, and a plausible aperture, on real data."""
    ep = load_episode(path)
    ep["transforms"] = {k: arkit_to_demo(v) for k, v in ep["transforms"].items()}
    pts = tcp_inputs(ep)

    T, aperture = gripper_frame(
        pts["wrist"], pts["thumb_mcp"], pts["index_mcp"], pts["thumb_tip"], pts["index_tip"]
    )

    assert T.shape == (ep["N"], 4, 4)
    R = T[:, :3, :3]
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)
    eye = np.broadcast_to(np.eye(3), R.shape)
    np.testing.assert_allclose(np.swapaxes(R, 1, 2) @ R, eye, atol=1e-9)

    assert np.isfinite(aperture).all()
    in_range = np.mean((aperture >= 0.0) & (aperture <= APERTURE_MAX))
    assert in_range >= 0.99, f"{path.name}: only {in_range:.1%} of apertures in [0, {APERTURE_MAX}] m"
