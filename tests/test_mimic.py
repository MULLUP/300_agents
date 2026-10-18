"""Tests for `common/mimic.py` — the object-centric transform (`conventions.md` section 8).

The defining property, and the one the whole approach rests on:

    re-anchoring a demo onto a scene must land the gripper exactly on the
    object at the grasp, and exactly on the target at the release.

If that holds for arbitrary demos and arbitrary scenes, the transform is right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.mimic import (  # noqa: E402
    BRIDGE_S,
    CUBE_YAW_PERIOD,
    Retargeted,
    SourceDemo,
    anchor,
    bridge,
    demo_anchors,
    heading_rotation,
    mirror_demo,
    retarget,
    select,
    snap_yaw,
    transform_segment,
)
from common.transforms import inv_T, make_T, rot_z, yaw_of  # noqa: E402

RATE = 50.0


def random_pose(rng) -> np.ndarray:
    """A pose with a full random rotation, so roll/pitch handling gets exercised."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return make_T(Q, rng.uniform(-0.5, 0.5, 3))


def make_demo(rng, n=40, grasp=10, release=30, **kw) -> SourceDemo:
    poses = np.stack([random_pose(rng) for _ in range(n)])
    grip = np.ones(n)
    grip[grasp:release] = 0.0
    return SourceDemo(poses, grip, grasp, release, **kw)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- schema validation ------------------------------------------------------


def test_demo_rejects_bad_indices(rng):
    poses = np.stack([random_pose(rng) for _ in range(10)])
    with pytest.raises(ValueError, match="release_idx"):
        SourceDemo(poses, np.ones(10), 5, 5)
    with pytest.raises(ValueError, match="release_idx"):
        SourceDemo(poses, np.ones(10), 7, 3)
    with pytest.raises(ValueError, match="release_idx"):
        SourceDemo(poses, np.ones(10), 0, 10)


def test_demo_rejects_mismatched_grip(rng):
    poses = np.stack([random_pose(rng) for _ in range(10)])
    with pytest.raises(ValueError, match="grip has"):
        SourceDemo(poses, np.ones(9), 2, 7)


def test_measured_anchor_requires_obj0(rng):
    poses = np.stack([random_pose(rng) for _ in range(10)])
    with pytest.raises(ValueError, match="requires T_demo_obj0"):
        SourceDemo(poses, np.ones(10), 2, 7, anchor_type="measured")


def test_unknown_anchor_type(rng):
    poses = np.stack([random_pose(rng) for _ in range(10)])
    with pytest.raises(ValueError, match="unknown anchor_type"):
        SourceDemo(poses, np.ones(10), 2, 7, anchor_type="guessed")


# --- step 2: anchors --------------------------------------------------------


def test_anchor_zeroes_roll_and_pitch(rng):
    for _ in range(10):
        T = random_pose(rng)
        a = anchor(T)
        # Yaw preserved, z-axis vertical, position untouched.
        assert yaw_of(a) == pytest.approx(yaw_of(T), abs=1e-9)
        np.testing.assert_allclose(a[:3, 2], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(a[:3, 3], T[:3, 3])
        assert np.linalg.det(a[:3, :3]) == pytest.approx(1.0)


def test_anchor_is_idempotent(rng):
    T = random_pose(rng)
    np.testing.assert_allclose(anchor(anchor(T)), anchor(T), atol=1e-12)


def test_demo_anchors_use_the_gripper_for_egodex(rng):
    demo = make_demo(rng)
    g, r = demo_anchors(demo)
    np.testing.assert_allclose(g, anchor(demo.T_demo_tcp[demo.grasp_idx]))
    np.testing.assert_allclose(r, anchor(demo.T_demo_tcp[demo.release_idx]))


def test_demo_anchors_use_the_object_for_teleop(rng):
    obj0 = random_pose(rng)
    demo = make_demo(rng, anchor_type="measured", T_demo_obj0=obj0)
    g, r = demo_anchors(demo)
    np.testing.assert_allclose(g, anchor(obj0))
    np.testing.assert_allclose(r, anchor(demo.T_demo_tcp[demo.release_idx]))


# --- step 3: heading -------------------------------------------------------


def test_heading_rotation_aligns_the_transport_direction(rng):
    for _ in range(10):
        g = make_T(np.eye(3), rng.uniform(-0.3, 0.3, 3))
        r = make_T(np.eye(3), g[:3, 3] + np.array([*rng.uniform(-0.3, 0.3, 2), 0.1]))
        obj = make_T(np.eye(3), rng.uniform(-0.3, 0.3, 3))
        tgt = make_T(np.eye(3), obj[:3, 3] + np.array([*rng.uniform(-0.3, 0.3, 2), 0.05]))

        align = heading_rotation(g, r, obj, tgt)
        demo_vec = ((align @ r)[:3, 3] - (align @ g)[:3, 3])[:2]
        scene_vec = (tgt[:3, 3] - obj[:3, 3])[:2]
        # Headings now match (directions, not magnitudes).
        assert np.arctan2(*demo_vec[::-1]) == pytest.approx(
            np.arctan2(*scene_vec[::-1]), abs=1e-9
        )


def test_heading_rotation_keeps_the_grasp_anchor_fixed(rng):
    g = make_T(rot_z(0.4), [0.2, -0.1, 0.05])
    r = make_T(np.eye(3), [0.4, 0.1, 0.1])
    obj = make_T(np.eye(3), [0.3, -0.2, 0.02])
    tgt = make_T(np.eye(3), [0.1, 0.3, 0.02])
    align = heading_rotation(g, r, obj, tgt)
    np.testing.assert_allclose((align @ g)[:3, 3], g[:3, 3], atol=1e-12)


def test_heading_rotation_is_identity_for_a_vertical_transport():
    """No horizontal direction exists, so no rotation is defined."""
    g = make_T(np.eye(3), [0.3, 0.0, 0.02])
    r = make_T(np.eye(3), [0.3, 0.0, 0.20])  # straight up
    obj = make_T(np.eye(3), [0.3, -0.2, 0.02])
    tgt = make_T(np.eye(3), [0.1, 0.3, 0.02])
    np.testing.assert_allclose(heading_rotation(g, r, obj, tgt), np.eye(4))
    # ... and symmetrically when the scene target sits on the object.
    np.testing.assert_allclose(heading_rotation(g, make_T(np.eye(3), [0.5, 0.1, 0.02]), obj, obj), np.eye(4))


def test_heading_rotation_is_a_pure_z_rotation(rng):
    g = make_T(rot_z(0.2), [0.2, -0.1, 0.05])
    r = make_T(np.eye(3), [0.4, 0.1, 0.1])
    obj = make_T(np.eye(3), [0.3, -0.2, 0.02])
    tgt = make_T(np.eye(3), [0.1, 0.3, 0.02])
    R = heading_rotation(g, r, obj, tgt)[:3, :3]
    np.testing.assert_allclose(R[2, :], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-12)


# --- step 4: segment transform ---------------------------------------------


def test_transform_segment_maps_src_onto_tgt(rng):
    """The anchor itself must land exactly on the target. This is the core identity."""
    for _ in range(10):
        src, tgt = random_pose(rng), random_pose(rng)
        out = transform_segment(src[None], src, tgt)
        np.testing.assert_allclose(out[0], tgt, atol=1e-12)


def test_transform_segment_is_rigid(rng):
    """Re-anchoring must not stretch the demo: pairwise distances are preserved."""
    demo = np.stack([random_pose(rng) for _ in range(12)])
    out = transform_segment(demo, random_pose(rng), random_pose(rng))
    d_in = np.linalg.norm(demo[:, None, :3, 3] - demo[None, :, :3, 3], axis=-1)
    d_out = np.linalg.norm(out[:, None, :3, 3] - out[None, :, :3, 3], axis=-1)
    np.testing.assert_allclose(d_in, d_out, atol=1e-9)


def test_transform_segment_preserves_relative_poses(rng):
    demo = np.stack([random_pose(rng) for _ in range(6)])
    out = transform_segment(demo, random_pose(rng), random_pose(rng))
    for k in range(1, len(demo)):
        np.testing.assert_allclose(
            inv_T(demo[0]) @ demo[k], inv_T(out[0]) @ out[k], atol=1e-9
        )


# --- snap_yaw ---------------------------------------------------------------


def test_snap_yaw_returns_a_symmetric_equivalent():
    for target in np.linspace(-np.pi, np.pi, 17):
        for ref in np.linspace(-np.pi, np.pi, 17):
            got = snap_yaw(target, ref)
            n = (got - target) / CUBE_YAW_PERIOD
            assert n == pytest.approx(round(n), abs=1e-9)


def test_snap_yaw_picks_the_nearest():
    for target in np.linspace(-np.pi, np.pi, 17):
        for ref in np.linspace(-np.pi, np.pi, 17):
            got = snap_yaw(target, ref)
            # No equivalent is closer to the reference than the chosen one.
            assert abs(got - ref) <= CUBE_YAW_PERIOD / 2.0 + 1e-9


def test_snap_yaw_identity_when_already_aligned():
    assert snap_yaw(0.3, 0.3) == pytest.approx(0.3)


# --- step 5: bridging -------------------------------------------------------


def test_bridge_endpoints_and_length(rng):
    a, b = random_pose(rng), random_pose(rng)
    out = bridge(a, b, RATE)
    assert len(out) == int(round(BRIDGE_S * RATE))
    np.testing.assert_allclose(out[0], a, atol=1e-9)
    np.testing.assert_allclose(out[-1], b, atol=1e-9)


def test_bridge_stays_a_valid_pose_sequence(rng):
    out = bridge(random_pose(rng), random_pose(rng), RATE)
    for T in out:
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)


# --- retarget: the end-to-end property --------------------------------------


def test_retarget_lands_on_the_object_and_the_target(rng):
    """The whole point: grasp on the object, release on the target."""
    for _ in range(12):
        demo = make_demo(rng)
        obj = make_T(rot_z(rng.uniform(-np.pi, np.pi)), [*rng.uniform(0.2, 0.5, 2), 0.0225])
        tgt = make_T(rot_z(rng.uniform(-np.pi, np.pi)), [*rng.uniform(0.2, 0.5, 2), 0.10])

        out = retarget(demo, obj, tgt, RATE)
        i0, i1 = out.segments["approach"]
        j0, j1 = out.segments["transport"]

        # Last approach pose sits on the object's position.
        np.testing.assert_allclose(out.T_table_tcp[i1 - 1][:3, 3], obj[:3, 3], atol=1e-9)
        # Last transport pose sits on the place target's position.
        np.testing.assert_allclose(out.T_table_tcp[j1 - 1][:3, 3], tgt[:3, 3], atol=1e-9)


def test_retarget_segments_tile_the_output(rng):
    demo = make_demo(rng)
    out = retarget(
        demo,
        make_T(np.eye(3), [0.3, -0.2, 0.0225]),
        make_T(np.eye(3), [0.4, 0.1, 0.1]),
        RATE,
        T_table_tcp_now=make_T(np.eye(3), [0.4, -0.15, 0.15]),
    )
    order = ["bridge_in", "approach", "bridge_mid", "transport"]
    assert list(out.segments) == order
    assert out.segments[order[0]][0] == 0
    for a, b in zip(order[:-1], order[1:]):
        assert out.segments[a][1] == out.segments[b][0]
    assert out.segments[order[-1]][1] == len(out.T_table_tcp) == len(out.grip)


def test_retarget_without_a_current_pose_has_no_lead_bridge(rng):
    out = retarget(
        make_demo(rng),
        make_T(np.eye(3), [0.3, -0.2, 0.0225]),
        make_T(np.eye(3), [0.4, 0.1, 0.1]),
        RATE,
    )
    assert "bridge_in" not in out.segments
    assert out.segments["approach"][0] == 0


def test_retarget_grip_is_closed_through_the_mid_bridge_and_transport(rng):
    demo = make_demo(rng)
    out = retarget(
        demo, make_T(np.eye(3), [0.3, -0.2, 0.0225]), make_T(np.eye(3), [0.4, 0.1, 0.1]), RATE
    )
    b0, b1 = out.segments["bridge_mid"]
    assert np.all(out.grip[b0:b1] == 0.0), "must hold the object across the join"
    t0, t1 = out.segments["transport"]
    # The demo's own grip said closed from grasp_idx to release_idx.
    assert out.grip[t0] == 0.0


def test_retarget_grip_is_open_at_the_start(rng):
    demo = make_demo(rng)
    out = retarget(
        demo,
        make_T(np.eye(3), [0.3, -0.2, 0.0225]),
        make_T(np.eye(3), [0.4, 0.1, 0.1]),
        RATE,
        T_table_tcp_now=make_T(np.eye(3), [0.4, -0.15, 0.15]),
    )
    assert out.grip[0] == 1.0


def test_retarget_rejects_bad_shapes(rng):
    demo = make_demo(rng)
    with pytest.raises(ValueError, match="T_table_obj must be 4x4"):
        retarget(demo, np.eye(3), make_T(np.eye(3), [0.4, 0.1, 0.1]), RATE)
    with pytest.raises(ValueError, match="T_table_tgt must be 4x4"):
        retarget(demo, make_T(np.eye(3), [0.3, -0.2, 0.02]), np.eye(3), RATE)


def test_retarget_output_poses_are_valid(rng):
    out = retarget(
        make_demo(rng),
        make_T(np.eye(3), [0.3, -0.2, 0.0225]),
        make_T(np.eye(3), [0.4, 0.1, 0.1]),
        RATE,
    )
    for T in out.T_table_tcp:
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)


# --- step 1: mirroring ------------------------------------------------------


def test_mirror_demo_flips_y_and_tags_the_arm(rng):
    demo = make_demo(rng)
    m = mirror_demo(demo)
    assert m.arm == "left"
    np.testing.assert_allclose(m.T_demo_tcp[:, 1, 3], -demo.T_demo_tcp[:, 1, 3])
    # Still proper rotations, i.e. not a reflection left behind.
    for T in m.T_demo_tcp:
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_mirror_demo_preserves_timing_and_grip(rng):
    demo = make_demo(rng)
    m = mirror_demo(demo)
    assert (m.grasp_idx, m.release_idx) == (demo.grasp_idx, demo.release_idx)
    np.testing.assert_allclose(m.grip, demo.grip)


def test_mirror_demo_refuses_a_left_demo(rng):
    with pytest.raises(ValueError, match="already a left-arm demo"):
        mirror_demo(mirror_demo(make_demo(rng)))


def test_mirror_demo_mirrors_the_measured_object(rng):
    obj0 = random_pose(rng)
    m = mirror_demo(make_demo(rng, anchor_type="measured", T_demo_obj0=obj0))
    assert m.T_demo_obj0[1, 3] == pytest.approx(-obj0[1, 3])


# --- index and selection ----------------------------------------------------


def test_index_matches_the_schema(rng):
    demo = make_demo(rng)
    g = demo.T_demo_tcp[demo.grasp_idx][:3, 3]
    r = demo.T_demo_tcp[demo.release_idx][:3, 3]
    assert demo.index["dist_xy"] == pytest.approx(np.hypot(*(r - g)[:2]))
    assert demo.index["dz"] == pytest.approx(r[2] - g[2])


def test_select_retrieval_takes_the_nearest():
    index = np.array([[0.10, 0.0], [0.20, 0.05], [0.30, -0.02]])
    assert select(index, [0.205, 0.045]) == 1
    assert select(index, [0.0, 0.0]) == 0


def test_select_datagen_stays_within_the_k_nearest():
    index = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.9, 0.0]])
    rng = np.random.default_rng(0)
    picked = {select(index, [0.15, 0.0], mode="datagen", k=2, rng=rng) for _ in range(60)}
    assert picked <= {0, 1}
    assert len(picked) == 2, "datagen should actually vary its choice"


def test_select_datagen_k_larger_than_the_library():
    index = np.array([[0.1, 0.0], [0.2, 0.0]])
    rng = np.random.default_rng(0)
    assert select(index, [0.15, 0.0], mode="datagen", k=99, rng=rng) in (0, 1)


def test_select_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        select(np.array([[0.1, 0.0]]), [0.1, 0.0], mode="vibes")
