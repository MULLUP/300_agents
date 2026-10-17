"""Pose math and frame conversions (`conventions.md` sections 2 and 3).

Poses are 4x4 homogeneous float64 matrices named ``T_a_b`` -- the pose of frame
``b`` expressed in frame ``a``. Every function here accepts either a single
``(4,4)`` pose or a batch ``(N,4,4)`` and returns the matching shape.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# arkit (+y up) -> demo (+z up). conventions.md section 2.
R_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Left-right mirror in the `table` frame. conventions.md section 2.
M_LR = np.diag([1.0, -1.0, 1.0])


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble ``(...,3,3)`` rotations and ``(...,3)`` positions into ``(...,4,4)``."""
    R = np.asarray(R, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    batch = R.shape[:-2]
    T = np.zeros(batch + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, using ``R^T`` rather than a general solve."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make_T(Rt, -(Rt @ p[..., None])[..., 0])


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """``(...,4)`` quaternions in MuJoCo ``(w,x,y,z)`` order -> ``(...,3,3)``."""
    q = np.asarray(q, dtype=np.float64)
    batch = q.shape[:-1]
    xyzw = np.roll(q.reshape(-1, 4), -1, axis=-1)  # (w,x,y,z) -> (x,y,z,w) for SciPy
    return Rotation.from_quat(xyzw).as_matrix().reshape(batch + (3, 3))


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """``(...,3,3)`` rotations -> ``(...,4)`` quaternions in ``(w,x,y,z)`` order.

    The sign is canonicalized to ``w >= 0`` so round-trips and dot products
    between nearby rotations stay continuous.
    """
    R = np.asarray(R, dtype=np.float64)
    batch = R.shape[:-2]
    xyzw = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat()
    q = np.roll(xyzw, 1, axis=-1)  # (x,y,z,w) -> (w,x,y,z)
    q = q * np.where(q[:, 0:1] < 0, -1.0, 1.0)
    return q.reshape(batch + (4,))


def yaw_of(T: np.ndarray) -> np.ndarray | float:
    """Heading of the frame's x-axis in the table plane: ``atan2(R[1,0], R[0,0])``."""
    T = np.asarray(T, dtype=np.float64)
    yaw = np.arctan2(T[..., 1, 0], T[..., 0, 0])
    return float(yaw) if yaw.ndim == 0 else yaw


def rot_z(theta: float | np.ndarray) -> np.ndarray:
    """Rotation about +z. Scalar -> ``(3,3)``; ``(N,)`` -> ``(N,3,3)``."""
    th = np.asarray(theta, dtype=np.float64)
    c, s = np.cos(th), np.sin(th)
    one, zero = np.ones_like(c), np.zeros_like(c)
    return np.stack([c, -s, zero, s, c, zero, zero, zero, one], axis=-1).reshape(th.shape + (3, 3))


def rot_z_T(theta: float | np.ndarray) -> np.ndarray:
    """``rot_z`` as a 4x4 transform with zero translation."""
    R = rot_z(theta)
    return make_T(R, np.zeros(R.shape[:-2] + (3,)))


def arkit_to_demo(T: np.ndarray) -> np.ndarray:
    """``T_demo_x = T_zup @ T_arkit_x`` (conventions.md section 2). Batch-safe."""
    T = np.asarray(T, dtype=np.float64)
    T_zup = make_T(R_ZUP, np.zeros(3))
    return T_zup @ T


def mirror_lr(T: np.ndarray) -> np.ndarray:
    """Mirror a `table`-frame pose across the robot's left-right plane.

    ``p' = M p``, ``R' = M R M`` with ``M = diag(1, -1, 1)``. ``M R M`` keeps the
    determinant at +1, so the result is still a proper rotation.
    """
    T = np.asarray(T, dtype=np.float64)
    return make_T(M_LR @ T[..., :3, :3] @ M_LR, (M_LR @ T[..., :3, 3, None])[..., 0])


def mj_cam_to_cv(R_table_cam_mj: np.ndarray) -> np.ndarray:
    """MuJoCo camera rotation (-z forward, +y up) -> OpenCV (+z forward, +y down)."""
    return np.asarray(R_table_cam_mj, dtype=np.float64) @ np.diag([1.0, -1.0, -1.0])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


def gripper_frame(
    wrist: np.ndarray,
    thumb_mcp: np.ndarray,
    index_mcp: np.ndarray,
    thumb_tip: np.ndarray,
    index_tip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """HumanEgo hand -> `tcp` construction (conventions.md section 2).

    All inputs are ``(N,3)`` positions **in the same frame** (we use `demo`).
    Returns ``(T_tcp (N,4,4), aperture (N,))``:

    - ``p = (thumbTip + indexTip) / 2``, ``a = ||thumbTip - indexTip||``
    - ``z_tcp = normalize(mid(thumbMCP, indexMCP) - wrist)`` (approach)
    - ``y_tcp = normalize(indexMCP - thumbMCP)``, Gram-Schmidt against z (jaw)
    - ``x_tcp = y_tcp x z_tcp``
    """
    wrist, thumb_mcp, index_mcp, thumb_tip, index_tip = (
        np.asarray(a, dtype=np.float64) for a in (wrist, thumb_mcp, index_mcp, thumb_tip, index_tip)
    )

    p = 0.5 * (thumb_tip + index_tip)
    aperture = np.linalg.norm(thumb_tip - index_tip, axis=-1)

    z = _unit(0.5 * (thumb_mcp + index_mcp) - wrist)
    j = _unit(index_mcp - thumb_mcp)
    y = _unit(j - np.sum(j * z, axis=-1, keepdims=True) * z)
    x = np.cross(y, z)

    R = np.stack([x, y, z], axis=-1)  # columns are the tcp axes in the input frame
    return make_T(R, p), aperture


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project ``(...,3,3)`` onto the nearest proper rotation via SVD.

    Used after smoothing, which averages rotation matrices elementwise and so
    leaves them slightly non-orthogonal.
    """
    R = np.asarray(R, dtype=np.float64)
    U, _, Vt = np.linalg.svd(R)
    det = np.linalg.det(U @ Vt)
    # Flip the last singular direction where the naive product is a reflection.
    U = U.copy()
    U[..., :, -1] *= np.sign(det)[..., None]
    return U @ Vt


def slerp(R0: np.ndarray, R1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation interpolation. ``R0``/``R1`` ``(N,3,3)``, ``u`` ``(N,)``."""
    q0 = mat_to_quat_wxyz(R0)
    q1 = mat_to_quat_wxyz(R1)
    # Pick the sign of q1 that gives the shortest arc (q and -q are the same rotation).
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(-1.0, 1.0)

    u = np.asarray(u, dtype=np.float64)[..., None]
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    # Near-parallel quaternions: fall back to lerp to avoid dividing by ~0.
    near = sin_theta < 1e-8
    w0 = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(near, 1.0, sin_theta))
    w1 = np.where(near, u, np.sin(u * theta) / np.where(near, 1.0, sin_theta))
    q = w0 * q0 + w1 * q1
    return quat_wxyz_to_mat(q / np.linalg.norm(q, axis=-1, keepdims=True))


def interp_poses(T0: np.ndarray, T1: np.ndarray, n: int) -> np.ndarray:
    """``n`` poses from ``T0`` to ``T1`` inclusive: linear position, slerp rotation."""
    u = np.linspace(0.0, 1.0, n)
    p = (1.0 - u)[:, None] * T0[:3, 3] + u[:, None] * T1[:3, 3]
    R = slerp(np.repeat(T0[None, :3, :3], n, 0), np.repeat(T1[None, :3, :3], n, 0), u)
    return make_T(R, p)


def resample_poses(t_src: np.ndarray, T_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Resample a pose trajectory onto ``t_dst``: linear position, slerp rotation."""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64).clip(t_src[0], t_src[-1])

    # Bracketing source indices and the fractional position between them.
    i1 = np.searchsorted(t_src, t_dst, side="right").clip(1, len(t_src) - 1)
    i0 = i1 - 1
    span = t_src[i1] - t_src[i0]
    u = np.where(span > 0, (t_dst - t_src[i0]) / np.where(span > 0, span, 1.0), 0.0)

    p = (1.0 - u)[:, None] * T_src[i0, :3, 3] + u[:, None] * T_src[i1, :3, 3]
    R = slerp(T_src[i0, :3, :3], T_src[i1, :3, :3], u)
    return make_T(R, p)


def pos_yaw_pose(T: np.ndarray) -> np.ndarray:
    """Reduce a pose to position + yaw, zeroing roll and pitch (anchors, section 8)."""
    T = np.asarray(T, dtype=np.float64)
    return make_T(rot_z(yaw_of(T)), T[..., :3, 3])


def rot_y(theta: float | np.ndarray) -> np.ndarray:
    """Rotation about +y by `theta` radians. Supports array input like :func:`rot_z`."""
    theta = np.asarray(theta, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack(
        [np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2
    )


def grasp_pose(pos: np.ndarray, yaw: float, tilt: float = 0.0) -> np.ndarray:
    """A near-top-down gripper pose, tilted `tilt` radians off vertical.

    `tilt = 0` is exactly :func:`top_down_pose`. The tilt is applied about the
    gripper's **own +y axis**, which is the jaw-closing axis, so:

    * the jaw axis stays horizontal — the jaws still straddle a cube's vertical
      faces, and both fingers tilt together rather than one dipping toward the
      table;
    * the approach axis leans by exactly `tilt` from `table` -z, within the
      vertical plane containing the gripper's x axis;
    * `yaw_of` is unchanged for small tilts, so the section 8 anchor math still
      sees the yaw it expects.

    A strictly vertical approach is reachable in only a thin slice of this arm's
    configuration space; allowing a few degrees of tilt widens it substantially
    and is harmless for grasping a cube. See `docs/conventions_proposals.md` P7.
    """
    return make_T(
        rot_z(yaw) @ np.diag([1.0, -1.0, -1.0]) @ rot_y(tilt),
        np.asarray(pos, dtype=np.float64),
    )


def top_down_pose(pos: np.ndarray, yaw: float) -> np.ndarray:
    """A top-down gripper pose in the `table` frame (conventions.md section 2).

    "Top-down grasp: `tcp` +z points along `table` -z". `yaw` is the pose's yaw in
    the section 3 sense, so ``yaw_of(top_down_pose(p, yaw)) == yaw`` — which is
    what the section 8 anchor math assumes when it aligns headings with
    :func:`yaw_of` and :func:`rot_z`.

    Built as ``rot_z(yaw) @ diag(1, -1, -1)``: the flip sends `tcp` +z to `table`
    -z while leaving `tcp` +x on `table` +x, and is right-handed (det = +1). The
    jaw-closing axis (`tcp` +y) therefore lies along `table` -y at ``yaw = 0``;
    the jaws are symmetric, so the sign carries no meaning.
    """
    R_flip = np.diag([1.0, -1.0, -1.0])
    return make_T(rot_z(yaw) @ R_flip, np.asarray(pos, dtype=np.float64))


def flatten_top_down(T: np.ndarray) -> np.ndarray:
    """Replace every pose's orientation with the top-down one of the same yaw.

    Keeps position and yaw, discards roll and pitch — :func:`pos_yaw_pose` for
    the *trajectory* rather than the anchor, expressed in the gripper convention
    of section 2 instead of the z-up one.

    Needed because a human's `tcp` +z (wrist -> MCP midpoint, section 2) is the
    palm's forward direction, which sits a median **71 deg** off `table` -z at the
    grasp: a human picks a cube up with the hand pointing forward and down, not
    straight down. Replaying that orientation on the robot drives the fingers into
    the table and out of the top-down reachable map. See
    `docs/conventions_proposals.md` P8.

    Safe to compose with section 8: the retarget rigid transform is a pure
    z-rotation plus translation, so flattening before it commutes with it, and
    `yaw_of` is preserved.
    """
    T = np.asarray(T, dtype=np.float64)
    return make_T(
        rot_z(yaw_of(T)) @ np.diag([1.0, -1.0, -1.0]), T[..., :3, 3]
    )


def spin_about_approach(T: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
    """Rotate poses about their **own** +z (approach) axis by `delta` radians.

    Post-multiplication, so the translation is untouched: ``T @ rot_z`` leaves
    ``T[:3, 3]`` exactly as it was. That is the whole point — the wrist's spin
    about its approach axis is the one gripper degree of freedom a top-down cube
    task does not constrain (away from the jaws closing), so this changes
    *only* that, and cannot disturb the position trajectory, the section 8
    heading alignment or the `(dist_xy, dz)` index.

    `delta` may be a scalar or one angle per pose, which is how a smooth ramp
    between two per-segment offsets is applied.
    """
    T = np.asarray(T, dtype=np.float64)
    R = rot_z(np.asarray(delta, dtype=np.float64))
    out = T.copy()
    out[..., :3, :3] = T[..., :3, :3] @ R
    return out


def pose_to_dict(T: np.ndarray) -> dict:
    """Serialize a pose as ``{"pos": [...], "quat_wxyz": [...]}`` (section 3)."""
    T = np.asarray(T, dtype=np.float64)
    return {
        "pos": T[:3, 3].tolist(),
        "quat_wxyz": mat_to_quat_wxyz(T[:3, :3]).tolist(),
    }


def pose_from_dict(d: dict) -> np.ndarray:
    """Inverse of :func:`pose_to_dict`."""
    return make_T(quat_wxyz_to_mat(np.asarray(d["quat_wxyz"])), np.asarray(d["pos"]))


def rot_angle(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray | float:
    """Geodesic angle in radians between two rotations."""
    R_rel = np.swapaxes(np.asarray(R_a, dtype=np.float64), -1, -2) @ np.asarray(R_b, dtype=np.float64)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    ang = np.arccos(((trace - 1.0) / 2.0).clip(-1.0, 1.0))
    return float(ang) if ang.ndim == 0 else ang
