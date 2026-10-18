"""The object-centric transform: source demo -> new scene (`conventions.md` section 8).

This is the MimicGen core. A source demo is a gripper-pose trajectory recorded in
its own `demo` frame around *some* object placement; this module re-anchors it
onto a *new* object and target in the `table` frame, so one human demonstration
becomes many feasible robot trajectories.

The five steps of section 8, in order, each a separate function so each can be
tested on its own:

1. :func:`mirror_demo` — left arm only (section 2's `M = diag(1, -1, 1)`).
2. :func:`anchor` — reduce a pose to position + yaw, zeroing roll and pitch.
3. :func:`heading_rotation` — rotate the demo about +z about the grasp anchor so
   its grasp -> release heading matches the scene's object -> target heading.
4. :func:`transform_segment` — `T_table_tgt @ inv_T(T_demo_src) @ T_demo_tcp[k]`.
5. :func:`bridge` — 0.5 s linear/slerp joins, gripper closed.

:func:`retarget` runs all of them and is what callers normally use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.transforms import (
    interp_poses,
    inv_T,
    make_T,
    mirror_lr,
    pos_yaw_pose,
    rot_z,
    yaw_of,
)

CUBE_YAW_PERIOD = np.pi / 2.0  # conventions.md section 2: "Yaw has 90 deg symmetry"
BRIDGE_S = 0.5  # conventions.md section 5: "Segment bridging | 0.5 s"


@dataclass(frozen=True)
class SourceDemo:
    """One source demo, as stored in `data/retargeted/*.npz` (section 7.2).

    `T_demo_tcp` is `(N, 4, 4)`; `grip` is `(N,)` with 1 = open, 0 = closed.
    `T_demo_obj0` is present only for teleop demos (`anchor_type = "measured"`).

    `grasp_anchor_z` overrides the z of the approach anchor only. It exists
    because `grasp_idx` is the frame at which the hand's *aperture* crossed its
    threshold, and that happens mid-descent rather than on the object — the
    measured hand speed there has median 0.211 m/s
    (`docs/conventions_proposals.md` P2). The hand therefore keeps descending
    afterwards, by a median of 6.7 mm and up to 173 mm, so anchoring the cube
    centre on that frame sends the rest of the trajectory *through the table*.
    Set by `common.demos.load_source_demo`; see proposal **P9**.
    """

    T_demo_tcp: np.ndarray
    grip: np.ndarray
    grasp_idx: int
    release_idx: int
    arm: str = "right"
    anchor_type: str = "grasp_proxy"
    T_demo_obj0: np.ndarray | None = None
    grasp_anchor_z: float | None = None

    def __post_init__(self) -> None:
        n = len(self.T_demo_tcp)
        if not 0 <= self.grasp_idx < self.release_idx < n:
            raise ValueError(
                f"need 0 <= grasp_idx < release_idx < {n}, got {self.grasp_idx}, {self.release_idx}"
            )
        if len(self.grip) != n:
            raise ValueError(f"grip has {len(self.grip)} samples, poses have {n}")
        if self.anchor_type not in ("grasp_proxy", "measured"):
            raise ValueError(f"unknown anchor_type {self.anchor_type!r}")
        if self.anchor_type == "measured" and self.T_demo_obj0 is None:
            raise ValueError("anchor_type 'measured' requires T_demo_obj0")

    @property
    def index(self) -> dict[str, float]:
        """The `(dist_xy, dz)` retrieval index of section 7.3.

        Measured between the two **anchors**, not the raw grasp and release
        frames, because the anchors are what the section 8 step 4 transform
        actually maps onto the scene: the index has to predict where the demo
        puts the object, and with `grasp_anchor_z` set those differ.
        """
        g, r = demo_anchors(self)
        delta = r[:3, 3] - g[:3, 3]
        return {"dist_xy": float(np.linalg.norm(delta[:2])), "dz": float(delta[2])}


# --- step 1: mirroring ------------------------------------------------------


def mirror_demo(demo: SourceDemo) -> SourceDemo:
    """Mirror a right-arm demo for the left arm. Must happen BEFORE anchoring."""
    if demo.arm == "left":
        raise ValueError("demo is already a left-arm demo")
    return SourceDemo(
        T_demo_tcp=mirror_lr(demo.T_demo_tcp),
        grip=demo.grip.copy(),
        grasp_idx=demo.grasp_idx,
        release_idx=demo.release_idx,
        arm="left",
        anchor_type=demo.anchor_type,
        T_demo_obj0=None if demo.T_demo_obj0 is None else mirror_lr(demo.T_demo_obj0),
    )


# --- step 2: anchors --------------------------------------------------------


def anchor(T: np.ndarray) -> np.ndarray:
    """Position + yaw only, roll and pitch zeroed (section 8 step 2)."""
    return pos_yaw_pose(T)


def demo_anchors(demo: SourceDemo) -> tuple[np.ndarray, np.ndarray]:
    """`(T_demo_grasp, T_demo_release)` anchors for a demo.

    EgoDex (`grasp_proxy`) uses the gripper pose at the grasp. Teleop
    (`measured`) uses the measured initial object pose for the approach anchor;
    both use the gripper pose at release for the transport anchor.
    """
    release = anchor(demo.T_demo_tcp[demo.release_idx])
    if demo.anchor_type == "measured":
        return anchor(demo.T_demo_obj0), release

    grasp = anchor(demo.T_demo_tcp[demo.grasp_idx])
    if demo.grasp_anchor_z is not None:
        grasp = grasp.copy()
        grasp[2, 3] = float(demo.grasp_anchor_z)
    return grasp, release


# --- step 3: global heading -------------------------------------------------


def heading_rotation(
    T_demo_grasp: np.ndarray,
    T_demo_release: np.ndarray,
    T_table_obj: np.ndarray,
    T_table_tgt: np.ndarray,
) -> np.ndarray:
    """A `table`-frame rotation about +z that aligns the demo's transport heading.

    Rotates about the **grasp anchor's position**, so the grasp point itself does
    not move. Returns a 4x4 to left-multiply onto every demo pose.

    If either heading is degenerate (the grasp and release are vertically
    stacked, so there is no horizontal direction), the identity is returned —
    rotating by an arbitrary angle would be worse than not rotating.
    """
    demo_vec = (np.asarray(T_demo_release)[:3, 3] - np.asarray(T_demo_grasp)[:3, 3])[:2]
    scene_vec = (np.asarray(T_table_tgt)[:3, 3] - np.asarray(T_table_obj)[:3, 3])[:2]

    eps = 1e-9
    if np.linalg.norm(demo_vec) < eps or np.linalg.norm(scene_vec) < eps:
        return np.eye(4, dtype=np.float64)

    delta = float(np.arctan2(scene_vec[1], scene_vec[0]) - np.arctan2(demo_vec[1], demo_vec[0]))
    pivot = np.asarray(T_demo_grasp, dtype=np.float64)[:3, 3]
    to_origin = make_T(np.eye(3), -pivot)
    back = make_T(np.eye(3), pivot)
    return back @ make_T(rot_z(delta), np.zeros(3)) @ to_origin


# --- step 4: segment transform ---------------------------------------------


def transform_segment(
    T_demo_tcp: np.ndarray, T_demo_src: np.ndarray, T_table_tgt: np.ndarray
) -> np.ndarray:
    """`T_table_tcp[k] = T_table_tgt @ inv_T(T_demo_src) @ T_demo_tcp[k]` (section 8 step 4)."""
    rigid = np.asarray(T_table_tgt, dtype=np.float64) @ inv_T(np.asarray(T_demo_src, dtype=np.float64))
    return rigid @ np.asarray(T_demo_tcp, dtype=np.float64)


def snap_yaw(target_yaw: float, reference_yaw: float, period: float = CUBE_YAW_PERIOD) -> float:
    """The `target_yaw + n * period` closest to `reference_yaw` (section 8 step 4).

    A cube's yaw is only defined up to 90 deg, so the approach target's yaw is
    snapped to whichever symmetric equivalent is nearest the demo's anchor yaw
    after heading alignment. Without this the gripper can be asked to rotate up
    to 45 deg more than necessary, for no gain.
    """
    n = np.round((reference_yaw - target_yaw) / period)
    return float(target_yaw + n * period)


# --- step 5: bridging -------------------------------------------------------


def bridge(
    T_from: np.ndarray, T_to: np.ndarray, rate_hz: float, duration_s: float = BRIDGE_S
) -> np.ndarray:
    """A `(n, 4, 4)` linear-position / slerp-rotation join between two poses."""
    n = max(int(round(float(duration_s) * float(rate_hz))), 2)
    return interp_poses(np.asarray(T_from, dtype=np.float64), np.asarray(T_to, dtype=np.float64), n)


# --- the whole thing --------------------------------------------------------


@dataclass
class Retargeted:
    """A demo re-anchored onto a new scene, ready for IK."""

    T_table_tcp: np.ndarray  # (M, 4, 4)
    grip: np.ndarray  # (M,)
    segments: dict[str, tuple[int, int]]  # label -> [start, end) in the output
    heading_delta: float


def retarget(
    demo: SourceDemo,
    T_table_obj: np.ndarray,
    T_table_tgt: np.ndarray,
    rate_hz: float,
    T_table_tcp_now: np.ndarray | None = None,
    snap_object_yaw: bool = True,
) -> Retargeted:
    """Run section 8 end to end for one demo and one scene.

    `T_table_obj` is the detected object pose (the approach target) and
    `T_table_tgt` the place target (bin + clearance, or the spot). Pass
    `T_table_tcp_now` to prepend the section 8 step 5 bridge from where the
    gripper currently is.

    The approach segment is `[0, grasp_idx]` and the transport segment is
    `[grasp_idx, release_idx]`, matching the library schema in section 7.3.
    """
    for name, T in (("T_table_obj", T_table_obj), ("T_table_tgt", T_table_tgt)):
        if np.asarray(T).shape != (4, 4):
            raise ValueError(f"{name} must be 4x4, got {np.asarray(T).shape}")

    # Steps 2 and 3: anchor, then align the transport heading about the grasp.
    T_grasp, T_release = demo_anchors(demo)
    align = heading_rotation(T_grasp, T_release, T_table_obj, T_table_tgt)
    heading_delta = float(yaw_of(align))

    T_aligned = align @ np.asarray(demo.T_demo_tcp, dtype=np.float64)
    T_grasp_a = anchor(align @ T_grasp)
    T_release_a = anchor(align @ T_release)

    # Step 4, approach: target is the detected object, its yaw snapped to the
    # symmetric equivalent nearest the aligned demo anchor.
    obj_tgt = anchor(T_table_obj)
    if snap_object_yaw:
        obj_tgt = make_T(
            rot_z(snap_yaw(float(yaw_of(obj_tgt)), float(yaw_of(T_grasp_a)))),
            np.asarray(T_table_obj, dtype=np.float64)[:3, 3],
        )

    approach = transform_segment(T_aligned[: demo.grasp_idx + 1], T_grasp_a, obj_tgt)
    transport = transform_segment(
        T_aligned[demo.grasp_idx : demo.release_idx + 1], T_release_a, anchor(T_table_tgt)
    )

    pieces: list[np.ndarray] = []
    grips: list[np.ndarray] = []
    segments: dict[str, tuple[int, int]] = {}

    # Step 5: bridge from the current pose to the approach start, gripper open —
    # section 8 says the gripper is closed for bridges, but that applies to the
    # approach -> transport join; nothing is held yet on the way in.
    if T_table_tcp_now is not None:
        lead = bridge(T_table_tcp_now, approach[0], rate_hz)
        pieces.append(lead)
        grips.append(np.full(len(lead), float(demo.grip[0])))
        segments["bridge_in"] = (0, len(lead))

    start = sum(len(p) for p in pieces)
    pieces.append(approach)
    grips.append(np.asarray(demo.grip[: demo.grasp_idx + 1], dtype=np.float64))
    segments["approach"] = (start, start + len(approach))

    # The two segments are re-anchored to different targets, so the approach's
    # last pose and the transport's first pose are the same demo frame mapped two
    # different ways: they do not coincide and must be bridged, gripper closed.
    join = bridge(approach[-1], transport[0], rate_hz)
    start = sum(len(p) for p in pieces)
    pieces.append(join)
    grips.append(np.zeros(len(join)))
    segments["bridge_mid"] = (start, start + len(join))

    start = sum(len(p) for p in pieces)
    pieces.append(transport)
    grips.append(np.asarray(demo.grip[demo.grasp_idx : demo.release_idx + 1], dtype=np.float64))
    segments["transport"] = (start, start + len(transport))

    return Retargeted(
        T_table_tcp=np.concatenate(pieces, axis=0),
        grip=np.concatenate(grips, axis=0),
        segments=segments,
        heading_delta=heading_delta,
    )


# --- step 6: selection ------------------------------------------------------


def select(
    index: np.ndarray, query: np.ndarray, mode: str = "retrieval", k: int = 5, rng=None
) -> int:
    """Pick a library entry by `(dist_xy, dz)` (section 8 step 6).

    `retrieval` takes the nearest; `datagen` samples uniformly among the `k`
    nearest, which is what gives generated datasets their variety.
    """
    index = np.asarray(index, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(index - np.asarray(query, dtype=np.float64).reshape(1, 2), axis=1)
    if mode == "retrieval":
        return int(np.argmin(d))
    if mode == "datagen":
        rng = np.random.default_rng() if rng is None else rng
        return int(rng.choice(np.argsort(d)[: max(min(int(k), len(d)), 1)]))
    raise ValueError(f"mode must be 'retrieval' or 'datagen', got {mode!r}")
