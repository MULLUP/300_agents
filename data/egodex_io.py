"""Read EgoDex ``basic_pick_place`` episodes out of their HDF5 files.

One episode is one HDF5 file: ``transforms/<joint>`` holds an ``(N,4,4)`` pose
track per hand joint in the `arkit` frame, ``confidences/<joint>`` an ``(N,)``
tracking confidence, and the file attributes carry the language description.
This module only reads; the `arkit` -> `demo` conversion happens in
`data/retarget.py` via `common/transforms.arkit_to_demo`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

EGODEX_FPS = 30.0

# Joints we read, per hand. The names are ARKit's; see data/README.md for which
# of them we treat as the MCPs.
HAND_JOINTS = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerMetacarpal",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
)

# The five joints the HumanEgo tcp construction actually consumes
# (conventions.md section 2). Keys are our roles, values are ARKit suffixes.
TCP_JOINT_ROLES = {
    "wrist": "Hand",
    "thumb_mcp": "ThumbIntermediateBase",
    "index_mcp": "IndexFingerKnuckle",
    "thumb_tip": "ThumbTip",
    "index_tip": "IndexFingerTip",
}


def joint_names(side: str = "right") -> list[str]:
    """Full ARKit dataset names for one hand, e.g. ``rightThumbTip``."""
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    return [f"{side}{j}" for j in HAND_JOINTS]


def load_episode(path: str | Path, side: str = "right") -> dict:
    """Load one episode's hand tracks.

    Returns a dict with:
      ``transforms``  {joint name -> (N,4,4) float64, `arkit` frame}
      ``confidences`` {joint name -> (N,) float64} or ``None`` if absent
      ``llm_description`` str (may be ``""``)
      ``N``           int, number of frames
      ``fps``         float, 30.0
      ``t``           (N,) float64 seconds from the episode start
      ``source_path`` str
      ``arm``         str, the requested side
      ``intrinsic``   (3,3) float64 or ``None``
      ``attrs``       the raw HDF5 attributes, for provenance
    """
    path = Path(path)
    names = joint_names(side)

    with h5py.File(path, "r") as f:
        missing = [n for n in names if f"transforms/{n}" not in f]
        if missing:
            raise KeyError(f"{path.name}: missing transforms {missing}")

        transforms = {n: np.asarray(f[f"transforms/{n}"][()], dtype=np.float64) for n in names}

        if all(f"confidences/{n}" in f for n in names):
            confidences = {n: np.asarray(f[f"confidences/{n}"][()], dtype=np.float64) for n in names}
        else:
            confidences = None

        intrinsic = (
            np.asarray(f["camera/intrinsic"][()], dtype=np.float64)
            if "camera/intrinsic" in f
            else None
        )
        attrs = {k: _plain(v) for k, v in f.attrs.items()}

    n = int(next(iter(transforms.values())).shape[0])
    if any(t.shape != (n, 4, 4) for t in transforms.values()):
        shapes = {k: v.shape for k, v in transforms.items()}
        raise ValueError(f"{path.name}: ragged transform shapes {shapes}")

    return {
        "transforms": transforms,
        "confidences": confidences,
        "llm_description": str(attrs.get("llm_description", "") or ""),
        "N": n,
        "fps": EGODEX_FPS,
        "t": np.arange(n, dtype=np.float64) / EGODEX_FPS,
        "source_path": str(path),
        "arm": side,
        "intrinsic": intrinsic,
        "attrs": attrs,
    }


def tcp_inputs(episode: dict) -> dict[str, np.ndarray]:
    """The five ``(N,3)`` positions the tcp construction needs, by role.

    Works on the dict from :func:`load_episode` whether its transforms are still
    in `arkit` or have already been converted to `demo`.
    """
    side = episode["arm"]
    return {
        role: episode["transforms"][f"{side}{suffix}"][:, :3, 3]
        for role, suffix in TCP_JOINT_ROLES.items()
    }


def tcp_conf_min(episode: dict) -> float:
    """Lowest confidence over the joints the tcp construction uses, or ``nan``."""
    conf = episode["confidences"]
    if conf is None:
        return float("nan")
    side = episode["arm"]
    used = [conf[f"{side}{suffix}"] for suffix in TCP_JOINT_ROLES.values()]
    return float(np.min(np.stack(used)))


def episode_paths(raw_dir: str | Path) -> list[Path]:
    """Episode files sorted numerically, so ``2.hdf5`` precedes ``10.hdf5``."""
    paths = list(Path(raw_dir).glob("*.hdf5"))
    return sorted(paths, key=lambda p: (len(p.stem), p.stem))


def episode_index(path: str | Path) -> str:
    """The episode index used in output filenames (``.../42.hdf5`` -> ``"42"``)."""
    return Path(path).stem


def _plain(v):
    """Turn numpy/bytes HDF5 attribute values into JSON-serializable Python."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v
