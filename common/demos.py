"""Loading `data/retargeted/*.npz` source demos into :class:`common.mimic.SourceDemo`.

One place that knows the section 7.2 npz schema, so its consumers —
`data/generate_dataset.py` and the optional `sim/feasibility.py` diagnostic —
cannot drift apart.

The one non-obvious thing here is :data:`ORIENTATION_MODES`. See
:func:`common.transforms.flatten_top_down` and
`docs/conventions_proposals.md` P8.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from common import config
from common.mimic import SourceDemo
from common.segment import descent_window_end
from common.transforms import flatten_top_down

ORIENTATION_MODES = ("top_down", "replay")


def demo_dir() -> Path:
    return config.path("data", "retargeted")


def demo_paths(source: str = "egodex") -> list[Path]:
    """Every source demo of `source`, ordered by episode then closure index.

    `sorted()` on the filenames gives `egodex_10` before `egodex_2`, which would
    make library ids depend on string order for no reason. Stems are
    `{source}_{episode}_{closure}` (one episode can yield several demos — see
    `data/retarget.py`), so both trailing numbers are sorted numerically and an
    episode's closures stay adjacent and in time order.
    """

    def key(p: Path) -> tuple[int, int, str]:
        m = re.search(r"_(\d+)(?:_(\d+))?$", p.stem)
        if m is None:
            return (-1, -1, p.stem)
        return (int(m.group(1)), int(m.group(2) or 0), p.stem)

    return sorted(demo_dir().glob(f"{source}_*.npz"), key=key)


def orientation_mode() -> str:
    mode = str(config.skills()["library"]["orientation"])
    if mode not in ORIENTATION_MODES:
        raise ValueError(f"library.orientation must be one of {ORIENTATION_MODES}, got {mode!r}")
    return mode


def demo_id(path: Path) -> str:
    """The library entry id for a source demo: its file stem (section 12)."""
    return Path(path).stem


def trim_approach(demo: SourceDemo, max_path_m: float, min_samples: int = 2) -> SourceDemo:
    """Drop the front of the approach, keeping at most `max_path_m` of tcp path.

    A source demo's approach starts wherever the human's hand happened to be:
    measured over the 106 kept demos, the straight-line start -> grasp distance
    has median **0.179 m** and reaches **0.63 m**, and 58% exceed 0.15 m. The
    arm's usable top-down workspace is only ~25 cm across
    (`AGENT_HANDOFF.md` section 11a), so re-anchoring a long lead-in onto the cube
    puts its first poses outside the workspace every time — `start_unreachable`
    was 5 of the first 8 demos screened.

    Nothing is lost: conventions.md section 8 step 5 already bridges from
    wherever the gripper is to the approach start, so the lead-in is exactly the
    part the bridge replaces. What the library is for is the final descent and
    the transport.

    Path length, not straight-line distance, so a demo whose hand circled around
    before descending is trimmed by how far it travelled.
    """
    T = demo.T_demo_tcp
    g = demo.grasp_idx
    if g < min_samples:
        return demo

    step = np.linalg.norm(np.diff(T[: g + 1, :3, 3], axis=0), axis=1)
    # Cumulative path length measured backwards from the grasp.
    back = np.concatenate([[0.0], np.cumsum(step[::-1])])[::-1]
    keep = int(np.argmax(back <= float(max_path_m)))
    keep = min(keep, g - (min_samples - 1))
    if keep <= 0:
        return demo

    return SourceDemo(
        T_demo_tcp=T[keep:],
        grip=demo.grip[keep:],
        grasp_idx=g - keep,
        release_idx=demo.release_idx - keep,
        arm=demo.arm,
        anchor_type=demo.anchor_type,
        T_demo_obj0=demo.T_demo_obj0,
    )


def load_source_demo(path: Path, orientation: str | None = None) -> SourceDemo:
    """Read one `data/retargeted/*.npz` into a :class:`SourceDemo`.

    `orientation` defaults to `config/skills.yaml` `library.orientation`:

    * `top_down` — keep each pose's position and yaw, force the approach axis
      straight down. What the robot can actually execute.
    * `replay` — use the recorded human wrist orientation verbatim. Kept so the
      cost of the choice stays measurable (`sim/feasibility.py --orientation`).
    """
    mode = orientation_mode() if orientation is None else orientation
    if mode not in ORIENTATION_MODES:
        raise ValueError(f"orientation must be one of {ORIENTATION_MODES}, got {mode!r}")

    with np.load(path, allow_pickle=False) as d:
        T = np.asarray(d["T_demo_tcp"], dtype=np.float64)
        demo = SourceDemo(
            T_demo_tcp=flatten_top_down(T) if mode == "top_down" else T,
            grip=np.asarray(d["grip"], dtype=np.float64),
            grasp_idx=int(d["grasp_idx"]),
            release_idx=int(d["release_idx"]),
            arm=str(d["arm"]),
            anchor_type="grasp_proxy",
        )
    demo = trim_approach(demo, float(config.skills()["library"]["approach_lead_m"]))
    return with_grasp_anchor_z(demo)


def with_grasp_anchor_z(demo: SourceDemo, window_s: float | None = None) -> SourceDemo:
    """Lower the approach anchor's z onto the lowest point *near the grasp*.

    `grasp_idx` is where the hand's aperture crossed its threshold, which is
    mid-descent, so the hand keeps going down afterwards — median 6.7 mm, p90
    84 mm, max 173 mm over the 106 kept demos. Anchoring the cube *centre* on
    that frame therefore maps the rest of the descent below the table: 24 of 106
    demos were rejected as `tcp_below_table` before this.

    The window matters, and getting it wrong is worse than not doing this at all.
    Taking the minimum over the whole kept span `[0, release_idx]` picks up dips
    that belong to the *carry*, which anchors the commanded grasp BELOW the cube:
    the fingers then close on the table instead of the cube and the cube is left
    untouched at its spawn pose — observed on 3 of 6 physics trials. The window
    ends `window_s` after the grasp, which is long enough for the descent to
    bottom out and short enough to exclude the carry.

    It also moves `dz` toward what the scene needs, because the lift is measured
    from a lower starting point. See `docs/conventions_proposals.md` P9.
    """
    if window_s is None:
        window_s = float(config.skills()["library"]["grasp_window_s"])
    rate_hz = float(config.skills()["retarget"]["fps_out"])
    end = descent_window_end(demo.grasp_idx, demo.release_idx, window_s, rate_hz)
    z = demo.T_demo_tcp[:end, 2, 3]
    return SourceDemo(
        T_demo_tcp=demo.T_demo_tcp,
        grip=demo.grip,
        grasp_idx=demo.grasp_idx,
        release_idx=demo.release_idx,
        arm=demo.arm,
        anchor_type=demo.anchor_type,
        T_demo_obj0=demo.T_demo_obj0,
        grasp_anchor_z=float(z.min()),
    )


def load_metadata(path: Path) -> dict:
    """The provenance fields the library schema (section 7.3) has to carry over."""
    with np.load(path, allow_pickle=False) as d:
        return {
            "id": demo_id(path),
            "source": str(d["source"]),
            "source_path": str(d["source_path"]),
            "arm": str(d["arm"]),
            "grasp_idx": int(d["grasp_idx"]),
            "release_idx": int(d["release_idx"]),
            "n_samples": int(len(d["grip"])),
            "conf_min": float(d["conf_min"]),
        }
