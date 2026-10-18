# data

Track A source data: EgoDex `basic_pick_place` → retargeted source demos.

| | |
|---|---|
| `conventions_version` | 2 |
| Raw source | <https://ml-site.cdn-apple.com/datasets/egodex/test.zip>, task `basic_pick_place` |
| License | CC-BY-NC-ND — **research only, never redistribute the raw data** |

## Layout

| Path | Contents | Committed |
|---|---|---|
| `egodex_raw/` | 277 unmodified episode `.hdf5` files, plus 3 `.mp4` for the pitch | no (`.gitignore`) |
| `retargeted/egodex_<idx>.npz` | Source demos, `conventions.md` section 7.2 | no |
| `retargeted/summary.json` | Filter counts per reason | yes |
| `retargeted/feasibility.json` | IK feasibility per demo | yes |
| `retargeted/plots/` | Evidence plots | selected |
| `library/egodex_candidates_v1.json` | Library candidates before physics validation | yes |
| `library/egodex_v1.json` | Validated library, `conventions.md` section 7.3 | yes |

## Scripts

| Script | Purpose |
|---|---|
| `download_egodex.py` | Stream the remote zip, extract only `basic_pick_place` members |
| `egodex_io.py` | HDF5 loader; the only place that touches the raw files |
| `inspect_joints.py` | MCP joint identification (evidence below) |
| `retarget.py` | `arkit` → `demo`, hand → `tcp`, grasp/release, filters, 50 Hz resample |
| `build_library.py` | Segments, yaw-only anchors, retrieval index |

## Download

Step 1 ran before this handoff. Current state on disk:

```
277 .hdf5 files, 161 MB total  (data/egodex_raw/)
```

That is the full `basic_pick_place` test split — above the 150-episode escalation
floor and far under the 5 GB download cap, so no escalation was needed.

## MCP joint identification (step 2, binding per `conventions.md` section 2)

**Answer:**

| Role | ARKit joint name |
|---|---|
| wrist | `rightHand` |
| **thumb MCP** | **`rightThumbIntermediateBase`** |
| **index MCP** | **`rightIndexFingerKnuckle`** |
| thumb tip | `rightThumbTip` |
| index tip | `rightIndexFingerTip` |

This confirms the expectation recorded in the handoff. The five roles are encoded
once, in `egodex_io.TCP_JOINT_ROLES`, and consumed by `transforms.gripper_frame`.

**Evidence** — median wrist-to-joint distance, mean over 5 episodes
(`uv run python data/inspect_joints.py`):

| Chain position from wrist | Thumb chain | | Index chain | |
|---|---|---|---|---|
| | joint | dist | joint | dist |
| 1 | `ThumbKnuckle` | 3.21 cm | `IndexFingerMetacarpal` | 2.99 cm |
| **2 (MCP)** | **`ThumbIntermediateBase`** | **7.78 cm** | **`IndexFingerKnuckle`** | **9.20 cm** |
| 3 | `ThumbIntermediateTip` | 10.51 cm | `IndexFingerIntermediateBase` | 12.42 cm |
| 4 | `ThumbTip` | 13.21 cm | `IndexFingerIntermediateTip` | 13.96 cm |
| 5 | — | — | `IndexFingerTip` | 15.42 cm |

Three independent facts pin the answer down:

1. **Monotonicity.** Wrist distance increases strictly along both chains in all 5
   episodes, so ARKit lists each chain in order outward from the wrist and
   "second from the wrist" is unambiguous. `inspect_joints.py` exits non-zero if
   this ever fails.
2. **Pairing by distance.** The *first* joints of the two chains sit together at
   3.21 / 2.99 cm — that is the hand base (CMC / carpometacarpal level), not the
   knuckle line. The *second* joints sit together at 7.78 / 9.20 cm, only 1.42 cm
   apart: this is the knuckle line, i.e. the MCPs. Picking `ThumbKnuckle` instead
   would pair a 3.21 cm joint against a 9.20 cm one and tilt the jaw axis badly.
3. **Naming.** Despite the name, `ThumbKnuckle` is the chain's *base*; ARKit's
   thumb has no metacarpal entry, so its chain is one joint shorter than the
   index chain and the names are offset by one.

Plot: `retargeted/plots/joint_identification.png` — one frame of episode 0 in the
`demo` frame, thumb and index chains drawn, the two inferred MCPs in red.

## Notes on the raw data

- `basic_pick_place` is **not** cube-specific: episode descriptions include
  staplers, bread, bowls. We use only the hand motion, never the object, so this
  does not matter — but it is why the grasp anchor is a `grasp_proxy` rather than
  a measured object pose (`conventions.md` section 8).
- Episode length varies widely (126–307 frames at 30 Hz in the first 5 alone).
- `arkit` origin is **not consistent across episodes**, so absolute positions are
  meaningless across demos. Only the anchored, relative motion carries over.
- Confidences are present for every joint in the episodes checked (min ~0.88 on
  the wrist). `retarget.py` still handles their absence.
