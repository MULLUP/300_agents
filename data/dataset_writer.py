"""Write recorded episodes as a LeRobot v3.0 dataset — the project's deliverable.

This is the only module that knows the on-disk dataset format. Everything
upstream deals in :class:`sim.sim_robot.Episode`.

**The schema here was read out of the installed `lerobot` 0.4.4 and
`SmolVLAConfig`, not from `conventions.md` section 7.4, which is wrong about the
image size and says nothing about the token or episode-length limits.** Facts
that cost something to rediscover:

* `timestamp`, `frame_index`, `episode_index`, `index` and `task_index` are
  added by `add_frame` itself. Writing them raises.
* `task` is passed per frame and popped; it is not a feature.
* `action` is **absolute** joint targets. Delta actions raise
  `NotImplementedError` in this version.
* Values are written **raw**. SmolVLA normalises state and action with MEAN_STD
  and images with IDENTITY, computing the statistics from the dataset itself, so
  normalising here would be applied twice.
* `LeRobotDataset` is **not** multi-process safe. Each worker writes its own
  shard and :func:`merge_shards` combines them afterwards.

**Reading the result back on this machine needs `video_backend="pyav"`:**

    LeRobotDataset("local/<name>", root=..., video_backend="pyav")

LeRobot defaults to `torchcodec`, whose bundled library links `libavutil.so.56`
(ffmpeg 4). This box has ffmpeg 6 (`libavutil.so.58`), so the default backend
fails to load with an `OSError` about `libtorchcodec_core4.so` — at *read* time,
long after the data is written, and the written files are fine. PyAV 15.1 is
installed and bundles its own ffmpeg. Pass this on with the dataset; it is an
environment quirk, not a property of the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.joints import JointIndex  # noqa: E402
from sim.sim_robot import Episode  # noqa: E402

ROBOT_TYPE = "openarm_bimanual"


def dataset_root(name: str) -> Path:
    return config.path("data", "lerobot", name)


def state_slots(index: JointIndex, side: str) -> list[int]:
    """The 8 columns of a 16-slot joint vector that the dataset records.

    The arm's 7 joints then its gripper, for one side. The simulation has 16
    actuators because the upstream model is bimanual, and only one arm performs
    this task — writing all 16 would hand the policy 8 constant columns.

    Taken from `JointIndex` rather than written out, per the handoff rule: joint
    order is a property of the loaded model, never a literal.
    """
    return [*index.arm_slots[side], index.gripper_slot[side]]


def features(image_size: int, camera: str, n_joints: int) -> dict:
    """The LeRobot feature spec for one overhead camera and one arm.

    `observation.state` and `action` are `(n_joints,)` = 8. SmolVLA pads state
    to `max_state_dim` 32 internally, so 8 is fine and 32 would waste bytes.
    """
    joint = {"dtype": "float32", "shape": (n_joints,), "names": None}
    return {
        "observation.state": dict(joint),
        "action": dict(joint),
        f"observation.images.{camera}": {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
    }


class ShardWriter:
    """One process's dataset shard. Append successful episodes, then `close()`.

    Not thread- or process-safe; that is the point of sharding.
    """

    def __init__(self, name: str, root: Path | None = None, side: str = "right"):
        ds = config.skills()["dataset"]
        self.camera = str(ds["camera"])
        self.task = str(ds["task"])
        self.min_frames = int(ds["min_frames"])
        self.image_size = int(ds["image_size"])
        self.side = side

        # One rate end to end: `exec.rate_hz` is the control rate, one control
        # cycle is one frame, so a separate dataset fps could only disagree.
        # LeRobot types fps as int.
        rate = float(config.skills()["exec"]["rate_hz"])
        if rate != int(rate):
            raise ValueError(f"exec.rate_hz must be a whole number of Hz for LeRobot, got {rate}")
        self.fps = int(rate)

        self.name = name
        self.root = Path(root) if root is not None else dataset_root(name)
        self._ds = None
        self.n_written = 0
        self.n_too_short = 0

    def _dataset(self, n_joints: int):
        """Created on the first episode, so an empty shard leaves nothing behind."""
        if self._ds is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._ds = LeRobotDataset.create(
                repo_id=f"local/{self.name}",
                fps=self.fps,
                features=features(self.image_size, self.camera, n_joints),
                root=self.root,
                robot_type=ROBOT_TYPE,
                use_videos=bool(config.skills()["dataset"]["use_videos"]),
            )
        return self._ds

    def write(self, episode: Episode, index: JointIndex) -> bool:
        """Append one episode. Returns False if it was rejected as too short.

        SmolVLA's `chunk_size` is 50: a training sample needs a 50-step action
        horizon, so a shorter episode cannot produce one and is dead weight in
        the dataset rather than merely small.
        """
        if episode.jpegs is None:
            raise ValueError("episode has no frames — record it with a camera")
        if len(episode) < self.min_frames:
            self.n_too_short += 1
            return False

        slots = state_slots(index, self.side)
        state = np.asarray(episode.state[:, slots], dtype=np.float32)
        action = np.asarray(episode.action[:, slots], dtype=np.float32)
        frames = episode.frames()

        ds = self._dataset(len(slots))
        for i in range(len(episode)):
            ds.add_frame(
                {
                    "observation.state": state[i],
                    "action": action[i],
                    f"observation.images.{self.camera}": frames[i],
                    "task": self.task,
                }
            )
        ds.save_episode()
        self.n_written += 1
        return True

    def close(self) -> None:
        """Flush buffered metadata. **A shard is unreadable without this.**

        `save_episode` writes the parquet and video, but episode metadata goes
        through a buffer `metadata_buffer_size` (10) deep that only reaches
        `meta/episodes` on `finalize()`. A shard of fewer than 10 episodes
        therefore lands on disk looking complete — data/, videos/, info.json,
        stats.json all present — while `LeRobotDatasetMetadata` cannot load it,
        and falls back to resolving the repo id against the HuggingFace hub.
        The error surfaces as a 401, hundreds of trials after the real mistake.

        Idempotent, because it runs from an `atexit` hook in the generator's
        workers and may also be called directly.
        """
        if self._ds is not None:
            self._ds.finalize()
            self._ds = None


def merge_shards(shard_names: list[str], out_name: str, roots: list[Path] | None = None) -> Path:
    """Combine per-worker shards into the single dataset we hand off.

    Uses LeRobot's own `aggregate_datasets`, which renumbers episode and frame
    indices and recomputes statistics — all three of which are easy to get
    subtly wrong by hand, and none of which fail loudly when you do.
    """
    from lerobot.datasets.aggregate import aggregate_datasets

    roots = roots if roots is not None else [dataset_root(n) for n in shard_names]
    out = dataset_root(out_name)
    aggregate_datasets(
        repo_ids=[f"local/{n}" for n in shard_names],
        aggr_repo_id=f"local/{out_name}",
        roots=roots,
        aggr_root=out,
    )
    return out
