"""Generate the SmolVLA training dataset: run demos in physics, keep what works.

This replaces the screen-then-verdict shape of `sim/feasibility.py` (step 6) and
`sim/validate_library.py` (step 9). Both of those did exactly this work — retarget
a source demo onto a sampled cube, search the wrist angles, solve IK, execute —
and then threw the trajectory away and recorded a boolean or a rate. A generator
does the same work once and **keeps the episode whenever it succeeds**, which is
MimicGen as actually specified: generate many, keep the successes.

Because nothing downstream has to decide what to pass on, the thresholds those
stages needed (`library.max_index_offset`, `validation.min_success_rate`, gate
G4's "50 entries") have nothing left to gate. Every failed trial is logged with
its **full** reason string and the cube's **final pose**, which is what makes the
dominant failure diagnosable instead of merely countable.

    uv run python data/generate_dataset.py --trials 4 --limit 8     # a taste
    uv run python data/generate_dataset.py                          # the real run

Needs a GL context: `MUJOCO_GL=egl` on this machine (`osmesa` is a fallback).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from multiprocessing import Manager, Pool, util
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, demos  # noqa: E402
from common.ik import IKSolver  # noqa: E402
from data.dataset_writer import ShardWriter, dataset_root, merge_shards  # noqa: E402
from sim.execute import run_demo  # noqa: E402
from sim.sim_robot import SimRobot  # noqa: E402

FEASIBILITY = config.path("sim", "plots", "feasibility_egodex_top_down.json")

_W: dict = {}


def sample_scenes(n: int, seed: int) -> list[tuple[list[float], float]]:
    """`n` INDEPENDENT (cube_xy, cube_yaw) draws — one per trial, not per demo.

    Reusing one set across demos gave the whole dataset 20 distinct cube poses.
    Yaw spans +/- 45 deg because a cube is 90 deg-symmetric.
    """
    scene = config.skills()["scene"]
    centre = np.asarray(scene["pick_zone"]["center"], dtype=np.float64)
    size = np.asarray(scene["pick_zone"]["size"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return [
        ((centre + rng.uniform(-0.5, 0.5, 2) * size).tolist(), float(rng.uniform(-np.pi / 4, np.pi / 4)))
        for _ in range(n)
    ]


def source_demo_ids(limit: int | None = None, feasible_only: bool = False) -> list[str]:
    """Every retargeted source demo, straight from `data/retargeted/`.

    **Deliberately not filtered.** The pipeline used to run this same work twice
    before here — `sim/feasibility.py` to record a boolean, `sim/validate_library.py`
    to record a rate — and each needed a threshold to decide what to pass on.
    Physics is the only honest judge of whether a demo works on a given cube, and
    it is about to run anyway, so a demo that fails costs one trial and is logged
    with its reason. A demo wrongly rejected by a screen costs all of its trials
    and is invisible.

    `feasible_only` is an opt-in speed-up for a quick run, not the default: it
    reuses a previous feasibility screen, which is a nominal-scene approximation
    of the question this script answers exactly.
    """
    paths = sorted(config.path("data", "retargeted").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            "data/retargeted/ has no demos — run `uv run python data/retarget.py` first"
        )
    ids = [p.stem for p in paths]

    if feasible_only:
        if not FEASIBILITY.exists():
            raise FileNotFoundError(
                f"{FEASIBILITY} is missing — run `uv run python sim/feasibility.py`, "
                "or drop --feasible-only"
            )
        keep = {d["id"] for d in json.loads(FEASIBILITY.read_text())["demos"] if d.get("feasible")}
        ids = [i for i in ids if i in keep]

    return ids[:limit] if limit else ids


def quiet_subprocess_logging() -> None:
    """Silence the per-episode encoder and HF-datasets chatter.

    `save_episode` runs a PyAV encode and a `datasets` map, each of which writes
    its own progress to the terminal. Interleaved across workers they bury the
    run's own progress bar.
    """
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass
    try:
        import datasets

        datasets.disable_progress_bars()
    except Exception:
        pass


def _init_worker(shard_prefix: str, side: str, base: Path | None = None, stop=None) -> None:
    """One robot, solver and shard per process.

    `render=True` is required: the headless scene has its visual geometry
    stripped and cannot be photographed. It costs ~1.9 GB of RSS per worker
    against ~0.26 GB, which is what caps the worker count — see `--processes`.
    """
    quiet_subprocess_logging()
    # Ignore SIGINT in workers: Ctrl-C must be handled by the parent only, so it
    # can drain and flush rather than killing mid-episode.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _W["stop"] = stop
    _W["robot"] = SimRobot(render=True)
    _W["solver"] = IKSolver()
    _W["demos"] = {}
    _W["side"] = side
    _W["writer"] = ShardWriter(f"{shard_prefix}_w{os.getpid()}", side=side, base=base)

    # The shard lives and dies inside this process, so the parent has no moment
    # at which it could flush it. Without this the buffered episode metadata is
    # never written and every shard is silently unreadable — see
    # `ShardWriter.close`.
    #
    # `multiprocessing.util.Finalize`, NOT `atexit`: a pool worker ends in
    # `BaseProcess._bootstrap`, which calls `util._exit_function` and then
    # `os._exit()`. `os._exit` skips `atexit` entirely, so an atexit hook here
    # runs never — the shards came out exactly as unreadable as with no hook at
    # all. `util.Finalize` registers into the handler that _does_ run.
    util.Finalize(_W["writer"], _W["writer"].close, exitpriority=10)


def _demo(entry_id: str):
    """The loaded source demo, cached per process (the npz re-read is not free)."""
    cache = _W["demos"]
    if entry_id not in cache:
        cache[entry_id] = demos.load_source_demo(
            config.path("data", "retargeted", f"{entry_id}.npz")
        )
    return cache[entry_id]


def _trial(task: tuple[str, list[float], float, int]) -> dict:
    """Run one (demo, scene) pair and, if it works, write the episode.

    The episode never crosses the process boundary — even JPEG-buffered it is
    tens of megabytes — so the worker writes it to its own shard and returns
    only the log record.

    `start_episode` is called *before* `run_demo`, which resets the robot as its
    first act and does not disturb an open recording. That deliberately puts the
    HOME -> approach bridge inside the episode: a policy at inference time starts
    from HOME, so if the dataset began at the approach pose it would have no data
    for how to get there.
    """
    entry_id, xy, yaw, appearance_seed = task
    stop = _W.get("stop")
    if stop is not None and stop.is_set():
        # Cancelled. Return immediately so the pool drains in seconds and every
        # worker reaches its shard flush, instead of being SIGTERMed mid-run.
        return {"id": entry_id, "written": False, "skipped": True, "shard": _W["writer"].name}

    robot, writer = _W["robot"], _W["writer"]

    # Appearance only, so the trial's physics is unaffected. Before the episode
    # opens, so frame 0 already shows this episode's colours.
    appearance = robot.randomize_appearance(np.random.default_rng(appearance_seed))

    robot.start_episode(camera=writer.camera)
    t0 = time.perf_counter()
    try:
        res = run_demo(robot, _demo(entry_id), np.asarray(xy), float(yaw), _W["side"], _W["solver"])
    finally:
        # Always close the recording, or a raising trial leaks its frames into
        # the next one and silently corrupts that episode.
        episode = robot.end_episode()
    elapsed = time.perf_counter() - t0

    rec = {
        "id": entry_id,
        "cube_xy": list(xy),
        "cube_yaw": float(yaw),
        "success": bool(res),
        # The FULL string, not a collapsed key. `_reason_key` in
        # validate_library.py folded "cube at (0.31,-0.22) outside the bin
        # footprint" down to "cube_not_in_bin" before anything was saved, which
        # is why the dominant failure mode was never diagnosable.
        "reason": res.reason,
        "cube_final": None if res.cube_pose is None else res.cube_pose[:3, 3].tolist(),
        "n_frames": len(episode),
        "seconds": round(elapsed, 2),
        "written": False,
        "appearance": appearance,
        # Which shard this episode went to. Reported by the worker rather than
        # discovered by globbing `data/`: a glob for "*_w*" also matched
        # `data/egodex_raw_wipe_screen`, and the merge then tried to resolve
        # that as a HuggingFace repo.
        "shard": writer.name,
    }
    if res:
        rec["written"] = writer.write(episode, robot.index)
        if not rec["written"]:
            rec["reason"] = f"too_short ({len(episode)} frames)"
    return rec


def _show(p: Path) -> str:
    """Repo-relative if it is inside the repo, absolute otherwise (`--root`)."""
    try:
        return str(p.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(p)


def written_shards(rows: list[dict]) -> list[str]:
    """Shards that actually received an episode.

    A worker whose every trial failed never creates its dataset — `ShardWriter`
    builds it lazily — so including it in the merge would point
    `aggregate_datasets` at a directory with no metadata.
    """
    return sorted({r["shard"] for r in rows if r["written"]})


def classify(rec: dict, half: float) -> str:
    """A countable bucket for one trial — derived from the log, never stored in it.

    The distinction the handoff asks for: a cube at *exactly* its spawn pose was
    never gripped, which is a different bug from one that travelled and fell
    short. Collapsing these two into `cube_not_in_bin` is what hid the problem.
    """
    if rec["success"]:
        return "success" if rec["written"] else "success_but_too_short"
    if rec["cube_final"] is None:
        return "no_execution: " + rec["reason"].split(" ")[0].split("(")[0]
    moved = float(np.linalg.norm(np.asarray(rec["cube_final"][:2]) - np.asarray(rec["cube_xy"])))
    lifted = rec["cube_final"][2] > half + 0.01
    if moved < 0.02 and not lifted:
        return "never_gripped (cube did not move)"
    if moved < 0.08:
        return "slipped (<8 cm)"
    return "carried but missed (>8 cm)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="egodex_pick_place", help="dataset name under the output root")
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="output directory (default data/lerobot/). A full run is ~12-16 GB including "
        "the unmerged shards, which does not fit on this box's root filesystem.",
    )
    ap.add_argument("--trials", type=int, default=None, help="scenes per demo")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N demos")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--side", default="right")
    ap.add_argument("--no-merge", action="store_true", help="leave the shards unmerged")
    ap.add_argument(
        "--feasible-only",
        action="store_true",
        help="reuse a previous feasibility screen instead of letting physics decide",
    )
    args = ap.parse_args()

    if not os.environ.get("MUJOCO_GL"):
        print("MUJOCO_GL is unset — rendering needs a GL backend. Try MUJOCO_GL=egl.")
        return 1

    quiet_subprocess_logging()  # the merge runs in THIS process
    dcfg = config.skills()["dataset"]
    n_trials = int(dcfg["trials_per_demo"]) if args.trials is None else args.trials
    seed = int(dcfg["seed"]) if args.seed is None else args.seed
    # Rendering workers hold ~1.9 GB each; `n_processes` (16) is tuned for the
    # headless batch steps and will not fit here.
    n_proc = args.processes if args.processes is not None else 8

    ids = source_demo_ids(args.limit, args.feasible_only)

    # One INDEPENDENT scene per trial — see `sample_scenes`. Drawing `n_trials`
    # and reusing them across demos gave the whole dataset 20 distinct cube
    # poses.
    scenes = iter(sample_scenes(len(ids) * n_trials, seed))
    # The appearance seed travels WITH the task rather than being drawn in the
    # worker, so a trial's colours depend on the trial and not on which process
    # happened to pick it up or in what order. `imap_unordered` makes any
    # worker-side RNG unreproducible.
    tasks = [(i, *next(scenes), seed + k) for k, i in enumerate(i for i in ids for _ in range(n_trials))]
    half = float(config.skills()["scene"]["cube"]["size"]) / 2.0

    print(
        f"generating from {len(ids)} demos x {n_trials} scenes = {len(tasks)} trials "
        f"on {n_proc} processes (seed {seed})"
    )

    rows, t0 = [], time.perf_counter()
    cancelled = False
    manager = Manager()
    stop = manager.Event()
    with Pool(
        processes=n_proc, initializer=_init_worker, initargs=(args.name, args.side, args.root, stop)
    ) as pool:
        bar = tqdm(
            pool.imap_unordered(_trial, tasks, chunksize=1),
            total=len(tasks),
            desc="trials",
            unit="trial",
            smoothing=0.05,
            dynamic_ncols=True,
            # stdout, so the caller can send the encoders' stderr to a file and
            # keep the bar readable.
            file=sys.stdout,
        )
        n_ok = 0
        try:
            for rec in bar:
                rows.append(rec)
                n_ok += bool(rec["written"])
                bar.set_postfix(kept=n_ok, yield_=f"{n_ok / len(rows):.0%}", refresh=False)
        except KeyboardInterrupt:
            # Keep everything generated so far. Setting the flag turns every
            # queued trial into an immediate no-op, so the pool drains in
            # seconds and each worker still reaches its shard flush; killing the
            # workers instead would leave every shard unreadable.
            cancelled = True
            stop.set()
            bar.close()
            print("\ncancelled — draining workers and flushing shards, do not kill", flush=True)
        # Shut the workers down GRACEFULLY, before the `with` block's exit calls
        # `terminate()`. Each worker flushes its shard from an `atexit` hook, and
        # `atexit` does not run on the SIGTERM that `terminate()` sends — so
        # letting the context manager clean up would discard every shard's
        # metadata and leave the merge reading half-written datasets.
        if not cancelled:
            print("  finalising shards", flush=True)
        pool.close()
        pool.join()

    rows = [r for r in rows if not r.get("skipped")]

    elapsed = time.perf_counter() - t0
    kept = [r for r in rows if r["written"]]
    frames = sum(r["n_frames"] for r in kept)

    buckets = Counter(classify(r, half) for r in rows)
    print(f"\n=== {len(kept)}/{len(rows)} trials kept in {elapsed / 60:.1f} min ===")
    if kept:
        print(
            f"  {frames} frames, median episode {int(np.median([r['n_frames'] for r in kept]))} "
            f"frames ({np.median([r['n_frames'] for r in kept]) / 50.0:.1f} s)"
        )
        print(f"  {len(kept) / elapsed * 3600:.0f} episodes/hour at {n_proc} processes")
    print("  outcomes:")
    for name, count in buckets.most_common():
        print(f"    {name:<40s} {count:5d}  ({count / len(rows):.1%})")

    log = dataset_root(f"{args.name}_trials.json", args.root)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "conventions_version": config.CONVENTIONS_VERSION,
                "seed": seed,
                "n_demos": len(ids),
                "n_trials_per_demo": n_trials,
                "n_kept": len(kept),
                "elapsed_s": round(elapsed, 1),
                "trials": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {_show(log)}")

    shards = written_shards(rows)
    if shards and not args.no_merge:
        print(f"merging {len(shards)} shards -> {args.name}")
        out = merge_shards(shards, args.name, base=args.root)
        print(f"wrote {_show(out)}")

    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
