"""Does replaying the human's recorded orientation beat flattening it to top-down?

Measures the IK SCREEN only (`fit_yaw_offsets`), which is what produces
`no_feasible_yaw` -- no physics, so it is ~10x cheaper than a full trial.

Demos are taken straight from data/retargeted/, NOT from the feasibility file:
that file lists demos feasible UNDER top_down, so using it would rig the
comparison in favour of the thing being tested.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/home/uko/ERH_hackathon")

N_DEMOS = 6
N_SCENES = 1
# (orientation, max tilt degrees). The demos sit at a ~52 deg median approach,
# so a 45 deg cap cannot reach where they actually are.
CONFIGS = [("top_down", 45), ("replay", 45), ("top_down", 60), ("replay", 60)]

W = {}


def tilt_list(max_deg):
    out = [0.0]
    for d in (15, 30, 45, 60, 75):
        if d <= max_deg:
            out += [float(d), -float(d)]
    return out


def init():
    from common.ik import IKSolver
    from sim.sim_robot import SimRobot

    W["robot"] = SimRobot(render=False)
    W["solver"] = IKSolver()


def screen(task):
    from common import config, demos
    from sim import execute as ex

    orient, max_tilt, did, xy, yaw = task
    config.skills()["exec"]["demo_tilt_deg"] = tilt_list(max_tilt)

    robot, solver = W["robot"], W["solver"]
    half = float(config.skills()["scene"]["cube"]["size"]) / 2.0
    robot.reset()
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    robot.set_free_body_pose("cube_free", [xy[0], xy[1], half], q)

    try:
        demo = demos.load_source_demo(
            config.path("data", "retargeted", did + ".npz"), orientation=orient
        )
        plan = ex.plan_from_demo(demo, robot.body_pose("cube"), "right", solver)
        path, why = ex.fit_yaw_offsets(
            solver, plan, "right", float(config.skills()["exec"]["rate_hz"])
        )
        ok = path is not None
    except Exception as e:
        ok, why = False, f"error: {type(e).__name__}"
    return {"orient": orient, "tilt": max_tilt, "id": did, "ok": bool(ok), "why": "" if ok else why}


def main():
    from common import config

    ids = [p.stem for p in sorted(config.path("data", "retargeted").glob("*.npz"))][:N_DEMOS]
    sc = config.skills()["scene"]
    c = np.asarray(sc["pick_zone"]["center"])
    s = np.asarray(sc["pick_zone"]["size"])
    rng = np.random.default_rng(20261017)
    scenes = [
        ((c + rng.uniform(-0.5, 0.5, 2) * s).tolist(), float(rng.uniform(-np.pi / 4, np.pi / 4)))
        for _ in range(N_SCENES)
    ]
    tasks = [(o, t, d, xy, y) for (o, t) in CONFIGS for d in ids for xy, y in scenes]
    print(f"{len(tasks)} screens = {len(CONFIGS)} configs x {len(ids)} demos x {len(scenes)} scenes",
          flush=True)

    t0 = time.perf_counter()
    rows = []
    with Pool(processes=12, initializer=init) as pool:
        for r in pool.imap_unordered(screen, tasks, chunksize=1):
            rows.append(r)
            print(f"    {len(rows):3d}/{len(tasks)} {r['orient']:<9s}+/-{r['tilt']:<3d} "
                  f"{'OK  ' if r['ok'] else 'FAIL'} {r['id']}", flush=True)
    el = time.perf_counter() - t0

    n = len(ids) * len(scenes)
    print(f"\ndone in {el:.0f} s\n")
    print("  orientation  tilt_cap   paths solvable")
    for o, t in CONFIGS:
        rs = [r for r in rows if r["orient"] == o and r["tilt"] == t]
        ok = sum(r["ok"] for r in rs)
        print(f"  {o:<11s}  +/-{t:3d}      {ok:3d}/{len(rs)}  ({ok / max(len(rs), 1):5.1%})")

    print("\n  top failure reasons:")
    from collections import Counter

    for o, t in CONFIGS:
        rs = [r for r in rows if r["orient"] == o and r["tilt"] == t and not r["ok"]]
        c2 = Counter(r["why"].split("(")[0].strip()[:34] for r in rs)
        print(f"   {o} +/-{t}: " + ", ".join(f"{k}={v}" for k, v in c2.most_common(3)))


if __name__ == "__main__":
    main()
