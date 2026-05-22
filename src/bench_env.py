"""Throughput profiler for the ow_sim env loop — finds the CPU/rayon ceiling.

Isolates the Rust env step (observe_batch + step_from_samples + reset) with
random actions, NO torch/NN. Sweeps N_ENVS so we can see how SPS scales with
parallel work per step and where the Python wrapper (per-env loops) caps it.

Usage: RAYON_NUM_THREADS=80 bench_env.py [n_steps]
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import ow_sim

ROOT = Path(__file__).resolve().parent.parent
MAXP = 48


def load_template():
    p = sorted(glob.glob(str(ROOT / "data" / "bovard" / "**" / "*.json"),
                         recursive=True))[0]
    o = json.loads(open(p).read())["steps"][0][0]["observation"]
    return ow_sim.State(o, 6.0, 500)


def bench(tmpl, n_envs, n_steps, num_agents=2):
    pool = ow_sim.EnvPool(tmpl, n_envs)
    rng = np.random.default_rng(0)
    # warmup (build caches / first alloc)
    pool.observe_batch(num_agents)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        pf, pm, om, gf, _ = pool.observe_batch(num_agents)
        om = om.reshape(n_envs, num_agents, MAXP)
        launch = (rng.random((n_envs, num_agents, MAXP)) < 0.1).astype(np.float32) * om
        tgt = rng.integers(0, MAXP, size=(n_envs, num_agents, MAXP)).astype(np.int64)
        sf = np.ones((n_envs, num_agents, MAXP), np.float32)
        pool.step_from_samples(np.ascontiguousarray(launch),
                               np.ascontiguousarray(tgt), sf, num_agents)
        dm = pool.done_mask()
        for e in range(n_envs):
            if dm[e]:
                pool.reset_one(e, tmpl)
    dt = time.perf_counter() - t0
    sps = n_envs * n_steps / dt
    print(f"N_ENVS={n_envs:>5}  {sps:>9.0f} env-steps/s  "
          f"({dt:.2f}s / {n_steps} steps, {1000*dt/n_steps:.1f} ms/step)",
          flush=True)
    return sps


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    tmpl = load_template()
    import os
    print(f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS','(default)')}",
          flush=True)
    for ne in [128, 256, 512, 1024, 2048]:
        bench(tmpl, ne, n_steps)
