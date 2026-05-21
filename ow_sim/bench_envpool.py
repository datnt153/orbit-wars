"""R2 — vectorized batched-step throughput benchmark.

Compares single-env stepping vs EnvPool (rayon-parallel) at various N.
Gate: ≥80k SPS aggregate on 24 threads (i7-13700K).
"""
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ow_sim


def make_template():
    eps_dir = ROOT / "data" / "bovard" / "2026-05-04" / "episodes" / "episodes"
    rep = json.loads(sorted(eps_dir.glob("*.json"))[0].read_text())
    obs = rep["steps"][0][0]["observation"]
    num_agents = len(rep["steps"][0])
    return ow_sim.State(obs, 6.0, 500), num_agents


def gen_actions_from_state(state_dict, num_agents, rng):
    acts = []
    for ag in range(num_agents):
        moves = []
        owned = [p for p in state_dict["planets"] if int(p[1]) == ag]
        targets = [p for p in state_dict["planets"]]
        for op in owned:
            if rng.random() > 0.3 or op[5] < 2: continue
            tgt = rng.choice(targets)
            if int(tgt[0]) == int(op[0]): continue
            n = max(1, int(op[5] * rng.uniform(0.3, 1.0)))
            angle = math.atan2(tgt[3] - op[3], tgt[2] - op[2])
            moves.append([int(op[0]), float(angle), int(n)])
        acts.append(moves)
    return acts


def bench(n_envs, n_steps=200):
    template, num_agents = make_template()
    pool = ow_sim.EnvPool(template, n_envs)
    rng = random.Random(0)
    # Pre-generate actions on a single template state (cheap, realistic enough).
    # Better: per-env per-step, but it requires get_state — expensive in batch.
    # For throughput test the action shape matters more than content; use a
    # representative batch every step.
    template_dict = template.to_dict()

    t = time.perf_counter()
    for _ in range(n_steps):
        batch = [gen_actions_from_state(template_dict, num_agents, rng)
                 for _ in range(n_envs)]
        pool.step_batch(batch, num_agents)
    dt = time.perf_counter() - t
    total_steps = n_envs * n_steps
    sps = total_steps / dt
    return dt, sps


def main():
    print("EnvPool throughput  (envs × 200 steps each)")
    print(f"  {'envs':>4}  {'wall':>7}  {'aggregate SPS':>14}  {'per-env SPS':>12}")
    for n in (1, 4, 8, 16, 24, 48, 96, 256):
        dt, sps = bench(n, 200)
        per = sps / n
        print(f"  {n:>4}  {dt:>6.2f}s  {sps:>14,.0f}  {per:>12,.0f}")


if __name__ == "__main__":
    main()
