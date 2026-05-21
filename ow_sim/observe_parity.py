"""Parity + speed check: Rust ow_sim.EnvPool.observe_batch vs Python
src.policy_encode.encode_state. Must match within f32 precision.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ow_sim
from policy_encode import encode_state


def gen_actions(state_dict, num_agents, rng):
    pos = {int(p[0]): (p[2], p[3]) for p in state_dict["planets"]}
    acts = []
    for ag in range(num_agents):
        moves = []
        owned = [p for p in state_dict["planets"] if int(p[1]) == ag]
        targets = state_dict["planets"]
        for op in owned:
            if rng.random() > 0.3 or op[5] < 2: continue
            tgt = rng.choice(targets)
            if int(tgt[0]) == int(op[0]): continue
            n = max(1, int(op[5] * rng.uniform(0.3, 1.0)))
            angle = math.atan2(tgt[3] - op[3], tgt[2] - op[2])
            moves.append([int(op[0]), float(angle), int(n)])
        acts.append(moves)
    return acts


def main():
    eps = sorted((ROOT/"data/bovard/2026-05-04/episodes/episodes").glob("*.json"))
    rep = json.loads(eps[0].read_text())
    obs0 = rep["steps"][0][0]["observation"]
    num_agents = len(rep["steps"][0])
    template = ow_sim.State(obs0, 6.0, 500)

    # ---- PARITY: step through, compare each turn ----
    pool = ow_sim.EnvPool(template, 1)
    rng = random.Random(0)
    py_state_d = template.to_dict()
    max_pf = max_pm = max_om = max_gf = 0.0
    pid_match = True
    n_steps = 60
    for t in range(n_steps):
        for p in range(num_agents):
            pf_py, pm_py, om_py, gf_py, pids_py = encode_state(py_state_d, p)
            pf_rs, pm_rs, om_rs, gf_rs, pids_rs = pool.observe_batch(num_agents)
            max_pf = max(max_pf, float(np.abs(pf_py - pf_rs[0, p]).max()))
            max_pm = max(max_pm, float(np.abs(pm_py - pm_rs[0, p]).max()))
            max_om = max(max_om, float(np.abs(om_py - om_rs[0, p]).max()))
            max_gf = max(max_gf, float(np.abs(gf_py - gf_rs[0, p]).max()))
            if list(pids_py) != list(pids_rs[0]):
                pid_match = False
        # advance both
        acts = gen_actions(py_state_d, num_agents, rng)
        # Python ref encoder doesn't have a step; we use the Rust state for
        # stepping (it's already parity-tested at the step level). encode
        # parity only needs the state observed to be the same.
        pool.step_batch([acts], num_agents)
        py_state_d = pool.get_state(0).to_dict()

    print(f"OBSERVE PARITY over {n_steps} steps × {num_agents} players "
          f"({n_steps*num_agents} samples):")
    print(f"  max|Δ| pf={max_pf:.2e}  pm={max_pm:.2e}  om={max_om:.2e}  gf={max_gf:.2e}")
    print(f"  planet_ids match: {pid_match}")
    ok = (max(max_pf, max_pm, max_om, max_gf) < 1e-5) and pid_match
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        return

    # ---- SPEED: encode_batch Python vs Rust at N envs ----
    print("\nSPEED encode N envs × 2 players:")
    print(f"  {'envs':>4}  {'Python(ms)':>11}  {'Rust(ms)':>10}  {'speedup':>8}")
    for n_envs in (16, 64, 256):
        pool = ow_sim.EnvPool(template, n_envs)
        states = [pool.get_state(e).to_dict() for e in range(n_envs)]
        # Python encode_batch equivalent
        from policy_encode import MAXP, F, G
        def py_encode_batch():
            pf = np.zeros((n_envs, num_agents, MAXP, F), np.float32)
            pm = np.zeros((n_envs, num_agents, MAXP), np.float32)
            om = np.zeros((n_envs, num_agents, MAXP), np.float32)
            gf = np.zeros((n_envs, num_agents, G), np.float32)
            for e, sd in enumerate(states):
                for p in range(num_agents):
                    obs = {"planets": sd["planets"], "fleets": sd["fleets"],
                           "comet_planet_ids": [], "step": int(sd.get("step", 0))}
                    f_, m, o, g, _ = encode_state(obs, p)
                    pf[e, p] = f_; pm[e, p] = m; om[e, p] = o; gf[e, p] = g
            return pf, pm, om, gf
        # warm
        py_encode_batch(); pool.observe_batch(num_agents)
        K = 20
        t = time.perf_counter()
        for _ in range(K): py_encode_batch()
        py_ms = (time.perf_counter() - t) / K * 1000
        t = time.perf_counter()
        for _ in range(K): pool.observe_batch(num_agents)
        rs_ms = (time.perf_counter() - t) / K * 1000
        print(f"  {n_envs:>4}  {py_ms:>11.2f}  {rs_ms:>10.2f}  {py_ms/max(rs_ms,1e-6):>7.1f}×")


if __name__ == "__main__":
    main()
