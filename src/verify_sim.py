"""Verify fast_sim matches the real engine step-for-step.

Manual-step a real kaggle_environments game, and at every turn feed the
pre-step observation + joint actions into fast_sim.step(), then compare the
resulting planets/fleets/step against the engine's actual next state.
"""
import math
import os
import sys

os.environ.setdefault("KAGGLE_LOG_LEVEL", "ERROR")
sys.path.insert(0, "/tmp/ow_sim")

import logging
logging.disable(logging.CRITICAL)

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
import fast_sim

EPS = 1e-6


def obs_to_dict(o):
    return {
        "planets": [list(p) for p in fast_sim_get(o, "planets", [])],
        "fleets": [list(f) for f in fast_sim_get(o, "fleets", [])],
        "initial_planets": [list(p) for p in fast_sim_get(o, "initial_planets", [])],
        "comets": fast_sim_get(o, "comets", []),
        "comet_planet_ids": list(fast_sim_get(o, "comet_planet_ids", [])),
        "next_fleet_id": fast_sim_get(o, "next_fleet_id", 0),
        "angular_velocity": fast_sim_get(o, "angular_velocity", 0.0),
        "step": fast_sim_get(o, "step", 0),
    }


def fast_sim_get(o, k, d):
    if isinstance(o, dict):
        return o.get(k, d)
    return getattr(o, k, d)


def cmp_planets(sim, real, step):
    sim_by = {p[0]: p for p in sim}
    real_by = {p[0]: p for p in real}
    if set(sim_by) != set(real_by):
        return f"step{step}: planet ids differ sim={sorted(sim_by)} real={sorted(real_by)}"
    for pid, rp in real_by.items():
        spl = sim_by[pid]
        if int(spl[1]) != int(rp[1]):
            return f"step{step} p{pid}: owner sim={spl[1]} real={rp[1]}"
        if int(spl[5]) != int(rp[5]):
            return f"step{step} p{pid}: ships sim={spl[5]} real={rp[5]}"
        if abs(spl[2] - rp[2]) > EPS or abs(spl[3] - rp[3]) > EPS:
            return f"step{step} p{pid}: pos sim=({spl[2]:.6f},{spl[3]:.6f}) real=({rp[2]:.6f},{rp[3]:.6f})"
    return None


def cmp_fleets(sim, real, step):
    if len(sim) != len(real):
        return f"step{step}: fleet count sim={len(sim)} real={len(real)}"
    sim_by = {f[0]: f for f in sim}
    real_by = {f[0]: f for f in real}
    if set(sim_by) != set(real_by):
        return f"step{step}: fleet ids differ"
    for fid, rf in real_by.items():
        sf = sim_by[fid]
        if int(sf[1]) != int(rf[1]) or int(sf[6]) != int(rf[6]):
            return f"step{step} f{fid}: owner/ships sim=({sf[1]},{sf[6]}) real=({rf[1]},{rf[6]})"
        if abs(sf[2] - rf[2]) > EPS or abs(sf[3] - rf[3]) > EPS:
            return f"step{step} f{fid}: pos mismatch"
    return None


def run_one(seed, num_agents):
    env = make("orbit_wars", configuration={"agents": num_agents, "seed": seed}, debug=False)
    env.reset(num_agents=num_agents)
    state = env.state
    mismatches = 0
    steps_checked = 0
    crossed_comet_spawn = False
    while True:
        # Pre-step observation (player 0 sees full state)
        pre = obs_to_dict(state[0]["observation"])
        cur_step = pre["step"]
        if any(cur_step + 1 == s for s in [50, 150, 250, 350, 450]):
            crossed_comet_spawn = True
        actions = []
        for pidx in range(num_agents):
            if state[pidx]["status"] != "ACTIVE":
                actions.append([])
                continue
            o = state[pidx]["observation"]
            actions.append(random_agent(o if isinstance(o, dict) else {
                "player": fast_sim_get(o, "player", pidx),
                "planets": list(fast_sim_get(o, "planets", [])),
            }) or [])

        # fast_sim prediction
        ss = fast_sim.new_state(pre)
        fast_sim.step(ss, actions, num_agents)

        # real engine step
        state = env.step(actions)
        post = obs_to_dict(state[0]["observation"])

        # step 0 = engine map init (obs has no planets yet), not a game turn
        # Comet spawn steps: fast_sim intentionally skips spawning -> skip cmp
        if cur_step >= 1 and not crossed_comet_spawn:
            e1 = cmp_planets(ss["planets"], post["planets"], cur_step)
            e2 = cmp_fleets(ss["fleets"], post["fleets"], cur_step)
            if e1 or e2:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  MISMATCH seed{seed}: {e1 or e2}")
            steps_checked += 1

        if all(s["status"] != "ACTIVE" for s in state):
            break
        if post["step"] > 480:
            break
    return steps_checked, mismatches, crossed_comet_spawn


if __name__ == "__main__":
    total_checked = 0
    total_mm = 0
    for seed in range(8):
        for na in (2, 4):
            chk, mm, comet = run_one(seed, na)
            total_checked += chk
            total_mm += mm
            tag = " (has comet-spawn, post-spawn steps skipped)" if comet else ""
            print(f"seed={seed} {na}P: checked={chk} mismatches={mm}{tag}")
    print(f"\n=== TOTAL: {total_checked} steps checked, {total_mm} mismatches ===")
    print("PERFECT MATCH" if total_mm == 0 else f"DIVERGENCE: {total_mm}")
