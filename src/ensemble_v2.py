"""Ensemble v2: N-candidate pool, multi-opponent exact rollout.

Each turn, several public agents (v4, marcodg, romantamrazov, ykhnkf)
each propose a joint move. Every candidate is scored by deep fast_sim
rollouts against one or more opponent models (v4, marco). v4's move is
the default; another candidate overrides it only with a clear margin
→ floor ≡ v4.

Tunable via env vars for arena sweeps:
  OWE_MARGIN, OWE_DEPTH, OWE_OPPS (csv of v4,marco), OWE_AGENTS (csv)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

sys.path.insert(0, "/tmp/ow_sim")
import fast_sim
import v4_instance

_PATHS = {
    "v4": "/tmp/ow_train/rule-based submission.py",
    "marco": "/tmp/hunt/marcodg_marco-dg-v3-3-top-score-1060-5/main.py",
    "roman": "/tmp/hunt/romantamrazov_orbit-star-wars-lb-max-1224/main.py",
    "ykhnkf": "/tmp/hunt/ykhnkf_distance-prioritized-agent-lb-max-score-1100/main.py",
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# Continuous instances for real-turn candidate generation (1 call/turn).
_REAL = {}
for _k, _p in _PATHS.items():
    try:
        _REAL[_k] = _load(_p, f"_real_{_k}")
    except Exception as _e:
        print(f"[ensemble_v2] failed to load {_k}: {_e}")

# Isolated v4 pool for rollout opponent model (reset per rollout).
_V4_OPP_POOL = [v4_instance.make_v4() for _ in range(4)]
# marco is stateless across turns → one instance reused as opp model.
_MARCO_OPP = _REAL.get("marco")

TIME_BUDGET_S = float(os.environ.get("OWE_BUDGET", "0.80"))
ROLLOUT_DEPTH = int(os.environ.get("OWE_DEPTH", "140"))
MARGIN_ABS = float(os.environ.get("OWE_MARGIN", "10"))
OPPS = os.environ.get("OWE_OPPS", "v4").split(",")
AGENTS = os.environ.get("OWE_AGENTS", "v4,marco").split(",")


def _read(obs, k, d=None):
    if isinstance(obs, dict):
        return obs.get(k, d)
    return getattr(obs, k, d)


def _cand_move(key, obs, config):
    m = _REAL.get(key)
    if m is None:
        return []
    try:
        if key == "marco":
            r = m.agent(obs, config)
        else:
            r = m.agent(obs)
        return list(r) if r else []
    except Exception:
        return []


def _opp_view(state, player):
    return {
        "player": player,
        "planets": [list(p) for p in state["planets"]],
        "fleets": [list(f) for f in state["fleets"]],
        "initial_planets": [list(p) for p in state["initial_planets"]],
        "comets": state["comets"],
        "comet_planet_ids": list(state["comet_planet_ids"]),
        "angular_velocity": state["angular_velocity"],
        "next_fleet_id": state["next_fleet_id"],
        "step": state["step"],
    }


def _score_for(state, me, num_agents):
    if state.get("_done"):
        rew = state.get("_rewards")
        if rew and me < len(rew):
            return 1e6 if rew[me] == 1 else -1e6
    my = opp = 0.0
    for p in state["planets"]:
        v = p[5] + p[6] * 8.0
        if p[1] == me:
            my += v
        elif p[1] >= 0:
            opp += v
    for f in state["fleets"]:
        if f[1] == me:
            my += f[6]
        elif f[1] >= 0:
            opp += f[6]
    return my - opp


def _opp_act(opp_kind, slot, obs):
    if opp_kind == "v4":
        return v4_instance.act(_V4_OPP_POOL[slot], obs)
    if opp_kind == "marco" and _MARCO_OPP is not None:
        try:
            r = _MARCO_OPP.agent(obs, None)
            return list(r) if r else []
        except Exception:
            return []
    return []


def _rollout(state, me, num_agents, first_move, opp_kind, depth, deadline):
    s = fast_sim.clone(state)
    if opp_kind == "v4":
        for p in range(num_agents):
            v4_instance.reset(_V4_OPP_POOL[p], _opp_view(s, p))
    acts = []
    for p in range(num_agents):
        acts.append(first_move if p == me
                    else _opp_act(opp_kind, p, _opp_view(s, p)))
    fast_sim.step(s, acts, num_agents)
    for _ in range(depth - 1):
        if s.get("_done") or time.perf_counter() > deadline:
            break
        acts = [_opp_act(opp_kind, p, _opp_view(s, p))
                for p in range(num_agents)]
        fast_sim.step(s, acts, num_agents)
    return _score_for(s, me, num_agents)


def agent(obs, config=None):
    t0 = time.perf_counter()
    deadline = t0 + TIME_BUDGET_S
    me = int(_read(obs, "player", 0))
    planets = _read(obs, "planets", []) or []
    owners = {int(p[1]) for p in planets if int(p[1]) >= 0}
    num_agents = 4 if (owners and max(owners) >= 2) else 2

    v4_move = _cand_move("v4", obs, config)
    # Build candidate list: v4 first (default), then the rest, dedup.
    moves = {"v4": v4_move}
    for k in AGENTS:
        if k == "v4":
            continue
        moves[k] = _cand_move(k, obs, config)

    # Score v4 first
    def multi_rollout(mv):
        scores = []
        for opp in OPPS:
            if time.perf_counter() > deadline:
                break
            scores.append(
                _rollout(fast_sim.new_state(obs), me, num_agents,
                         mv, opp, ROLLOUT_DEPTH, deadline)
            )
        return min(scores) if scores else -1e18  # robust: worst-case opp

    s_v4 = multi_rollout(v4_move)
    best_key, best_move, best_score = "v4", v4_move, s_v4
    margin = max(MARGIN_ABS, 0.05 * abs(s_v4))
    for k in AGENTS:
        if k == "v4":
            continue
        mv = moves.get(k)
        if not mv or mv == v4_move:
            continue
        if time.perf_counter() > deadline:
            break
        sc = multi_rollout(mv)
        if sc > s_v4 + margin and sc > best_score:
            best_key, best_move, best_score = k, mv, sc
    return best_move


__all__ = ["agent"]
