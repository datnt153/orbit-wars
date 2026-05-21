"""Ensemble: pick between v4's and marcodg's joint move via DEEP exact rollout.

v4 (LB ~982) and marcodg (~46% vs v4 — wins different board states) each
propose a full joint move. For each, run a DEEP fast_sim rollout where the
opponent plays v4 (the dominant ladder archetype) and the controlled side
continues with that same agent. Score by real outcome / long-horizon
ship+production differential.

Only 2 candidates → afford a deep rollout (low eval noise, unlike the
24-step / 25-candidate search that failed). v4's move is the default;
marcodg only overrides it when the rollout shows a clear margin → floor ≡ v4.
"""
from __future__ import annotations

import importlib.util
import sys
import time

sys.path.insert(0, "/tmp/ow_sim")
import fast_sim
import v4_instance

_V4_PATH = "/tmp/ow_train/rule-based submission.py"
_MARCO_PATH = "/tmp/hunt/marcodg_marco-dg-v3-3-top-score-1060-5/main.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # required so @dataclass can resolve its module
    spec.loader.exec_module(m)
    return m

# Continuous instances for real-turn move generation (one chain per game).
_V4_REAL = _load(_V4_PATH, "_v4_real")
_MARCO_REAL = _load(_MARCO_PATH, "_marco_real")

# Isolated v4 instances for rollouts (reset per rollout; not reentrant).
_ROLLOUT_POOL = [v4_instance.make_v4() for _ in range(4)]

TIME_BUDGET_S = 0.85
ROLLOUT_DEPTH = 140


def _read(obs, k, d=None):
    if isinstance(obs, dict):
        return obs.get(k, d)
    return getattr(obs, k, d)


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


def _rollout(state, me, num_agents, first_move, depth, deadline):
    s = fast_sim.clone(state)
    for p in range(num_agents):
        v4_instance.reset(_ROLLOUT_POOL[p], _opp_view(s, p))
    acts = []
    for p in range(num_agents):
        acts.append(
            first_move if p == me
            else v4_instance.act(_ROLLOUT_POOL[p], _opp_view(s, p))
        )
    fast_sim.step(s, acts, num_agents)
    for _ in range(depth - 1):
        if s.get("_done") or time.perf_counter() > deadline:
            break
        acts = [
            v4_instance.act(_ROLLOUT_POOL[p], _opp_view(s, p))
            for p in range(num_agents)
        ]
        fast_sim.step(s, acts, num_agents)
    return _score_for(s, me, num_agents)


def agent(obs, config=None):
    t0 = time.perf_counter()
    deadline = t0 + TIME_BUDGET_S
    me = int(_read(obs, "player", 0))
    planets = _read(obs, "planets", []) or []
    owners = {int(p[1]) for p in planets if int(p[1]) >= 0}
    num_agents = 4 if (owners and max(owners) >= 2) else 2

    try:
        v4_move = _V4_REAL.agent(obs) or []
    except Exception:
        v4_move = []
    try:
        marco_move = _MARCO_REAL.agent(obs, config) or []
    except Exception:
        marco_move = []

    if not marco_move or marco_move == v4_move:
        return v4_move
    if not v4_move:
        # v4 idle — trust marco only if rollout says it's clearly positive
        pass

    state = fast_sim.new_state(obs)
    s_v4 = _rollout(state, me, num_agents, v4_move, ROLLOUT_DEPTH, deadline)
    if time.perf_counter() > deadline:
        return v4_move
    s_marco = _rollout(state, me, num_agents, marco_move, ROLLOUT_DEPTH, deadline)

    margin = max(10.0, 0.05 * abs(s_v4))
    if s_marco > s_v4 + margin:
        return marco_move
    return v4_move
