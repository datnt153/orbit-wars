"""Assemble a self-contained main.py for the ensemble agent.

Embeds v4 + marcodg source as base64 (exec'd into fresh namespaces so v4
stays reentrant for rollouts), inlines fast_sim, and the ensemble logic.
No file/path/__file__ dependencies → safe under Kaggle's exec loader.
"""
import base64
import pathlib

# Source-of-truth = repo (no /tmp; embedded sources verified byte-identical
# to these on 2026-05-18). build.py lives in submit/, repo root is parent.
ROOT = pathlib.Path(__file__).resolve().parent.parent
V4 = (ROOT / "agents" / "v4_rudra.py").read_text()
MARCO = (ROOT / "agents" / "marcodg_v33.py").read_text()
FAST_SIM = (ROOT / "src" / "fast_sim.py").read_text()

v4_b64 = base64.b64encode(V4.encode()).decode()
marco_b64 = base64.b64encode(MARCO.encode()).decode()

# Strip the leading module docstring + __future__ from fast_sim, keep code.
fs_lines = FAST_SIM.split("\n")
# find first line after the module docstring (it ends with the closing """)
start = 0
for i, ln in enumerate(fs_lines):
    if ln.strip() == '"""' and i > 0:
        start = i + 1
        break
fast_sim_body = "\n".join(fs_lines[start:])
# drop "from __future__ import annotations" (not needed, must be top-of-file)
fast_sim_body = fast_sim_body.replace(
    "from __future__ import annotations\n", ""
)

MAIN = '''"""Ensemble agent (self-contained). v4 + marcodg picked by deep exact rollout."""
import base64 as _b64
import math
import time

_V4_B64 = "{v4_b64}"
_MARCO_B64 = "{marco_b64}"


def _exec_src(b64):
    ns = {{}}
    src = _b64.b64decode(b64).decode()
    exec(compile(src, "<embedded>", "exec"), ns)
    return ns


# Continuous instances for real-turn move generation (one chain per game).
_V4_REAL = _exec_src(_V4_B64)
_MARCO_REAL = _exec_src(_MARCO_B64)


def _v4_real_agent(obs):
    try:
        r = _V4_REAL["agent"](obs)
        return list(r) if r else []
    except Exception:
        return []


def _marco_real_agent(obs, config=None):
    try:
        r = _MARCO_REAL["agent"](obs, config)
        return list(r) if r else []
    except Exception:
        return []


class _AttrDict(dict):
    """v4_rudra mixes obs.get(...) with obs.angular_velocity. _opp_view
    yields a plain dict → without this the rollout v4 raised AttributeError
    every call and act() returned [] (silent no-op opponent). Expose keys
    as attributes too. [fixed 2026-05-18]"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


# --- Isolated v4 instances for rollouts (reset per rollout; not reentrant) ---
class _V4Inst:
    __slots__ = ("ns",)

    def __init__(self):
        self.ns = _exec_src(_V4_B64)

    def reset(self, obs):
        obs = _AttrDict(obs)
        self.ns["steps"] = 3
        for g in ("fleet_trajectories", "reinforcement_trajectories",
                  "moving_planets", "planets_coords"):
            obj = self.ns.get(g)
            if isinstance(obj, list):
                obj.clear()
            elif isinstance(obj, dict):
                obj.clear()
        try:
            self.ns["fill_moving_planets"](obs)
        except Exception:
            pass

    def act(self, obs):
        obs = _AttrDict(obs)
        try:
            r = self.ns["agent"](obs)
            return list(r) if r else []
        except Exception:
            return []


_ROLLOUT_POOL = [_V4Inst() for _ in range(4)]

# ---------------------- fast_sim (inlined) ----------------------
{fast_sim_body}

# ---------------------- ensemble logic ----------------------
TIME_BUDGET_S = 0.80
ROLLOUT_DEPTH = 200


def _read(obs, k, d=None):
    if isinstance(obs, dict):
        return obs.get(k, d)
    return getattr(obs, k, d)


def _opp_view(state, player):
    return {{
        "player": player,
        "planets": [list(p) for p in state["planets"]],
        "fleets": [list(f) for f in state["fleets"]],
        "initial_planets": [list(p) for p in state["initial_planets"]],
        "comets": state["comets"],
        "comet_planet_ids": list(state["comet_planet_ids"]),
        "angular_velocity": state["angular_velocity"],
        "next_fleet_id": state["next_fleet_id"],
        "step": state["step"],
    }}


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
    s = clone(state)
    for p in range(num_agents):
        _ROLLOUT_POOL[p].reset(_opp_view(s, p))
    acts = []
    for p in range(num_agents):
        acts.append(first_move if p == me
                    else _ROLLOUT_POOL[p].act(_opp_view(s, p)))
    step(s, acts, num_agents)
    for _ in range(depth - 1):
        if s.get("_done") or time.perf_counter() > deadline:
            break
        acts = [_ROLLOUT_POOL[p].act(_opp_view(s, p))
                for p in range(num_agents)]
        step(s, acts, num_agents)
    return _score_for(s, me, num_agents)


def agent(obs, config=None):
    t0 = time.perf_counter()
    deadline = t0 + TIME_BUDGET_S
    me = int(_read(obs, "player", 0))
    planets = _read(obs, "planets", []) or []
    owners = {{int(p[1]) for p in planets if int(p[1]) >= 0}}
    num_agents = 4 if (owners and max(owners) >= 2) else 2

    v4_move = _v4_real_agent(obs)
    marco_move = _marco_real_agent(obs, config)

    if not marco_move or marco_move == v4_move:
        return v4_move

    # Fixed rollout (real v4 opponent) ≈ 37 ms/step → split the REMAINING
    # budget equally between the 2 candidates so both reach the SAME depth
    # (shared deadline starved candidate-2 to ~1 ply). [2026-05-18]
    state = new_state(obs)
    now = time.perf_counter()
    per = max(0.0, (deadline - now) / 2.0)
    s_v4 = _rollout(state, me, num_agents, v4_move, ROLLOUT_DEPTH, now + per)
    now2 = time.perf_counter()
    s_marco = _rollout(state, me, num_agents, marco_move,
                       ROLLOUT_DEPTH, now2 + per)

    margin = max(10.0, 0.05 * abs(s_v4))
    if s_marco > s_v4 + margin:
        return marco_move
    return v4_move


__all__ = ["agent"]
'''.format(v4_b64=v4_b64, marco_b64=marco_b64, fast_sim_body=fast_sim_body)

# Write the FIXED bundle separately — keep submit/main.py (known-good,
# no-op-rollout baseline) intact until the fix is validated (always-safe).
out = ROOT / "submit" / "main_fixed.py"
out.write_text(MAIN)
print(f"Wrote {out} ({len(MAIN)} bytes)")
print("v4/marco/fast_sim read from repo (agents/, src/); _AttrDict fix applied")
