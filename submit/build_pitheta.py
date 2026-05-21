"""Assemble the π_θ-opponent ensemble bundle (P1).

Same self-contained pattern as build.py, but the rollout opponent (and
the controlled side's continuation) is the fast numpy π_θ distilled from
bovard rated≥1500 — replacing the slow v4 (~9ms → ~1ms/call) so the
rollout reaches deep AND models the real ~1550 field instead of v4 982.

Candidates are still real v4 + marco moves (precise proposers). ONE
delta vs ensemble_fixed: the rollout actor model. encode_state /
numpy-forward are sliced verbatim from the parity-tested repo files
(single source → no train/inference skew). No torch, no /tmp, no
__file__. Writes submit/main_pitheta.py; leaves submit/main*.py intact.
"""
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
V4 = (ROOT / "agents" / "v4_rudra.py").read_text()
MARCO = (ROOT / "agents" / "marcodg_v33.py").read_text()
FAST_SIM = (ROOT / "src" / "fast_sim.py").read_text()
ENC = (ROOT / "src" / "policy_encode.py").read_text()
INF = (ROOT / "src" / "pi_theta_infer.py").read_text()
W_B64 = base64.b64encode((ROOT / "data" / "pi_theta_w.npz").read_bytes()).decode()
v4_b64 = base64.b64encode(V4.encode()).decode()
marco_b64 = base64.b64encode(MARCO.encode()).decode()


def _strip_future(s):
    return s.replace("from __future__ import annotations\n", "")


# fast_sim: drop module docstring + __future__ (must be top-of-file)
_fl = FAST_SIM.split("\n")
_st = next(i + 1 for i, l in enumerate(_fl) if l.strip() == '"""' and i > 0)
FAST_SIM_BODY = _strip_future("\n".join(_fl[_st:]))

# policy_encode: keep constants..encode_state (drop docstring/build/main)
ENC_BLOCK = _strip_future(ENC[ENC.index("MAXP = 48"):ENC.index("\ndef build_dataset")])

# pi_theta_infer: keep _ln.._mha,forward,decode_moves (drop torch/main/load)
INF_BLOCK = INF[INF.index("def _ln("):INF.index('if __name__')]

HEAD = '''"""π_θ-opponent ensemble (self-contained). Rollout opponent = fast
numpy policy distilled from bovard 1500+; candidates = real v4 + marco."""
import base64 as _b64
import io as _io
import math
import time

import numpy as np

_V4_B64 = "@@V4@@"
_MARCO_B64 = "@@MARCO@@"
_PI_W_B64 = "@@PIW@@"


def _exec_src(b64):
    ns = {}
    exec(compile(_b64.b64decode(b64).decode(), "<embedded>", "exec"), ns)
    return ns


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
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


# ---------------------- fast_sim (inlined) ----------------------
@@FASTSIM@@

# ---------------------- encode_state (inlined, parity-tested) ----------------------
@@ENC@@

# ---------------------- pi_theta numpy weights ----------------------
_z = np.load(_io.BytesIO(_b64.b64decode(_PI_W_B64)))
_W = {k: _z[k].astype(np.float32) for k in _z.files if not k.startswith("_")}
_META = (int(_z["_F"]), int(_z["_G"]), int(_z["_D"]))


def load(*a, **k):  # decode_moves guard no-op (weights already loaded)
    return _W, _META


# ---------------------- pi_theta numpy forward (inlined, parity-tested) ----------
@@INF@@

# ---------------------- ensemble logic ----------------------
TIME_BUDGET_S = 0.80
ROLLOUT_DEPTH = 400  # π_θ ~1ms/step → deadline (not depth) is the cap


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


def _pi_move(view, p):
    try:
        return decode_moves(view, p)
    except Exception:
        return []


def _rollout(state, me, num_agents, first_move, depth, deadline):
    s = clone(state)
    acts = [first_move if p == me else _pi_move(_opp_view(s, p), p)
            for p in range(num_agents)]
    step(s, acts, num_agents)
    for _ in range(depth - 1):
        if s.get("_done") or time.perf_counter() > deadline:
            break
        acts = [_pi_move(_opp_view(s, p), p) for p in range(num_agents)]
        step(s, acts, num_agents)
    return _score_for(s, me, num_agents)


def agent(obs, config=None):
    t0 = time.perf_counter()
    deadline = t0 + TIME_BUDGET_S
    me = int(_read(obs, "player", 0))
    planets = _read(obs, "planets", []) or []
    owners = {int(p[1]) for p in planets if int(p[1]) >= 0}
    num_agents = 4 if (owners and max(owners) >= 2) else 2

    v4_move = _v4_real_agent(obs)
    marco_move = _marco_real_agent(obs, config)
    if not marco_move or marco_move == v4_move:
        return v4_move

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
'''

MAIN = (HEAD.replace("@@V4@@", v4_b64).replace("@@MARCO@@", marco_b64)
        .replace("@@PIW@@", W_B64).replace("@@FASTSIM@@", FAST_SIM_BODY)
        .replace("@@ENC@@", ENC_BLOCK).replace("@@INF@@", INF_BLOCK))

out = ROOT / "submit" / "main_pitheta.py"
out.write_text(MAIN)
print(f"Wrote {out} ({len(MAIN)} bytes)")
print("rollout actor = numpy π_θ (bovard≥1500); candidates = real v4+marco")
