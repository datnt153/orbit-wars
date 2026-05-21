"""Fork A — π_θ as a 3rd CANDIDATE on the WORKING v4-opponent rollout.

Built on build.py (ensemble_fixed, LB 1002.3): keeps its v4-opponent
deep rollout and _AttrDict fix UNCHANGED. Adds π_θ (numpy, parity-tested)
as a candidate proposer. A cheap 1-ply prefilter picks the single
challenger = best of {marco_move, π_θ_move}; then EXACTLY two deep
rollouts (v4_move vs challenger) at budget/2 each — same depth as
ensemble_fixed, NO dilution. floor = v4.

⇒ strictly generalizes ensemble_fixed: if π_θ never wins the prefilter,
behaviour == ensemble_fixed; worst case = v4 floor (cannot regress below
the 1002.3 mechanism). π_θ only helps when it proposes a move that both
beats marco 1-ply AND beats v4 in the deep v4-opponent rollout.

Weights file is a build arg (default data/pi_theta_w_aug.npz). No torch,
no /tmp, no __file__. Writes submit/main_pitheta_cand.py.
"""
import base64
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
W_PATH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "data" / "pi_theta_w_aug.npz")
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "main_pitheta_cand.py"

V4 = (ROOT / "agents" / "v4_rudra.py").read_text()
MARCO = (ROOT / "agents" / "marcodg_v33.py").read_text()
FAST_SIM = (ROOT / "src" / "fast_sim.py").read_text()
ENC = (ROOT / "src" / "policy_encode.py").read_text()
INF = (ROOT / "src" / "pi_theta_infer.py").read_text()
v4_b64 = base64.b64encode(V4.encode()).decode()
marco_b64 = base64.b64encode(MARCO.encode()).decode()
W_B64 = base64.b64encode(W_PATH.read_bytes()).decode()


def _nf(s):
    return s.replace("from __future__ import annotations\n", "")


_fl = FAST_SIM.split("\n")
_st = next(i + 1 for i, l in enumerate(_fl) if l.strip() == '"""' and i > 0)
FAST_SIM_BODY = _nf("\n".join(_fl[_st:]))
ENC_BLOCK = _nf(ENC[ENC.index("MAXP = 48"):ENC.index("\ndef build_dataset")])
INF_BLOCK = INF[INF.index("def _ln("):INF.index('if __name__')]

HEAD = '''"""Fork A: π_θ candidate + v4-opponent deep rollout (ensemble_fixed core)."""
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


# --- Isolated v4 instances for the (WORKING) rollout opponent ---
class _V4Inst:
    __slots__ = ("ns",)

    def __init__(self):
        self.ns = _exec_src(_V4_B64)

    def reset(self, obs):
        obs = _AttrDict(obs)
        self.ns["steps"] = 3
        for g in ("fleet_trajectories", "reinforcement_trajectories",
                  "moving_planets", "planets_coords"):
            o = self.ns.get(g)
            if isinstance(o, (list, dict)):
                o.clear()
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
@@FASTSIM@@

# ---------------------- encode_state (inlined, parity-tested) ----------------------
@@ENC@@

# ---------------------- pi_theta numpy weights ----------------------
_z = np.load(_io.BytesIO(_b64.b64decode(_PI_W_B64)))
_W = {k: _z[k].astype(np.float32) for k in _z.files if not k.startswith("_")}
_META = (int(_z["_F"]), int(_z["_G"]), int(_z["_D"]))


def load(*a, **k):
    return _W, _META


# ---------------------- pi_theta numpy forward (inlined, parity-tested) ----------
@@INF@@

# ---------------------- ensemble logic ----------------------
TIME_BUDGET_S = 0.80
ROLLOUT_DEPTH = 200


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


def _pi_move(obs, me):
    try:
        return decode_moves(obs, me)
    except Exception:
        return []


def _prefilter(state, me, num_agents, mv):
    """Cheap 1-ply: my move only, others idle, fast_sim 1 step, leaf eval.
    Crude tiebreak to pick ONE challenger to spend a deep rollout on."""
    s = clone(state)
    acts = [mv if p == me else [] for p in range(num_agents)]
    step(s, acts, num_agents)
    return _score_for(s, me, num_agents)


def _rollout(state, me, num_agents, first_move, depth, deadline):
    s = clone(state)
    for p in range(num_agents):
        _ROLLOUT_POOL[p].reset(_opp_view(s, p))
    acts = [first_move if p == me else _ROLLOUT_POOL[p].act(_opp_view(s, p))
            for p in range(num_agents)]
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
    owners = {int(p[1]) for p in planets if int(p[1]) >= 0}
    num_agents = 4 if (owners and max(owners) >= 2) else 2

    v4_move = _v4_real_agent(obs)
    marco_move = _marco_real_agent(obs, config)
    pi_move = _pi_move(obs, me)

    # Challengers distinct from v4; pick ONE via cheap 1-ply prefilter.
    state = new_state(obs)
    challengers = [m for m in (marco_move, pi_move) if m and m != v4_move]
    if not challengers:
        return v4_move
    if len(challengers) == 1:
        chal = challengers[0]
    else:
        chal = max(challengers,
                   key=lambda m: _prefilter(state, me, num_agents, m))

    now = time.perf_counter()
    per = max(0.0, (deadline - now) / 2.0)
    s_v4 = _rollout(state, me, num_agents, v4_move, ROLLOUT_DEPTH, now + per)
    now2 = time.perf_counter()
    s_ch = _rollout(state, me, num_agents, chal, ROLLOUT_DEPTH, now2 + per)

    margin = max(10.0, 0.05 * abs(s_v4))
    if s_ch > s_v4 + margin:
        return chal
    return v4_move


__all__ = ["agent"]
'''

MAIN = (HEAD.replace("@@V4@@", v4_b64).replace("@@MARCO@@", marco_b64)
        .replace("@@PIW@@", W_B64).replace("@@FASTSIM@@", FAST_SIM_BODY)
        .replace("@@ENC@@", ENC_BLOCK).replace("@@INF@@", INF_BLOCK))
out = ROOT / "submit" / OUT_NAME
out.write_text(MAIN)
print(f"Wrote {out} ({len(MAIN)} bytes)  weights={W_PATH.name}")
print("v4-opponent rollout (ensemble_fixed core) + π_θ candidate + 1-ply prefilter")
