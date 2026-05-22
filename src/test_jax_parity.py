"""Parity tests: JAX env vs fast_sim reference. Grows one phase at a time.

Run: .venv/bin/python src/test_jax_parity.py
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import fast_sim
import jax_env

ROOT = Path(__file__).resolve().parent.parent
NUM_AGENTS = 4


def load_state(strip_comets=True):
    p = sorted(glob.glob(str(ROOT / "data" / "bovard" / "**" / "*.json"),
                         recursive=True))[0]
    obs0 = json.loads(open(p).read())["steps"][0][0]["observation"]
    s = fast_sim.new_state(obs0, ship_speed=6.0, episode_steps=500)
    if strip_comets:
        # JAX env defers comets (jax_port.md); compare on comet-free dynamics.
        s["comets"] = []
        s["comet_planet_ids"] = []
    return s


def _planets_close(a, b, atol=1e-3):
    if len(a) != len(b):
        return False, f"planet count {len(a)} vs {len(b)}"
    for i, (pa, pb) in enumerate(zip(a, b)):
        if int(pa[0]) != int(pb[0]) or int(pa[1]) != int(pb[1]):
            return False, f"planet {i} id/owner {pa[:2]} vs {pb[:2]}"
        if not np.allclose(pa[2:], pb[2:], atol=atol):
            return False, f"planet {i} fields {pa[2:]} vs {pb[2:]}"
    return True, ""


def _fleets_close(a, b, atol=1e-3):
    if len(a) != len(b):
        return False, f"fleet count {len(a)} vs {len(b)}"
    for i, (fa, fb) in enumerate(zip(a, b)):
        if int(fa[0]) != int(fa[0]) or int(fa[1]) != int(fb[1]):
            return False, f"fleet {i} id/owner"
        if not np.allclose([fa[2], fa[3], fa[4], fa[6]],
                           [fb[2], fb[3], fb[4], fb[6]], atol=atol):
            return False, f"fleet {i} fields"
    return True, ""


def test_roundtrip():
    s = load_state()
    js = jax_env.to_jax(s)
    s2 = jax_env.from_jax(js)
    okp, msgp = _planets_close(s["planets"], s2["planets"])
    okf, msgf = _fleets_close(s["fleets"], s2["fleets"])
    scal = (s["next_fleet_id"] == s2["next_fleet_id"]
            and abs(s["angular_velocity"] - s2["angular_velocity"]) < 1e-6
            and s["step"] == s2["step"])
    ok = okp and okf and scal
    print(f"[roundtrip] planets={okp} fleets={okf} scalars={scal} "
          f"({len(s['planets'])} planets, {len(s['fleets'])} fleets)"
          f"{'' if ok else '  FAIL: ' + (msgp or msgf)}")
    return ok


def test_production():
    s = load_state()
    # reference: owned planets ships += prod (fast_sim phase 2, lines 135-138)
    ref = {p[0]: (p[5] + p[6] if p[1] != -1 else p[5]) for p in s["planets"]}
    js2 = jax_env.production(jax_env.to_jax(s))
    s2 = jax_env.from_jax(js2)
    got = {p[0]: p[5] for p in s2["planets"]}
    ok = all(abs(ref[k] - got[k]) < 1e-3 for k in ref)
    bad = [k for k in ref if abs(ref[k] - got[k]) >= 1e-3]
    print(f"[production] {'PASS' if ok else 'FAIL'} "
          f"({sum(1 for p in s['planets'] if p[1] != -1)} owned planets)"
          f"{'' if ok else '  bad ids: ' + str(bad[:5])}")
    return ok


def random_action(state, rng):
    """Per-planet action. Picks integer ships k, sets frac=(k+0.5)/ships so
    floor(frac*ships)==k identically in float32 (JAX) and float64 (fast_sim)."""
    P = state["planets"]
    n = len(P)
    launch = np.zeros(jax_env.MAXP, np.bool_)
    target = np.zeros(jax_env.MAXP, np.int32)
    frac = np.zeros(jax_env.MAXP, np.float32)
    for i, p in enumerate(P):
        owner = p[1]
        if owner < 0 or owner >= NUM_AGENTS or p[5] < 1:
            continue
        if rng.random() < 0.5:
            continue
        k = int(rng.integers(1, int(p[5]) + 1))   # 1..floor(ships)
        launch[i] = True
        target[i] = int(rng.integers(0, n))
        frac[i] = (k + 0.5) / p[5]
    return {"launch": launch, "target": target, "frac": frac}


def derive_moves(state, action):
    """Same semantics the JAX launch phase uses -> fast_sim move lists."""
    P = state["planets"]
    moves = [[] for _ in range(NUM_AGENTS)]
    for i, p in enumerate(P):
        if not action["launch"][i]:
            continue
        owner = p[1]
        if owner < 0 or owner >= NUM_AGENTS:
            continue
        ships = int(math.floor(action["frac"][i] * p[5]))
        if ships <= 0 or p[5] < ships:
            continue
        tgt = int(action["target"][i])
        if tgt < 0 or tgt >= len(P):
            continue
        angle = math.atan2(P[tgt][3] - p[3], P[tgt][2] - p[2])
        moves[owner].append([p[0], angle, ships])
    return moves


def action_to_jax(action):
    return {"launch": jnp.asarray(action["launch"]),
            "target": jnp.asarray(action["target"]),
            "frac": jnp.asarray(action["frac"])}


def _cmp_planets(ref, got, ptol=2e-2, stol=1e-2):
    ra = {p[0]: p for p in ref["planets"]}
    ga = {p[0]: p for p in got["planets"]}
    if set(ra) != set(ga):
        return False, f"planet ids {set(ra) ^ set(ga)}", 0.0
    md = 0.0
    for k in ra:
        a, b = ra[k], ga[k]
        if int(a[1]) != int(b[1]):
            return False, f"planet {k} owner {a[1]} vs {b[1]}", 0.0
        md = max(md, abs(a[2] - b[2]), abs(a[3] - b[3]))
        if abs(a[2] - b[2]) > ptol or abs(a[3] - b[3]) > ptol:
            return False, f"planet {k} pos {a[2:4]} vs {b[2:4]}", md
        if abs(a[5] - b[5]) > stol:
            return False, f"planet {k} ships {a[5]} vs {b[5]}", md
    return True, "", md


def _cmp_fleets(ref, got, tol=2e-2):
    def key(f):
        return (int(f[1]), round(f[2], 2), round(f[3], 2), round(f[6], 2))
    ra = sorted(ref["fleets"], key=key)
    ga = sorted(got["fleets"], key=key)
    if len(ra) != len(ga):
        return False, f"fleet count {len(ra)} vs {len(ga)}", 0.0
    md = 0.0
    for a, b in zip(ra, ga):
        if int(a[1]) != int(b[1]):
            return False, f"fleet owner {a[1]} vs {b[1]}", md
        for j in (2, 3, 6):
            md = max(md, abs(a[j] - b[j]))
            if abs(a[j] - b[j]) > tol:
                return False, f"fleet field {j} {a[j]} vs {b[j]}", md
    return True, "", md


def test_step_single():
    rng = np.random.default_rng(0)
    fails = 0
    maxd = 0.0
    for trial in range(50):
        s = load_state()
        act = random_action(s, rng)
        ref = fast_sim.step(fast_sim.clone(s), derive_moves(s, act), NUM_AGENTS)
        gjs, done, scores, rew = jax_env.step(jax_env.to_jax(s),
                                              action_to_jax(act), NUM_AGENTS)
        got = jax_env.from_jax(gjs)
        okp, mp, dp = _cmp_planets(ref, got)
        okf, mf, df = _cmp_fleets(ref, got)
        maxd = max(maxd, dp, df)
        if not (okp and okf):
            fails += 1
            if fails <= 3:
                print(f"  [single] trial {trial} FAIL: {mp or mf}")
    ok = fails == 0
    print(f"[step_single] {'PASS' if ok else f'FAIL ({fails}/50)'} "
          f"maxdiff={maxd:.2e}")
    return ok


def test_rollout(n=12):
    rng = np.random.default_rng(7)
    s = load_state()
    js = jax_env.to_jax(s)
    maxd = 0.0
    for t in range(n):
        act = random_action(s, rng)
        s = fast_sim.step(s, derive_moves(s, act), NUM_AGENTS)
        js, done, scores, rew = jax_env.step(js, action_to_jax(act), NUM_AGENTS)
        got = jax_env.from_jax(js)
        okp, mp, dp = _cmp_planets(s, got)
        okf, mf, df = _cmp_fleets(s, got)
        maxd = max(maxd, dp, df)
        if not (okp and okf):
            print(f"[rollout] DIVERGE at step {t}: {mp or mf} maxdiff={maxd:.2e}")
            return False
        if s.get("_done"):
            break
    print(f"[rollout] PASS ({t + 1} steps) maxdiff={maxd:.2e}")
    return True


if __name__ == "__main__":
    results = [test_roundtrip(), test_production(),
               test_step_single(), test_rollout()]
    print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'} "
          f"({sum(results)}/{len(results)})")
