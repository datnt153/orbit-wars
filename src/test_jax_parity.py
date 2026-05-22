"""Parity tests: JAX env vs fast_sim reference. Grows one phase at a time.

Run: .venv/bin/python src/test_jax_parity.py
"""
import glob
import json
from pathlib import Path

import numpy as np

import fast_sim
import jax_env

ROOT = Path(__file__).resolve().parent.parent


def load_state():
    p = sorted(glob.glob(str(ROOT / "data" / "bovard" / "**" / "*.json"),
                         recursive=True))[0]
    obs0 = json.loads(open(p).read())["steps"][0][0]["observation"]
    return fast_sim.new_state(obs0, ship_speed=6.0, episode_steps=500)


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


if __name__ == "__main__":
    results = [test_roundtrip(), test_production()]
    print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'} "
          f"({sum(results)}/{len(results)})")
