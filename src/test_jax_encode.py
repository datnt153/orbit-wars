"""Parity: jax_encode.encode_one vs policy_encode.encode_state (numpy).

Run: .venv/bin/python src/test_jax_encode.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import fast_sim
import jax_env
import jax_encode
import policy_encode

ROOT = Path(__file__).resolve().parent.parent


def load_state():
    p = sorted(glob.glob(str(ROOT / "data" / "bovard" / "**" / "*.json"),
                         recursive=True))[0]
    obs0 = json.loads(open(p).read())["steps"][0][0]["observation"]
    s = fast_sim.new_state(obs0, ship_speed=6.0, episode_steps=500)
    s["comets"] = []
    s["comet_planet_ids"] = []
    return s


def main():
    s = load_state()
    js = jax_env.to_jax(s)
    obs = {"planets": s["planets"], "fleets": s["fleets"],
           "comet_planet_ids": [], "step": s["step"]}
    worst = 0.0
    fails = []
    for player in range(4):
        pf_r, pm_r, om_r, gf_r, _ = policy_encode.encode_state(obs, player)
        pf_j, pm_j, om_j, gf_j = jax_encode.encode_one(js, jnp.int32(player))
        for name, r, j in [("pf", pf_r, pf_j), ("pmask", pm_r, pm_j),
                           ("omask", om_r, om_j), ("gf", gf_r, gf_j)]:
            d = float(np.abs(np.asarray(r) - np.asarray(j)).max())
            worst = max(worst, d)
            if d > 1e-3:
                fails.append(f"p{player}/{name}={d:.2e}")
    ok = not fails
    print(f"[encode] {'PASS' if ok else 'FAIL ' + ','.join(fails)}  "
          f"maxdiff={worst:.2e} (4 players)")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
