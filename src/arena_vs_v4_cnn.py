"""Arena: CNN policy checkpoint vs rule-based v4 (kaggle_environments).

CNN inference runs in JAX (CPU during arena — slow but fine for a few dozen games).
Greedy gate + sun-masked argmax target + all ships. Gate ≥40 games (4P arena is
high-variance).

Usage: .venv/bin/python src/arena_vs_v4_cnn.py <cnn.npz> [n_games] [agents]
"""
import math
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agents"))

import fast_sim
import jax_env
import jax_raster
import jax_cnn_policy as cnn

_SUN_C, _SUN_R = 50.0, 10.0


def _seg_d(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    if l2 == 0.0:
        return ((px - vx) ** 2 + (py - vy) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2))
    qx, qy = vx + t * (wx - vx), vy + t * (wy - vy)
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def load_cnn(path):
    z = np.load(path)
    return {k[4:]: jnp.asarray(z[k]) for k in z.files if k.startswith("cnn.")}


def make_agent(params):
    def agent(obs):
        try:
            player = int(obs["player"] if isinstance(obs, dict) else obs.player)
            s = fast_sim.new_state(obs, ship_speed=6.0, episode_steps=500)
            s["comets"] = []; s["comet_planet_ids"] = []
            planets = s["planets"]
            if not planets:
                return []
            js = jax_env.to_jax(s)
            grid, ix, pm, om, gf = jax_raster.raster_one(js, jnp.int32(player))
            g, t, _ = cnn.forward(params, grid[None], ix[None], pm[None], gf[None])
            g = np.asarray(g[0]); t = np.asarray(t[0])
            om = np.asarray(om)
            pos = {int(p[0]): (p[2], p[3]) for p in planets}
            sh = {int(p[0]): p[5] for p in planets}
            pids = [int(p[0]) for p in planets]
            moves = []
            for r in np.where(om > 0.5)[0]:
                if r >= len(pids) or g[r] <= 0.0:
                    continue
                sid = pids[r]
                sx, sy = pos[sid]
                row = t[r].copy()
                for c in range(len(pids)):
                    tx, ty = pos[pids[c]]
                    if _seg_d(_SUN_C, _SUN_C, sx, sy, tx, ty) < _SUN_R + 1.0:
                        row[c] = -1e30
                tr = int(np.argmax(row[:len(pids)]))
                if row[tr] <= -1e29 or tr == r:
                    continue
                n = int(sh[sid])
                if n <= 0:
                    continue
                tx, ty = pos[pids[tr]]
                moves.append([sid, math.atan2(ty - sy, tx - sx), n])
            return moves
        except Exception:
            return []
    return agent


def main():
    ckpt = sys.argv[1]
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    agents = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    import v4_rudra
    from kaggle_environments import make

    ppo = make_agent(load_cnn(ckpt))

    def v4(obs):
        try:
            r = v4_rudra.agent(obs)
            return list(r) if r else []
        except Exception:
            return []

    wins = 0
    for gi in range(n_games):
        seat = gi % agents
        lineup = [ppo if k == seat else v4 for k in range(agents)]
        env = make("orbit_wars", configuration={"agents": agents}, debug=False)
        env.run(lineup)
        r = [c.get("reward", 0) for c in env.steps[-1]]
        wins += r[seat] > max(r[k] for k in range(agents) if k != seat)
    print(f"CNN {Path(ckpt).name} vs v4 ({agents}P): win {wins}/{n_games} "
          f"= {wins/n_games:.0%}")


if __name__ == "__main__":
    main()
