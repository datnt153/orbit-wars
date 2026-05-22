"""The TRUE gate: a PPO checkpoint vs rule-based v4 in kaggle_environments.

Self-play awr is NOT ladder strength (124M PPO had awr 0.56 but lost 0/10 to
v4). Gate every promotion / before any submit on THIS instead.

Usage: .venv/bin/python src/arena_vs_v4.py <ckpt.npz> [n_games] [agents]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agents"))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "ppo_best.npz")
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    agents = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    import pi_theta_infer as inf
    import v4_rudra
    from kaggle_environments import make

    z = np.load(ckpt)
    inf._W = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("_")}
    inf._META = (int(z.get("_F", 15)), int(z.get("_G", 8)), int(z.get("_D", 64)))

    def ppo(obs):
        p = obs["player"] if isinstance(obs, dict) else obs.player
        try:
            return inf.decode_moves(obs, int(p))
        except Exception:
            return []

    def v4(obs):
        try:
            r = v4_rudra.agent(obs)
            return list(r) if r else []
        except Exception:
            return []

    def play(lineup):
        env = make("orbit_wars", configuration={"agents": agents}, debug=False)
        env.run(lineup)
        return [s.get("reward", 0) for s in env.steps[-1]]

    wins = ties = 0
    for g in range(n_games):
        ppo_seat = g % agents                       # rotate PPO seat
        lineup = [ppo if s == ppo_seat else v4 for s in range(agents)]
        r = play(lineup)
        pr = r[ppo_seat]
        others = [r[s] for s in range(agents) if s != ppo_seat]
        win = pr > max(others)
        tie = pr == max(others)
        wins += win
        ties += tie and not win
        print(f"game{g} seat{ppo_seat}: ppo={pr} others={others} "
              f"-> {'PPO' if win else ('tie' if tie else 'LOSE')}")
    wr = wins / n_games
    print(f"\nPPO {Path(ckpt).name} vs v4 ({agents}P): "
          f"win {wins}/{n_games} = {wr:.0%} (ties {ties})")
    return wr


if __name__ == "__main__":
    main()
