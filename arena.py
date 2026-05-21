"""Local arena: play N games between 2 agents (or 4 mixed) and report win rate.

Usage:
    .venv/bin/python arena.py <agent_a.py> <agent_b.py> [--games 1000] [--players 2|4] [--workers 8]

Agents are passed as file paths so kaggle_environments loads them via its
exec-source loader — same path that Kaggle's grader uses.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Silence kaggle_environments OpenSpiel log spam
os.environ.setdefault("KAGGLE_LOG_LEVEL", "ERROR")


def play_game(args: tuple[list[str], int, int]) -> tuple[int, list[int]]:
    """Run one game. Returns (seed, [reward_player_0, reward_player_1, ...])."""
    import logging
    logging.disable(logging.CRITICAL)
    agent_paths, num_players, seed = args
    from kaggle_environments import make

    config = {"agents": num_players}
    env = make("orbit_wars", configuration=config, debug=False)
    if num_players == 2:
        agents = [agent_paths[0], agent_paths[1]]
    elif num_players == 4:
        agents = [agent_paths[i % len(agent_paths)] for i in range(4)]
    else:
        raise ValueError("num_players must be 2 or 4")
    env.run(agents)
    final = env.steps[-1]
    rewards = [int(s.reward) if s.reward is not None else 0 for s in final]
    return seed, rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_a", help="Path to agent A (e.g. main.py)")
    parser.add_argument("agent_b", help="Path to agent B")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, choices=[2, 4], default=2)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--alternate", action="store_true", help="Swap A/B sides every other game")
    args = parser.parse_args()

    agent_a = str(Path(args.agent_a).resolve())
    agent_b = str(Path(args.agent_b).resolve())
    print(f"A = {agent_a}")
    print(f"B = {agent_b}")
    print(f"Games: {args.games} | Players: {args.players} | Workers: {args.workers}")

    jobs = []
    for i in range(args.games):
        if args.alternate and i % 2 == 1:
            paths = [agent_b, agent_a]
            swapped = True
        else:
            paths = [agent_a, agent_b]
            swapped = False
        jobs.append((paths, args.players, i, swapped))

    a_wins = 0
    b_wins = 0
    draws = 0
    errors = 0
    completed = 0
    t0 = time.time()

    # Use spawn to avoid forking torch state
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(
            play_game,
            [(p, np_, s) for (p, np_, s, _) in jobs],
            chunksize=1,
        )
        swap_map = {j[2]: j[3] for j in jobs}
        for seed, rewards in results:
            completed += 1
            swapped = swap_map[seed]
            if args.players == 2:
                r0, r1 = rewards[0], rewards[1]
                a_idx, b_idx = (1, 0) if swapped else (0, 1)
                if rewards[a_idx] > rewards[b_idx]:
                    a_wins += 1
                elif rewards[b_idx] > rewards[a_idx]:
                    b_wins += 1
                else:
                    draws += 1
            else:
                # 4P: A plays positions 0,2 (or based on swap), B plays 1,3
                a_indices = [1, 3] if swapped else [0, 2]
                b_indices = [0, 2] if swapped else [1, 3]
                a_score = sum(rewards[i] for i in a_indices)
                b_score = sum(rewards[i] for i in b_indices)
                if a_score > b_score:
                    a_wins += 1
                elif b_score > a_score:
                    b_wins += 1
                else:
                    draws += 1

            if completed % max(1, args.games // 20) == 0 or completed == args.games:
                rate = completed / (time.time() - t0)
                wr = a_wins / max(1, completed) * 100
                print(
                    f"  [{completed:>4d}/{args.games}] A:{a_wins:>4d} B:{b_wins:>4d} D:{draws:>3d} "
                    f"E:{errors:>2d} | A_wr={wr:.1f}% | {rate:.1f} games/s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n=== Final ({elapsed:.1f}s, {completed/elapsed:.1f} games/s) ===")
    total = a_wins + b_wins + draws
    print(f"A wins: {a_wins:>4d} ({a_wins/total*100:.1f}%)")
    print(f"B wins: {b_wins:>4d} ({b_wins/total*100:.1f}%)")
    print(f"Draws : {draws:>4d} ({draws/total*100:.1f}%)")


if __name__ == "__main__":
    main()
