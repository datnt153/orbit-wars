"""Phase 0c — quantify WHY the v4-anchored search fades vs the strong field.

Part A (this file, CPU/$0, no agent wiring): how well does the leaf eval
the ensemble optimizes (`world_score_diff` = ship + 8*prod differential,
ensemble_agent.py:65-82) actually predict the final outcome, by game
phase? The depth-140 rollout's leaf lands MID-game; if the leaf eval is
near-random mid-game, every move ranking the search produces is noise →
this is the core motivation for replacing it with a learned V_phi.

Part B (scaffold, run after verifying v4 loads from agents/): substitute
v4's move for the strong player's actual move at each turn, roll fast_sim
forward on the OTHER players' *actual replay moves* for K turns (replay =
ground-truth opponents, no opponent-model bias), measure the score gap.

Reads data/bovard via replay_parser. Nothing here touches the network.
"""
from __future__ import annotations

import collections
import math

from replay_parser import iter_samples, world_score_diff

# Cap episodes/day (manifest priority = strongest matchups first). Some
# days hold 2600+ replays / 20 GiB — loading all would never finish.
MAX_PER_DAY = 60


def _phase(step: int, n_steps_hint: int = 200) -> str:
    # Episodes end on elimination (often <200) or step 498. Use a fixed
    # 500-turn frame so "early/mid/late" is comparable across episodes.
    frac = step / 500.0
    if frac < 0.20:
        return "open  (0-20%)"
    if frac < 0.40:
        return "early (20-40%)"
    if frac < 0.60:
        return "mid   (40-60%)"
    if frac < 0.80:
        return "late  (60-80%)"
    return "end   (80-100%)"


def part_a() -> None:
    # For each (state, player): x = leaf eval the search optimizes,
    #                           y = final outcome (+1 win / -1 not).
    buckets: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "win": 0, "lead_win": 0, "lead_n": 0,
                 "trail_win": 0, "trail_n": 0, "sx": 0.0, "sy": 0.0,
                 "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
    )
    overall = {"n": 0, "win": 0}
    for s in iter_samples(require_action=False, max_per_day=MAX_PER_DAY):
        if s.status != "ACTIVE":      # skip terminal post-game snapshot
            continue
        x = world_score_diff(s.observation, s.player)
        y = 1.0 if s.final_reward == 1 else -1.0
        b = buckets[_phase(s.step)]
        b["n"] += 1
        overall["n"] += 1
        won = s.final_reward == 1
        b["win"] += won
        overall["win"] += won
        if x > 0:
            b["lead_n"] += 1
            b["lead_win"] += won
        elif x < 0:
            b["trail_n"] += 1
            b["trail_win"] += won
        b["sx"] += x
        b["sy"] += y
        b["sxx"] += x * x
        b["syy"] += y * y
        b["sxy"] += x * y

    base = overall["win"] / max(1, overall["n"])
    print("=" * 78)
    print("PART A — does the search's leaf eval (ship+8·prod diff) predict "
          "winning?")
    print(f"  base rate P(win) over all states = {base:.3f}  "
          f"(4P: score==max>0 → +1)")
    print("  KEY: if P(win | leading) ≈ P(win | trailing) mid-game, the "
          "depth-140")
    print("       rollout's leaf is near-random → move ranking is noise.")
    print("-" * 78)
    print(f"{'phase':<16}{'n':>7}{'P(win|lead)':>13}{'P(win|trail)':>14}"
          f"{'lift':>8}{'corr(x,y)':>11}")
    for ph in ["open  (0-20%)", "early (20-40%)", "mid   (40-60%)",
               "late  (60-80%)", "end   (80-100%)"]:
        if ph not in buckets:
            continue
        b = buckets[ph]
        n = b["n"]
        pl = b["lead_win"] / max(1, b["lead_n"])
        pt = b["trail_win"] / max(1, b["trail_n"])
        cov = b["sxy"] / n - (b["sx"] / n) * (b["sy"] / n)
        vx = b["sxx"] / n - (b["sx"] / n) ** 2
        vy = b["syy"] / n - (b["sy"] / n) ** 2
        corr = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0
        print(f"{ph:<16}{n:>7}{pl:>13.3f}{pt:>14.3f}"
              f"{pl - pt:>8.3f}{corr:>11.3f}")
    print("-" * 78)
    print("Read: 'lift' = P(win|lead) − P(win|trail). Small lift early/mid "
          "= the\n  proxy can't tell winning from losing positions where "
          "the rollout\n  actually evaluates → V_φ (learned outcome "
          "predictor) is the fix.")
    print("=" * 78)


def rating_report() -> None:
    from replay_parser import iter_replays
    win_r, lose_r = [], []
    for _day, _eid, replay, mrow in iter_replays(max_per_day=MAX_PER_DAY):
        sc = (mrow or {}).get("scores") or []
        rew = replay.get("rewards") or []
        for p, r in enumerate(rew):
            if p < len(sc) and sc[p] is not None:
                (win_r if r == 1 else lose_r).append(float(sc[p]))
    def stat(v):
        v = sorted(v)
        return (f"n={len(v)} min={v[0]:.0f} p50={v[len(v)//2]:.0f} "
                f"max={v[-1]:.0f}") if v else "n=0"
    print(f"winner ratings : {stat(win_r)}")
    print(f"loser  ratings : {stat(lose_r)}")
    print(f"v4 baseline rating ≈ 982  →  search's rollout opponent is "
          f"~{(sorted(win_r)[len(win_r)//2] if win_r else 0) - 982:.0f} "
          f"rating points weaker than the median *winner* it actually faces")


class _AttrDict(dict):
    """v4_rudra.py mixes obs.get(...) with obs.angular_velocity. Replay
    observations are plain dicts → expose keys as attributes too."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _fresh_v4():
    """Load agents/v4_rudra.py into a private namespace (globals not
    reentrant — same pattern as src/v4_instance.py but no /tmp path)."""
    import importlib.util
    from pathlib import Path as _P
    p = _P(__file__).resolve().parent.parent / "agents" / "v4_rudra.py"
    spec = importlib.util.spec_from_file_location(f"_v4_{id(object())}", p)
    m = importlib.util.module_from_spec(spec)
    import sys as _s
    _s.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _v4_prime(inst, obs):
    inst.steps = 3
    inst.fleet_trajectories.clear()
    inst.reinforcement_trajectories.clear()
    inst.moving_planets.clear()
    inst.planets_coords.clear()
    try:
        inst.fill_moving_planets(obs)
    except Exception:
        pass


def _obs_view(state, player):
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


def part_b(max_episodes: int = 20, stride: int = 8, horizon: int = 6) -> None:
    """Counterfactual: at the strongest player's decision points, would v4's
    move have done as well? Substitute v4's move; continue p* with v4 for
    `horizon` turns; opponents play their ACTUAL replay moves (ground-truth,
    no opponent-model bias). Compare material vs the strong player's real
    trajectory. Isolates CANDIDATE-MOVE QUALITY (v4 pool ceiling) vs eval.
    """
    import fast_sim
    from replay_parser import iter_replays

    v4 = _fresh_v4()
    n_pts = 0
    v4_worse = 0
    v4_better_eq = 0
    sum_gap = 0.0
    gaps: list[float] = []
    for _day, eid, replay, mrow in iter_replays(max_per_day=max_episodes):
        steps = replay.get("steps") or []
        if len(steps) < horizon + 2:
            continue
        sc = (mrow or {}).get("scores") or []
        if not sc:
            continue
        pstar = max(range(len(sc)), key=lambda i: sc[i])  # highest-rated
        num_agents = len(steps[0])
        for t in range(2, len(steps) - horizon - 1, stride):
            entry = steps[t][pstar]
            if entry.get("status") != "ACTIVE":
                continue
            obs = _AttrDict(entry.get("observation") or {})
            actual_move = entry.get("action") or []
            try:
                inst = v4
                _v4_prime(inst, obs)
                v4_move = inst.agent(obs) or []
            except Exception:
                continue
            if list(v4_move) == list(actual_move):
                continue  # v4 already agrees → not a decision gap
            base = world_score_diff(
                steps[t + horizon][pstar].get("observation") or {}, pstar
            )  # the strong player's REAL material `horizon` turns later
            # Counterfactual: p* plays v4_move now, then v4 for the rest;
            # everyone else replays their actual moves.
            s = fast_sim.new_state(obs)
            acts = []
            for q in range(num_agents):
                if q == pstar:
                    acts.append(v4_move)
                else:
                    acts.append((steps[t][q].get("action") or [])
                                if q < len(steps[t]) else [])
            fast_sim.step(s, acts, num_agents)
            _v4_prime(inst, _AttrDict(_obs_view(s, pstar)))
            for k in range(1, horizon):
                if s.get("_done"):
                    break
                ra = steps[t + k]
                acts = []
                for q in range(num_agents):
                    if q == pstar:
                        acts.append(
                            inst.agent(_AttrDict(_obs_view(s, pstar))) or []
                        )
                    else:
                        acts.append((ra[q].get("action") or [])
                                    if q < len(ra) else [])
                fast_sim.step(s, acts, num_agents)
            cf = world_score_diff(_obs_view(s, pstar), pstar)
            gap = cf - base                     # <0 → v4 worse than strong
            n_pts += 1
            sum_gap += gap
            gaps.append(gap)
            if gap < -1.0:
                v4_worse += 1
            else:
                v4_better_eq += 1
    print("=" * 78)
    print("PART B — at a ~1550-rated player's decision points, is v4's move "
          "as good?")
    print(f"  decision points sampled (v4 ≠ strong move) : {n_pts}")
    if n_pts:
        gaps.sort()
        print(f"  v4 move WORSE than strong (material)        : "
              f"{v4_worse}  ({v4_worse / n_pts:.0%})")
        print(f"  v4 move ≥ strong                            : "
              f"{v4_better_eq}  ({v4_better_eq / n_pts:.0%})")
        print(f"  mean material gap (v4 − strong, {horizon}-turn) : "
              f"{sum_gap / n_pts:+.1f}")
        print(f"  gap p10/p50/p90                             : "
              f"{gaps[n_pts // 10]:+.1f} / {gaps[n_pts // 2]:+.1f} / "
              f"{gaps[n_pts * 9 // 10]:+.1f}")
    print("-" * 78)
    print("Read: high 'v4 WORSE %' + negative mean gap = the candidate POOL\n"
          "  (v4/marco moves) can't reach strong play → no leaf eval fixes a\n"
          "  choice between two weak moves. Then the lever is a STRONG\n"
          "  candidate generator (π distilled from 1500+ players), still\n"
          "  vetted by the exact-sim rollout + v4 floor (always-safe).")
    print("=" * 78)


if __name__ == "__main__":
    import sys
    rating_report()
    print()
    part_a()
    if "--b" in sys.argv:
        print()
        part_b()
