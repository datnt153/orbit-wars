"""P1 — offline (state → per-planet action) dataset from bovard 1500+.

Distill π_θ (fast rollout opponent + candidate, NOT the submitted policy)
by behavior cloning the strong field. The hard part — and exactly where
the old 8-discrete BC lost precision — is turning a replay's variable
move list `[[from_id, angle, ships], ...]` into a fixed per-owned-planet
label without throwing away target/precision. Design:

For player p at a state, for EACH planet p owns:
  launch    ∈ {0,1}        did p launch from this planet this turn
  target    = planet index  the launch is aimed at (derived from angle +
                             geometry; supports orbiting lead). -1 if none
  ship_frac ∈ (0,1]         ships sent / garrison-before-launch

Approximations (acceptable for an *opponent model*, not a submitted bot):
  - Multiple launches from the same planet in one turn → collapse to the
    single largest (rare; keeps one decision/planet).
  - target = the planet whose direction from the source best matches the
    launch angle (min angular error), among all planets; ties → nearest.
    This is a label-extraction heuristic, not the engine's truth, but it
    recovers intent far better than bucketing angle into 8 bins.

No GPU, no network. Reuses replay_parser (manifest rating filter).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from replay_parser import iter_samples

# Planet row: [id, owner, x, y, radius, ships, production]
_PID, _OWN, _X, _Y, _R, _SHIPS, _PROD = range(7)


@dataclass
class PolicySample:
    episode_id: str
    step: int
    num_agents: int
    player: int
    rating: float
    # state (raw — feature encoder is a separate, swappable concern)
    planets: list
    fleets: list
    # per-owned-planet labels, aligned to `src_planet_ids`
    src_planet_ids: list[int]
    launch: list[int]            # 0/1 per owned planet
    target_idx: list[int]        # index into `planets` (-1 if no launch)
    ship_frac: list[float]       # in (0,1], 0.0 if no launch


def _angle_to(px, py, qx, qy) -> float:
    return math.atan2(qy - py, qx - px)


def _ang_diff(a, b) -> float:
    d = (a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _resolve_target(src, planets, angle, num_agents):
    """Planet index whose bearing from `src` best matches `angle`.

    Prefers non-self-owned planets (the realistic intent of a launch);
    falls back to any. Returns -1 if no planet at all.
    """
    sx, sy = src[_X], src[_Y]
    best_i, best_err = -1, math.inf
    best_any_i, best_any_err = -1, math.inf
    for i, pl in enumerate(planets):
        if pl is src:
            continue
        err = _ang_diff(angle, _angle_to(sx, sy, pl[_X], pl[_Y]))
        if err < best_any_err:
            best_any_err, best_any_i = err, i
        if pl[_OWN] != src[_OWN] and err < best_err:
            best_err, best_i = err, i
    # Only trust the non-owned match if it's reasonably aligned (<~35°);
    # otherwise the launch may be a reinforcement to an owned planet.
    if best_i != -1 and best_err < 0.6:
        return best_i
    return best_any_i


def extract(days=None, min_rating: float = 1500.0, max_per_day=None):
    """Yield PolicySample for every (state, strong player) decision."""
    for s in iter_samples(
        days=days, require_action=True,
        min_rating=min_rating, max_per_day=max_per_day,
    ):
        if s.status != "ACTIVE":
            continue
        planets = s.observation.get("planets", [])
        if not planets:
            continue
        owned = [pl for pl in planets if pl[_OWN] == s.player]
        if not owned:
            continue
        # Aggregate the move list by source planet (collapse to largest).
        by_src: dict[int, list] = {}
        for mv in s.action or []:
            if not isinstance(mv, (list, tuple)) or len(mv) != 3:
                continue
            fid, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
            if ships <= 0:
                continue
            cur = by_src.get(fid)
            if cur is None or ships > cur[2]:
                by_src[fid] = [fid, ang, ships]

        src_ids, launch, tgt, frac = [], [], [], []
        for pl in owned:
            pid = pl[_PID]
            src_ids.append(pid)
            mv = by_src.get(pid)
            if mv is None or pl[_SHIPS] <= 0:
                launch.append(0)
                tgt.append(-1)
                frac.append(0.0)
                continue
            launch.append(1)
            tgt.append(_resolve_target(pl, planets, mv[1], s.num_agents))
            frac.append(min(1.0, mv[2] / max(1, pl[_SHIPS])))

        yield PolicySample(
            episode_id=s.episode_id, step=s.step, num_agents=s.num_agents,
            player=s.player, rating=s.rating or 0.0,
            planets=planets, fleets=s.observation.get("fleets", []),
            src_planet_ids=src_ids, launch=launch,
            target_idx=tgt, ship_frac=frac,
        )


if __name__ == "__main__":
    import collections

    n = n_owned = n_launch = 0
    frac_hist = collections.Counter()
    tgt_rel = collections.Counter()   # launch target ownership relation
    ratings = []
    bad_target = 0
    for ps in extract(min_rating=1500.0, max_per_day=60):
        n += 1
        ratings.append(ps.rating)
        n_owned += len(ps.src_planet_ids)
        for k, ln in enumerate(ps.launch):
            if not ln:
                continue
            n_launch += 1
            frac_hist[round(ps.ship_frac[k], 1)] += 1
            ti = ps.target_idx[k]
            if ti < 0 or ti >= len(ps.planets):
                bad_target += 1
                continue
            o = ps.planets[ti][_OWN]
            tgt_rel["self" if o == ps.player
                    else "neutral" if o == -1 else "enemy"] += 1
    print(f"decision states (strong player, ACTIVE) : {n}")
    if n:
        ratings.sort()
        print(f"rating  min={ratings[0]:.0f} p50={ratings[n//2]:.0f} "
              f"max={ratings[-1]:.0f}")
        print(f"owned-planet slots                       : {n_owned}")
        print(f"  with launch                            : {n_launch} "
              f"({n_launch/max(1,n_owned):.0%})  no-launch implicit label")
        print(f"launch target relation                   : "
              f"{dict(tgt_rel)}  bad/unresolved={bad_target}")
        print(f"ship_frac dist (rounded)                 : "
              f"{dict(sorted(frac_hist.items()))}")
        print("→ label sane if: launches mostly target enemy/neutral, "
              "ship_frac spread (not all 1.0), bad_target small.")
