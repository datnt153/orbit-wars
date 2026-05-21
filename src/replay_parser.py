"""Parse bovard top-10% Orbit Wars replays into training/audit records.

Replay schema (verified on data/bovard/2026-04-28/.../75601099.json):
  replay["steps"]    : list[T] of list[N] (N = 2 or 4 agents)
  replay["steps"][t][p] = {action, info, observation, reward, status}
    .observation     : {planets, fleets, initial_planets, comets,
                         comet_planet_ids, next_fleet_id,
                         angular_velocity, step, player}  ← fast_sim-ready
    .action          : [[from_id, angle, ships], ...]  (the move p played)
    .status          : "ACTIVE" | "DONE" | "INACTIVE" | "ERROR"
  replay["rewards"]  : [±1, ...] final per-player (+1 winner)
  replay["configuration"], replay["info"]["Agents"]

manifest.csv (one row per top-10% episode, priority order):
  episode_id, create_time, sum_score, min_score, avg_score,
  scores (json list, per-player Kaggle rating at episode time),
  submission_ids (json list), size_bytes

Index alignment: scores[p] / submission_ids[p] / rewards[p] / steps[t][p]
all use the same player index p (standard kaggle_environments order).

No hardcoded paths. Works directly on observations (fast_sim.new_state
consumes them as-is). Replay rewards are ground truth incl. comets.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "bovard"


@dataclass
class Sample:
    episode_id: str
    day: str
    step: int                 # turn index
    num_agents: int           # 2 or 4
    player: int               # p
    observation: dict         # full obs for player p (fast_sim-ready)
    action: list              # [[from_id, angle, ships], ...] p played, [] = pass
    final_reward: int         # ±1 outcome for player p (value-net label)
    rating: float | None      # player p Kaggle rating at episode time (manifest)
    status: str               # ACTIVE / DONE / ...


def _load_manifest(day_dir: Path) -> dict[str, dict]:
    """episode_id -> {scores: list[float], submission_ids: list[int], ...}."""
    mpath = day_dir / "manifest.csv"
    out: dict[str, dict] = {}
    if not mpath.exists():
        return out
    with mpath.open() as f:
        for row in csv.DictReader(f):
            try:
                row["scores"] = json.loads(row.get("scores") or "[]")
            except Exception:
                row["scores"] = []
            try:
                row["submission_ids"] = json.loads(
                    row.get("submission_ids") or "[]")
            except Exception:
                row["submission_ids"] = []
            out[str(row["episode_id"])] = row
    return out


def iter_replays(
    data_root: Path = DEFAULT_DATA_ROOT,
    days: list[str] | None = None,
    *,
    max_per_day: int | None = None,
) -> Iterator[tuple[str, str, dict, dict | None]]:
    """Yield (day, episode_id, replay_dict, manifest_row|None).

    Episode JSONs live at <root>/<day>/episodes/episodes/<id>.json
    (the dataset zip nests episodes/ twice).

    max_per_day: cap episodes/day, taken in manifest *priority order*
      (highest sum_score first → strongest matchups). Essential — some
      days (e.g. 2026-05-04) hold 2600+ replays / 20 GiB.
    """
    data_root = Path(data_root)
    day_dirs = (
        [data_root / d for d in days]
        if days
        else sorted(p for p in data_root.iterdir() if p.is_dir())
    )
    for day_dir in day_dirs:
        if not day_dir.is_dir():
            continue
        day = day_dir.name
        manifest = _load_manifest(day_dir)
        by_id = {p.stem: p for p in day_dir.glob("episodes/**/*.json")}
        # Manifest rows are already in priority order; fall back to a
        # stable name sort for any episode missing from the manifest.
        ordered_ids = [e for e in manifest if e in by_id]
        ordered_ids += sorted(set(by_id) - set(ordered_ids))
        if max_per_day is not None:
            ordered_ids = ordered_ids[:max_per_day]
        for eid in ordered_ids:
            try:
                replay = json.loads(by_id[eid].read_text())
            except Exception as e:
                print(f"[replay_parser] skip {by_id[eid]}: {e}")
                continue
            yield day, eid, replay, manifest.get(eid)


def iter_samples(
    data_root: Path = DEFAULT_DATA_ROOT,
    days: list[str] | None = None,
    *,
    require_action: bool = False,
    min_rating: float | None = None,
    max_per_day: int | None = None,
) -> Iterator[Sample]:
    """Flatten replays into per-(step, player) Samples.

    require_action: skip terminal/inactive entries with no real action
      (use True for opponent-model data; [] is a *valid* "pass" move and
       is kept — only None/missing action or non-ACTIVE status is dropped).
    min_rating: keep only players whose episode rating >= this
      (use e.g. 1400 to learn the *strong field* for the opponent model).
    """
    for day, eid, replay, mrow in iter_replays(
        data_root, days, max_per_day=max_per_day
    ):
        steps = replay.get("steps") or []
        rewards = replay.get("rewards") or []
        scores = (mrow or {}).get("scores") or []
        # Last step is the terminal post-game snapshot (action == None).
        for t, step in enumerate(steps):
            num_agents = len(step)
            for p, entry in enumerate(step):
                status = entry.get("status", "")
                action = entry.get("action")
                if require_action and (action is None or status != "ACTIVE"):
                    continue
                rating = (
                    float(scores[p]) if p < len(scores) and scores[p] is not None
                    else None
                )
                if min_rating is not None and (
                    rating is None or rating < min_rating
                ):
                    continue
                obs = entry.get("observation") or {}
                yield Sample(
                    episode_id=eid,
                    day=day,
                    step=t,
                    num_agents=num_agents,
                    player=p,
                    observation=obs,
                    action=action if action is not None else [],
                    final_reward=int(rewards[p]) if p < len(rewards) else 0,
                    rating=rating,
                    status=status,
                )


def world_score_diff(obs: dict, player: int) -> float:
    """The exact leaf eval the ensemble search uses (ship + 8*prod diff).

    Lets the counterfactual audit compare alternative moves on the SAME
    scale the agent optimizes — and is the baseline V_phi must beat.
    """
    my = opp = 0.0
    for pl in obs.get("planets", []):
        v = pl[5] + pl[6] * 8.0
        if pl[1] == player:
            my += v
        elif pl[1] >= 0:
            opp += v
    for fl in obs.get("fleets", []):
        if fl[1] == player:
            my += fl[6]
        elif fl[1] >= 0:
            opp += fl[6]
    return my - opp


if __name__ == "__main__":
    import collections

    n_ep = n_smp = 0
    by_day = collections.Counter()
    ratings: list[float] = []
    rew = collections.Counter()
    agents_dist = collections.Counter()
    for day, eid, replay, mrow in iter_replays():
        n_ep += 1
        by_day[day] += 1
        sc = (mrow or {}).get("scores") or []
        ratings += [float(x) for x in sc if x is not None]
        agents_dist[len(replay.get("steps", [[None]])[0])] += 1
    for s in iter_samples():
        n_smp += 1
        rew[s.final_reward] += 1
    print(f"episodes={n_ep}  samples={n_smp}")
    print(f"by day: {dict(sorted(by_day.items()))}")
    print(f"agents/episode (2P vs 4P): {dict(agents_dist)}")
    if ratings:
        ratings.sort()
        n = len(ratings)
        print(
            f"player ratings: n={n} min={ratings[0]:.0f} "
            f"p50={ratings[n // 2]:.0f} p90={ratings[int(n * 0.9)]:.0f} "
            f"max={ratings[-1]:.0f}"
        )
    print(f"final_reward dist: {dict(rew)}  (vs v4 baseline rating ~982)")
