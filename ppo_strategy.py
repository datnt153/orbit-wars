from __future__ import annotations

import contextlib
import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_BASELINE_MODULE = None
_MODEL_CACHE: dict[Path, "NumpyStrategyPolicy | None"] = {}
DEFAULT_MODEL_PATH = Path(__file__).with_name("ppo_strategy_policy.npz")


def set_baseline_module(module: Any) -> None:
    global _BASELINE_MODULE
    _BASELINE_MODULE = module


def baseline_bot():
    global _BASELINE_MODULE
    if _BASELINE_MODULE is None:
        _BASELINE_MODULE = importlib.import_module("main")
    return _BASELINE_MODULE


@dataclass(frozen=True)
class StrategyPreset:
    name: str
    overrides: dict[str, float | int]


STRATEGY_PRESETS: tuple[StrategyPreset, ...] = (
    StrategyPreset("balanced", {}),
    StrategyPreset(
        "static_expand",
        {
            "STATIC_NEUTRAL_VALUE_MULT": 1.58,
            "EARLY_STATIC_NEUTRAL_SCORE_MULT": 1.34,
            "ROTATING_OPENING_VALUE_MULT": 0.82,
            "DENSE_ROTATING_NEUTRAL_SCORE_MULT": 0.79,
            "COMET_VALUE_MULT": 0.55,
        },
    ),
    StrategyPreset(
        "comet_hunter",
        {
            "COMET_VALUE_MULT": 1.22,
            "COMET_MAX_CHASE_TURNS": 12,
            "COMET_MARGIN_RELIEF": 9,
            "LOW_VALUE_COMET_PRODUCTION": 0,
            "STATIC_NEUTRAL_VALUE_MULT": 1.28,
        },
    ),
    StrategyPreset(
        "hostile_rush",
        {
            "HOSTILE_TARGET_VALUE_MULT": 2.15,
            "OPENING_HOSTILE_TARGET_VALUE_MULT": 1.68,
            "FINISHING_HOSTILE_VALUE_MULT": 1.24,
            "PROACTIVE_DEFENSE_RATIO": 0.14,
            "MULTI_ENEMY_PROACTIVE_RATIO": 0.18,
        },
    ),
    StrategyPreset(
        "fortress",
        {
            "REINFORCE_VALUE_MULT": 1.58,
            "DEFENSE_FRONTIER_SCORE_MULT": 1.22,
            "PROACTIVE_DEFENSE_RATIO": 0.24,
            "MULTI_ENEMY_PROACTIVE_RATIO": 0.30,
            "HOSTILE_TARGET_VALUE_MULT": 1.62,
        },
    ),
    StrategyPreset(
        "swarm_tactics",
        {
            "SWARM_VALUE_MULT": 1.18,
            "SNIPE_VALUE_MULT": 1.18,
            "CRASH_EXPLOIT_VALUE_MULT": 1.26,
            "MULTI_SOURCE_PLAN_PENALTY": 1.00,
            "THREE_SOURCE_PLAN_PENALTY": 0.97,
            "MULTI_SOURCE_TOP_K": 6,
        },
    ),
    StrategyPreset(
        "rotating_gambit",
        {
            "ROTATING_OPENING_VALUE_MULT": 1.08,
            "FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT": 0.95,
            "ROTATING_OPENING_MAX_TURNS": 15,
            "SAFE_OPENING_TURN_LIMIT": 12,
            "STATIC_NEUTRAL_VALUE_MULT": 1.30,
        },
    ),
)


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom else 0.0


def _safe_mean(values: list[float]) -> float:
    return float(sum(values)) / len(values) if values else 0.0


def _planet_ship_stats(planets) -> tuple[float, float, float]:
    if not planets:
        return 0.0, 0.0, 0.0
    ships = [float(planet.ships) for planet in planets]
    return min(ships), _safe_mean(ships), max(ships)


def _fleet_ships(fleets, owner: int | None = None) -> float:
    total = 0.0
    for fleet in fleets:
        if owner is None or fleet.owner == owner:
            total += float(fleet.ships)
    return total


def _nearest_distance(bot, src_planets, dst_planets) -> float:
    if not src_planets or not dst_planets:
        return 100.0
    return min(
        bot.dist(src.x, src.y, dst.x, dst.y)
        for src in src_planets
        for dst in dst_planets
    )


def _reaction_gap_features(world, policy) -> tuple[float, float]:
    safe_neutral = 0
    contested = 0
    neutral_total = max(1, len(world.neutral_planets))
    for planet in world.neutral_planets:
        my_t, enemy_t = policy["reaction_time_map"].get(planet.id, (10**9, 10**9))
        if my_t <= enemy_t - 2:
            safe_neutral += 1
        elif abs(my_t - enemy_t) <= 2:
            contested += 1
    return safe_neutral / neutral_total, contested / neutral_total


def _reserve_pressure(world, policy) -> tuple[float, float]:
    reserve_ratios = []
    attack_ratios = []
    for planet in world.my_planets:
        ships = max(1.0, float(planet.ships))
        reserve = float(policy["reserve"].get(planet.id, 0))
        attack_budget = float(policy["attack_budget"].get(planet.id, 0))
        reserve_ratios.append(reserve / ships)
        attack_ratios.append(attack_budget / ships)
    return _safe_mean(reserve_ratios), _safe_mean(attack_ratios)


def extract_features(world, policy=None, modes=None) -> np.ndarray:
    bot = baseline_bot()
    if policy is None:
        policy = bot.build_policy_state(world)
    if modes is None:
        modes = bot.build_modes(world)

    total_planets = max(1, len(world.planets))
    visible_ships = max(1.0, float(world.total_visible_ships))
    total_prod = max(1.0, float(world.total_production))
    neutral_total = max(1, len(world.neutral_planets))
    comet_ids = list(world.comet_ids)
    comet_lives = [float(world.comet_life(pid)) for pid in comet_ids]
    my_min_ship, my_mean_ship, my_max_ship = _planet_ship_stats(world.my_planets)
    enemy_min_ship, enemy_mean_ship, enemy_max_ship = _planet_ship_stats(world.enemy_planets)
    safe_neutral_ratio, contested_neutral_ratio = _reaction_gap_features(world, policy)
    reserve_ratio, attack_ratio = _reserve_pressure(world, policy)

    owned_comets = sum(
        1
        for pid in comet_ids
        if pid in world.planet_by_id and world.planet_by_id[pid].owner == world.player
    )
    enemy_owned_comets = sum(
        1
        for pid in comet_ids
        if pid in world.planet_by_id and world.planet_by_id[pid].owner not in (-1, world.player)
    )
    rotating_neutrals = len(world.neutral_planets) - len(world.static_neutral_planets)
    my_frontier = _nearest_distance(bot, world.my_planets, world.enemy_planets or world.neutral_planets)
    enemy_frontier = _nearest_distance(bot, world.enemy_planets, world.my_planets)
    my_fleet_ships = _fleet_ships(world.fleets, world.player)
    enemy_fleet_ships = _fleet_ships(world.fleets) - my_fleet_ships
    threatened_planets = sum(
        1 for planet in world.my_planets if world.fall_turn_map.get(planet.id) is not None
    )
    weak_enemy = 1.0 if world.max_enemy_strength <= getattr(bot, "WEAK_ENEMY_THRESHOLD", 45) else 0.0

    feats = np.array(
        [
            _safe_ratio(world.step, float(bot.TOTAL_STEPS)),
            _safe_ratio(world.remaining_steps, float(bot.TOTAL_STEPS)),
            _safe_ratio(len(world.my_planets), total_planets),
            _safe_ratio(len(world.enemy_planets), total_planets),
            _safe_ratio(len(world.neutral_planets), total_planets),
            _safe_ratio(len(world.static_neutral_planets), neutral_total),
            _safe_ratio(rotating_neutrals, neutral_total),
            _safe_ratio(len(comet_ids), total_planets),
            _safe_ratio(owned_comets, max(1, len(comet_ids))),
            _safe_ratio(enemy_owned_comets, max(1, len(comet_ids))),
            _safe_ratio(max(comet_lives, default=0.0), 40.0),
            _safe_ratio(_safe_mean(comet_lives), 40.0),
            _safe_ratio(world.my_total, visible_ships),
            _safe_ratio(world.enemy_total, visible_ships),
            _safe_ratio(world.max_enemy_strength, visible_ships),
            _safe_ratio(world.my_prod, total_prod),
            _safe_ratio(world.enemy_prod, total_prod),
            float(modes["domination"]),
            float(modes["is_behind"]),
            float(modes["is_ahead"]),
            float(modes["is_finishing"]),
            float(world.is_opening),
            float(world.is_late),
            float(world.is_four_player),
            safe_neutral_ratio,
            contested_neutral_ratio,
            reserve_ratio,
            attack_ratio,
            _safe_ratio(my_frontier, 100.0),
            _safe_ratio(enemy_frontier, 100.0),
            _safe_ratio(my_fleet_ships, visible_ships),
            _safe_ratio(enemy_fleet_ships, visible_ships),
            _safe_ratio(my_mean_ship, 100.0),
            _safe_ratio(my_max_ship, 100.0),
            _safe_ratio(enemy_mean_ship, 100.0),
            _safe_ratio(enemy_max_ship, 100.0),
            _safe_ratio(threatened_planets, max(1, len(world.my_planets))),
            weak_enemy,
        ],
        dtype=np.float32,
    )
    return feats


def progress_score(world) -> float:
    ship_term = _safe_ratio(world.my_total - world.enemy_total, world.my_total + world.enemy_total)
    prod_term = _safe_ratio(world.my_prod - world.enemy_prod, world.my_prod + world.enemy_prod)
    planet_term = _safe_ratio(
        len(world.my_planets) - len(world.enemy_planets),
        len(world.my_planets) + len(world.enemy_planets),
    )
    comet_term = _safe_ratio(
        sum(
            1
            for pid in world.comet_ids
            if pid in world.planet_by_id and world.planet_by_id[pid].owner == world.player
        ),
        max(1, len(world.comet_ids)),
    )
    return 0.50 * ship_term + 0.30 * prod_term + 0.15 * planet_term + 0.05 * comet_term


class NumpyStrategyPolicy:
    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        policy_w: np.ndarray,
        policy_b: np.ndarray,
        value_w: np.ndarray,
        value_b: np.ndarray,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
    ):
        self.w1 = w1.astype(np.float32)
        self.b1 = b1.astype(np.float32)
        self.w2 = w2.astype(np.float32)
        self.b2 = b2.astype(np.float32)
        self.policy_w = policy_w.astype(np.float32)
        self.policy_b = policy_b.astype(np.float32)
        self.value_w = value_w.astype(np.float32)
        self.value_b = value_b.astype(np.float32)
        self.obs_mean = None if obs_mean is None else obs_mean.astype(np.float32)
        self.obs_std = None if obs_std is None else obs_std.astype(np.float32)

    @classmethod
    def load(cls, path: str | Path):
        payload = np.load(path)
        return cls(
            w1=payload["w1"],
            b1=payload["b1"],
            w2=payload["w2"],
            b2=payload["b2"],
            policy_w=payload["policy_w"],
            policy_b=payload["policy_b"],
            value_w=payload["value_w"],
            value_b=payload["value_b"],
            obs_mean=payload["obs_mean"] if "obs_mean" in payload else None,
            obs_std=payload["obs_std"] if "obs_std" in payload else None,
        )

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        if self.obs_mean is None or self.obs_std is None:
            return features.astype(np.float32, copy=False)
        return (features.astype(np.float32, copy=False) - self.obs_mean) / np.maximum(
            self.obs_std,
            1e-6,
        )

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        x = self._normalize(features)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        logits = h2 @ self.policy_w + self.policy_b
        value = float(np.asarray(h2 @ self.value_w + self.value_b).reshape(-1)[0])
        return logits.astype(np.float32), value

    def act(self, features: np.ndarray) -> tuple[int, np.ndarray, float]:
        logits, value = self.forward(features)
        return int(np.argmax(logits)), logits, value


def load_strategy_policy(path: str | Path | None = None) -> NumpyStrategyPolicy | None:
    resolved = Path(path) if path is not None else DEFAULT_MODEL_PATH
    cached = _MODEL_CACHE.get(resolved)
    if resolved in _MODEL_CACHE:
        return cached
    if not resolved.exists():
        _MODEL_CACHE[resolved] = None
        return None
    model = NumpyStrategyPolicy.load(resolved)
    _MODEL_CACHE[resolved] = model
    return model


def heuristic_strategy_id(world, policy=None, modes=None) -> int:
    bot = baseline_bot()
    if policy is None:
        policy = bot.build_policy_state(world)
    if modes is None:
        modes = bot.build_modes(world)

    if modes["is_finishing"] and world.is_late:
        return 3
    if any(world.fall_turn_map.get(planet.id) is not None for planet in world.my_planets):
        return 4
    if world.is_opening and len(world.static_neutral_planets) >= 4:
        return 1
    if world.comet_ids:
        live_life = max((world.comet_life(pid) for pid in world.comet_ids), default=0)
        if live_life >= 5 and world.step >= 45:
            return 2
    if modes["is_behind"]:
        return 5 if world.enemy_planets else 6
    if world.is_four_player and not world.static_neutral_planets:
        return 6
    return 0


def choose_strategy(world, policy=None, modes=None, model=None) -> tuple[int, np.ndarray | None, float | None]:
    if policy is None:
        policy = baseline_bot().build_policy_state(world)
    if modes is None:
        modes = baseline_bot().build_modes(world)
    heuristic = heuristic_strategy_id(world, policy=policy, modes=modes)
    if model is None:
        return heuristic, None, None
    features = extract_features(world, policy=policy, modes=modes)
    action, logits, value = model.act(features)
    if logits.size >= 2:
        top2 = np.partition(logits, -2)[-2:]
        confidence = float(top2[-1] - top2[-2])
    else:
        confidence = 1.0
    if confidence < 0.35:
        action = heuristic
    return action, logits, value


@contextlib.contextmanager
def apply_strategy(strategy_id: int):
    bot = baseline_bot()
    preset = STRATEGY_PRESETS[int(strategy_id)]
    original = {name: getattr(bot, name) for name in preset.overrides}
    for name, value in preset.overrides.items():
        setattr(bot, name, value)
    try:
        yield preset
    finally:
        for name, value in original.items():
            setattr(bot, name, value)


def _read(obs_or_cfg, key, default=None):
    if obs_or_cfg is None:
        return default
    if isinstance(obs_or_cfg, dict):
        return obs_or_cfg.get(key, default)
    return getattr(obs_or_cfg, key, default)


def plan_with_strategy(world, strategy_id: int, config=None, policy=None, modes=None, deadline=None):
    bot = baseline_bot()
    if not world.my_planets:
        return []
    if deadline is None:
        start_time = time.perf_counter()
        act_timeout = _read(config, "actTimeout", 1.0)
        soft_budget = min(bot.SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
        deadline = start_time + soft_budget
    if modes is None:
        modes = bot.build_modes(world)
    if policy is None:
        policy = bot.build_policy_state(world, deadline=deadline)
    with apply_strategy(strategy_id):
        return bot.plan_moves(world, deadline=deadline, modes=modes, policy=policy)


def agent(obs, config=None, model_path: str | Path | None = None):
    bot = baseline_bot()
    world = bot.build_world(obs)
    if not world.my_planets:
        return []
    start_time = time.perf_counter()
    act_timeout = _read(config, "actTimeout", 1.0)
    soft_budget = min(bot.SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
    deadline = start_time + soft_budget
    modes = bot.build_modes(world)
    policy = bot.build_policy_state(world, deadline=deadline)
    model = load_strategy_policy(model_path)
    strategy_id, _, _ = choose_strategy(world, policy=policy, modes=modes, model=model)
    return plan_with_strategy(
        world,
        strategy_id,
        config=config,
        policy=policy,
        modes=modes,
        deadline=deadline,
    )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "NumpyStrategyPolicy",
    "STRATEGY_PRESETS",
    "agent",
    "apply_strategy",
    "baseline_bot",
    "choose_strategy",
    "extract_features",
    "heuristic_strategy_id",
    "load_strategy_policy",
    "plan_with_strategy",
    "progress_score",
    "set_baseline_module",
]
