"""Isolated v4 instances for use inside rollouts.

v4 (rule-based submission.py) uses module-level globals (steps,
fleet_trajectories, reinforcement_trajectories, moving_planets,
planets_coords) that accumulate across turns and are NOT reentrant.

This loads v4 into a fresh module namespace per "player slot" so each
slot keeps its own globals through a rollout, then is discarded.

`reset(inst, obs)` primes a slot to act as if mid-game: skip the 2-turn
warmup (steps=3) and detect moving planets from the rollout-start obs.
"""
from __future__ import annotations

import importlib.util
import sys

_V4_PATH = "/tmp/ow_train/rule-based submission.py"
_counter = [0]


def make_v4():
    _counter[0] += 1
    name = f"_v4inst_{_counter[0]}"
    spec = importlib.util.spec_from_file_location(name, _V4_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def reset(inst, obs):
    """Prime a v4 instance to behave as a mid-game agent for `obs`."""
    inst.steps = 3
    inst.fleet_trajectories.clear()
    inst.reinforcement_trajectories.clear()
    inst.moving_planets.clear()
    inst.planets_coords.clear()
    try:
        inst.fill_moving_planets(obs)
    except Exception:
        pass


def act(inst, obs):
    try:
        r = inst.agent(obs)
        return list(r) if r else []
    except Exception:
        return []
