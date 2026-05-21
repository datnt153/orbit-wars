"""Fast exact Orbit Wars simulator.

Port of kaggle_environments/envs/orbit_wars/orbit_wars.py `interpreter()`
game loop (the post-init part, lines 419-711), as a pure function with no
kaggle_environments Env overhead.

State is a plain dict that mirrors the engine's observation fields. step()
mutates a deep-copied state and returns it. Combat / sweep / production
order / fleet speed match the engine bit-for-bit.

Limitation: comet *spawning* (engine lines 441-477) needs random ship
counts we cannot predict from an observation, so step() does NOT spawn new
comets. Existing comets advance exactly. For MCTS lookaheads that don't
cross a spawn step (50/150/250/350/450) this is exact; across one it's an
approximation (no new comets appear).
"""
from __future__ import annotations

import copy
import math

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0


def _distance(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _point_to_segment_distance(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    if l2 == 0.0:
        return _distance(px, py, vx, vy)
    t = max(0.0, min(1.0, ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2))
    projx = vx + t * (wx - vx)
    projy = vy + t * (wy - vy)
    return _distance(px, py, projx, projy)


def new_state(obs, *, ship_speed=6.0, episode_steps=500):
    """Build a sim state from an agent observation (dict or namespace)."""
    def g(key, default):
        if isinstance(obs, dict):
            return obs.get(key, default)
        return getattr(obs, key, default)

    return {
        "planets": [list(p) for p in g("planets", [])],
        "fleets": [list(f) for f in g("fleets", [])],
        "initial_planets": [list(p) for p in g("initial_planets", [])],
        "comets": copy.deepcopy(g("comets", [])),
        "comet_planet_ids": list(g("comet_planet_ids", [])),
        "next_fleet_id": g("next_fleet_id", 0),
        "angular_velocity": g("angular_velocity", 0.0),
        "step": g("step", 0),
        "ship_speed": ship_speed,
        "episode_steps": episode_steps,
    }


def clone(state):
    return {
        "planets": [p[:] for p in state["planets"]],
        "fleets": [f[:] for f in state["fleets"]],
        "initial_planets": [p[:] for p in state["initial_planets"]],
        "comets": copy.deepcopy(state["comets"]),
        "comet_planet_ids": list(state["comet_planet_ids"]),
        "next_fleet_id": state["next_fleet_id"],
        "angular_velocity": state["angular_velocity"],
        "step": state["step"],
        "ship_speed": state["ship_speed"],
        "episode_steps": state["episode_steps"],
    }


def step(state, joint_actions, num_agents):
    """Advance one turn. joint_actions: list[num_agents] of move lists.

    Each move = [from_planet_id, angle, ships]. Mutates & returns state.
    Adds state['_done'] and state['_scores'] when terminated.
    """
    planets = state["planets"]
    fleets = state["fleets"]
    av = state["angular_velocity"]
    step_no = state["step"]

    # --- 0a. Remove expired comets before fleet launch (engine 419-439) ---
    expired = []
    for group in state["comets"]:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired.append(pid)
    if expired:
        eset = set(expired)
        state["planets"] = planets = [p for p in planets if p[0] not in eset]
        state["initial_planets"] = [
            p for p in state["initial_planets"] if p[0] not in eset
        ]
        state["comet_planet_ids"] = [
            pid for pid in state["comet_planet_ids"] if pid not in eset
        ]
        for group in state["comets"]:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in eset
            ]
        state["comets"] = [g for g in state["comets"] if g["planet_ids"]]

    # --- 0b. Comet spawning: SKIPPED (cannot predict random ship count) ---

    # --- 1. Fleet launch (engine 480-512) ---
    pmap = {p[0]: p for p in planets}
    for pid in range(num_agents):
        action = joint_actions[pid] if pid < len(joint_actions) else None
        if not action or not isinstance(action, list):
            continue
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            fp = pmap.get(from_id)
            if fp is not None and fp[1] == pid:
                if fp[5] >= ships and ships > 0:
                    fp[5] -= ships
                    sx = fp[2] + math.cos(angle) * (fp[4] + 0.1)
                    sy = fp[3] + math.sin(angle) * (fp[4] + 0.1)
                    fleets.append(
                        [state["next_fleet_id"], pid, sx, sy, angle, from_id, ships]
                    )
                    state["next_fleet_id"] += 1

    # --- 2. Production (engine 514-517) ---
    for p in planets:
        if p[1] != -1:
            p[5] += p[6]

    # --- 3. Fleet movement + continuous collision (engine 519-551) ---
    max_speed = state["ship_speed"]
    to_remove_idx = set()
    combat = {p[0]: [] for p in planets}
    for fi, fleet in enumerate(fleets):
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        ox, oy = fleet[2], fleet[3]
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        nx, ny = fleet[2], fleet[3]
        if not (0 <= nx <= BOARD_SIZE and 0 <= ny <= BOARD_SIZE):
            to_remove_idx.add(fi)
            continue
        if _point_to_segment_distance(CENTER, CENTER, ox, oy, nx, ny) < SUN_RADIUS:
            to_remove_idx.add(fi)
            continue
        for p in planets:
            if _point_to_segment_distance(p[2], p[3], ox, oy, nx, ny) < p[4]:
                combat[p[0]].append(fleet)
                to_remove_idx.add(fi)
                break

    # --- 4. Planet rotation + sweep (engine 553-610) ---
    comet_pids = set(state["comet_planet_ids"])
    init_by_id = {p[0]: p for p in state["initial_planets"]}

    def sweep(planet, ox, oy, nx, ny):
        if ox == nx and oy == ny:
            return
        for fi, fleet in enumerate(fleets):
            if fi in to_remove_idx:
                continue
            if _point_to_segment_distance(fleet[2], fleet[3], ox, oy, nx, ny) < planet[4]:
                combat[planet[0]].append(fleet)
                to_remove_idx.add(fi)

    for planet in planets:
        if planet[0] in comet_pids:
            continue
        ip = init_by_id.get(planet[0])
        if not ip:
            continue
        dx = ip[2] - CENTER
        dy = ip[3] - CENTER
        r = math.sqrt(dx * dx + dy * dy)
        ox, oy = planet[2], planet[3]
        if r + planet[4] < ROTATION_RADIUS_LIMIT:
            ia = math.atan2(dy, dx)
            ca = ia + av * step_no
            planet[2] = CENTER + r * math.cos(ca)
            planet[3] = CENTER + r * math.sin(ca)
        sweep(planet, ox, oy, planet[2], planet[3])

    # Comet movement (engine 592-610)
    expired2 = []
    for group in state["comets"]:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = pmap.get(pid)
            if planet is None:
                continue
            ppath = group["paths"][i]
            if idx >= len(ppath):
                expired2.append(pid)
            else:
                ox, oy = planet[2], planet[3]
                planet[2] = ppath[idx][0]
                planet[3] = ppath[idx][1]
                if ox >= 0:
                    sweep(planet, ox, oy, planet[2], planet[3])

    if expired2:
        eset = set(expired2)
        state["planets"] = planets = [p for p in planets if p[0] not in eset]
        state["initial_planets"] = [
            p for p in state["initial_planets"] if p[0] not in eset
        ]
        state["comet_planet_ids"] = [
            pid for pid in state["comet_planet_ids"] if pid not in eset
        ]
        for group in state["comets"]:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in eset
            ]
        state["comets"] = [g for g in state["comets"] if g["planet_ids"]]
        combat = {k: v for k, v in combat.items() if k not in eset}

    state["fleets"] = [f for i, f in enumerate(fleets) if i not in to_remove_idx]

    # --- 5. Combat resolution (engine 630-669) ---
    pmap = {p[0]: p for p in state["planets"]}
    for pid, pf in combat.items():
        planet = pmap.get(pid)
        if not planet or not pf:
            continue
        player_ships = {}
        for fleet in pf:
            player_ships[fleet[1]] = player_ships.get(fleet[1], 0) + fleet[6]
        if not player_ships:
            continue
        sp = sorted(player_ships.items(), key=lambda kv: kv[1], reverse=True)
        top_player, top_ships = sp[0]
        if len(sp) > 1:
            second = sp[1][1]
            surv = top_ships - second
            if sp[0][1] == sp[1][1]:
                surv = 0
            surv_owner = top_player if surv > 0 else -1
        else:
            surv_owner = top_player
            surv = top_ships
        if surv > 0:
            if planet[1] == surv_owner:
                planet[5] += surv
            else:
                planet[5] -= surv
                if planet[5] < 0:
                    planet[1] = surv_owner
                    planet[5] = abs(planet[5])

    # --- 6. Termination + scoring (engine 678-711) ---
    state["step"] = step_no + 1
    terminated = False
    if state["step"] >= state["episode_steps"] - 2:
        terminated = True
    alive = set()
    for p in state["planets"]:
        if p[1] != -1:
            alive.add(p[1])
    for f in state["fleets"]:
        alive.add(f[1])
    if len(alive) <= 1:
        terminated = True

    state["_done"] = terminated
    if terminated:
        scores = [0] * num_agents
        for p in state["planets"]:
            if p[1] != -1 and p[1] < num_agents:
                scores[p[1]] += p[5]
        for f in state["fleets"]:
            if f[1] < num_agents:
                scores[f[1]] += f[6]
        state["_scores"] = scores
        mx = max(scores) if scores else 0
        state["_rewards"] = [
            1 if (s == mx and mx > 0) else -1 for s in scores
        ]
    return state
