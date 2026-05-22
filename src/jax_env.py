"""JAX-native Orbit Wars env — fast on-device self-play (removes bottleneck F2).

Port of src/fast_sim.py (parity-verified reference) into JAX: fixed-shape arrays
+ masks instead of Python lists, so the whole rollout JITs/vmaps onto the
accelerator with no per-step host round-trip.

PORTING STATUS (parity-gated, one phase at a time vs fast_sim — see jax_port.md):
  [x] S0 state pytree + dict<->jax converters (round-trip) + phase 2 production
  [ ] S1 phase 1 fleet launch
  [ ] S2 phase 3 fleet movement + collision
  [ ] S3 phase 4 planet rotation + sweep
  [ ] S4 phase 6 combat resolution
  [ ] S5 phase 7 termination + scoring
  [ ] S6 comets (advance; spawn skipped, matches fast_sim)
  [ ] S7 full-step parity  [ ] S8 vmap+lax.scan

float32 (training env); exact submission stays on fast_sim/numpy.
"""
import numpy as np
import jax
import jax.numpy as jnp

MAXP = 48       # max planets
MAXF = 512      # max simultaneous fleets (engine uncapped; 512 is generous)
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0

# planet cols: id, owner, x, y, rad, ships, prod   (fast_sim p[0..6])
# fleet cols:  id, owner, x, y, angle, from_id, ships   (fast_sim f[0..6])


def to_jax(s, maxp=MAXP, maxf=MAXF):
    """fast_sim dict state -> jax pytree (dict of arrays). Pads + masks."""
    planets = s["planets"]
    fleets = s["fleets"]
    init_by_id = {p[0]: p for p in s["initial_planets"]}

    p_id = np.zeros(maxp, np.int32); p_owner = np.full(maxp, -1, np.int32)
    p_x = np.zeros(maxp, np.float32); p_y = np.zeros(maxp, np.float32)
    p_rad = np.zeros(maxp, np.float32); p_ships = np.zeros(maxp, np.float32)
    p_prod = np.zeros(maxp, np.float32); p_valid = np.zeros(maxp, np.bool_)
    p_ix = np.zeros(maxp, np.float32); p_iy = np.zeros(maxp, np.float32)
    for i, p in enumerate(planets):
        if i >= maxp:
            break
        p_id[i] = p[0]; p_owner[i] = p[1]; p_x[i] = p[2]; p_y[i] = p[3]
        p_rad[i] = p[4]; p_ships[i] = p[5]; p_prod[i] = p[6]; p_valid[i] = True
        ip = init_by_id.get(p[0])
        p_ix[i] = ip[2] if ip else p[2]
        p_iy[i] = ip[3] if ip else p[3]

    f_id = np.zeros(maxf, np.int32); f_owner = np.full(maxf, -1, np.int32)
    f_x = np.zeros(maxf, np.float32); f_y = np.zeros(maxf, np.float32)
    f_angle = np.zeros(maxf, np.float32); f_from = np.zeros(maxf, np.int32)
    f_ships = np.zeros(maxf, np.float32); f_valid = np.zeros(maxf, np.bool_)
    for i, f in enumerate(fleets):
        if i >= maxf:
            break
        f_id[i] = f[0]; f_owner[i] = f[1]; f_x[i] = f[2]; f_y[i] = f[3]
        f_angle[i] = f[4]; f_from[i] = f[5]; f_ships[i] = f[6]; f_valid[i] = True

    return {
        "p_id": jnp.asarray(p_id), "p_owner": jnp.asarray(p_owner),
        "p_x": jnp.asarray(p_x), "p_y": jnp.asarray(p_y),
        "p_rad": jnp.asarray(p_rad), "p_ships": jnp.asarray(p_ships),
        "p_prod": jnp.asarray(p_prod), "p_valid": jnp.asarray(p_valid),
        "p_ix": jnp.asarray(p_ix), "p_iy": jnp.asarray(p_iy),
        "f_id": jnp.asarray(f_id), "f_owner": jnp.asarray(f_owner),
        "f_x": jnp.asarray(f_x), "f_y": jnp.asarray(f_y),
        "f_angle": jnp.asarray(f_angle), "f_from": jnp.asarray(f_from),
        "f_ships": jnp.asarray(f_ships), "f_valid": jnp.asarray(f_valid),
        "next_fleet_id": jnp.int32(s["next_fleet_id"]),
        "av": jnp.float32(s["angular_velocity"]),
        "step": jnp.int32(s["step"]),
        "ship_speed": jnp.float32(s["ship_speed"]),
        "episode_steps": jnp.int32(s["episode_steps"]),
    }


def from_jax(js):
    """jax pytree -> fast_sim-style dict (valid entities only, slot order).

    initial_planets/comets are reconstructed minimally (id,owner,x,y,rad from
    p_ix/p_iy) — enough for equality checks on planets/fleets/scalars.
    """
    pv = np.asarray(js["p_valid"])
    fv = np.asarray(js["f_valid"])
    pid = np.asarray(js["p_id"]); pow_ = np.asarray(js["p_owner"])
    px = np.asarray(js["p_x"]); py = np.asarray(js["p_y"])
    prad = np.asarray(js["p_rad"]); psh = np.asarray(js["p_ships"])
    ppr = np.asarray(js["p_prod"]); pix = np.asarray(js["p_ix"]); piy = np.asarray(js["p_iy"])
    fid = np.asarray(js["f_id"]); fow = np.asarray(js["f_owner"])
    fx = np.asarray(js["f_x"]); fy = np.asarray(js["f_y"])
    fan = np.asarray(js["f_angle"]); ffr = np.asarray(js["f_from"]); fsh = np.asarray(js["f_ships"])

    planets = [[int(pid[i]), int(pow_[i]), float(px[i]), float(py[i]),
                float(prad[i]), float(psh[i]), float(ppr[i])]
               for i in range(len(pv)) if pv[i]]
    initial_planets = [[int(pid[i]), int(pow_[i]), float(pix[i]), float(piy[i]),
                        float(prad[i])]
                       for i in range(len(pv)) if pv[i]]
    fleets = [[int(fid[i]), int(fow[i]), float(fx[i]), float(fy[i]),
               float(fan[i]), int(ffr[i]), float(fsh[i])]
              for i in range(len(fv)) if fv[i]]
    return {
        "planets": planets, "fleets": fleets, "initial_planets": initial_planets,
        "comets": [], "comet_planet_ids": [],
        "next_fleet_id": int(js["next_fleet_id"]),
        "angular_velocity": float(js["av"]),
        "step": int(js["step"]),
        "ship_speed": float(js["ship_speed"]),
        "episode_steps": int(js["episode_steps"]),
    }


# ---------------------------------------------------------------- geometry

def _seg_dist(px, py, vx, vy, wx, wy):
    """Point->segment distance, vectorized (broadcasts). Mirrors fast_sim."""
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    safe_l2 = jnp.where(l2 == 0.0, 1.0, l2)
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / safe_l2
    t = jnp.clip(t, 0.0, 1.0)
    projx = vx + t * (wx - vx)
    projy = vy + t * (wy - vy)
    d = jnp.sqrt((px - projx) ** 2 + (py - projy) ** 2)
    d0 = jnp.sqrt((px - vx) ** 2 + (py - vy) ** 2)
    return jnp.where(l2 == 0.0, d0, d)


def _first_true_idx(mask):
    """Index of first True along last axis; -1 if none. (argmax of int.)"""
    any_ = jnp.any(mask, axis=-1)
    idx = jnp.argmax(mask.astype(jnp.int32), axis=-1)
    return jnp.where(any_, idx, -1)


# ---------------------------------------------------------------- phases

def production(js):
    """Phase 2 (fast_sim 135-138): owned planets gain `prod` ships."""
    owned = js["p_valid"] & (js["p_owner"] >= 0)
    p_ships = jnp.where(owned, js["p_ships"] + js["p_prod"], js["p_ships"])
    return {**js, "p_ships": p_ships}


def launch(js, action, num_agents):
    """Phase 1 (fast_sim 113-133): per-planet launch.

    action: dict of [MAXP] arrays: launch (bool), target (int slot), frac.
    angle = atan2(target - source); ships = floor(frac * source_ships).
    New fleets packed into free fleet slots (f_valid == False), in planet-slot
    order. Fleet id order need not match fast_sim (game dynamics ignore id;
    parity compares fleets order-independently).
    """
    p_valid, p_owner = js["p_valid"], js["p_owner"]
    p_x, p_y, p_rad, p_ships = js["p_x"], js["p_y"], js["p_rad"], js["p_ships"]
    tgt = jnp.clip(action["target"], 0, MAXP - 1)
    ships_send = jnp.floor(action["frac"] * p_ships)
    owned = p_valid & (p_owner >= 0) & (p_owner < num_agents)
    do = owned & action["launch"] & (ships_send > 0) & (p_ships >= ships_send)

    angle = jnp.arctan2(p_y[tgt] - p_y, p_x[tgt] - p_x)
    sx = p_x + jnp.cos(angle) * (p_rad + 0.1)
    sy = p_y + jnp.sin(angle) * (p_rad + 0.1)

    # subtract launched ships from source planets
    p_ships2 = p_ships - jnp.where(do, ships_send, 0.0)

    # allocate a free fleet slot per launching planet, in planet-slot order
    f_valid = js["f_valid"]
    free_order = jnp.argsort(f_valid.astype(jnp.int32), stable=True)  # invalid first
    n_free = MAXF - jnp.sum(f_valid.astype(jnp.int32))
    launch_rank = jnp.cumsum(do.astype(jnp.int32)) - 1               # 0-based among launches
    do = do & (launch_rank < n_free)                                 # guard overflow
    # dst fleet slot for planet i (or MAXF dummy when not launching)
    rank_clip = jnp.clip(launch_rank, 0, MAXF - 1)
    dst = jnp.where(do, free_order[rank_clip], MAXF)

    def scatter(base, vals):
        padded = jnp.concatenate([base, base[:1]])      # len MAXF+1, dummy at end
        return padded.at[dst].set(vals)[:MAXF]

    f_owner = scatter(js["f_owner"], jnp.where(do, p_owner, -1))
    f_x = scatter(js["f_x"], sx)
    f_y = scatter(js["f_y"], sy)
    f_angle = scatter(js["f_angle"], angle)
    f_from = scatter(js["f_from"], js["p_id"])
    f_ships = scatter(js["f_ships"], ships_send)
    f_valid2 = scatter(f_valid, jnp.ones_like(p_valid))             # set True where launched
    # fleet ids (sequential; cosmetic)
    new_id = js["next_fleet_id"] + launch_rank
    f_id = scatter(js["f_id"], jnp.where(do, new_id, 0))
    next_fid = js["next_fleet_id"] + jnp.sum(do.astype(jnp.int32))

    return {**js, "p_ships": p_ships2, "next_fleet_id": next_fid,
            "f_id": f_id, "f_owner": f_owner, "f_x": f_x, "f_y": f_y,
            "f_angle": f_angle, "f_from": f_from, "f_ships": f_ships,
            "f_valid": f_valid2}


def movement(js):
    """Phase 3 (fast_sim 140-163): advance fleets, collide vs bounds/sun/planet.

    Returns updated js plus `combat_planet` [MAXF] = planet slot a fleet collided
    with via movement (or -1). Removed fleets have f_valid set False, but their
    combat membership persists in combat_planet for combat resolution.
    """
    fv = js["f_valid"]
    ships = jnp.maximum(js["f_ships"], 1.0)
    speed = 1.0 + (js["ship_speed"] - 1.0) * (jnp.log(ships) / jnp.log(1000.0)) ** 1.5
    speed = jnp.minimum(speed, js["ship_speed"])
    ox, oy = js["f_x"], js["f_y"]
    nx = ox + jnp.cos(js["f_angle"]) * speed
    ny = oy + jnp.sin(js["f_angle"]) * speed

    oob = ~((nx >= 0) & (nx <= BOARD_SIZE) & (ny >= 0) & (ny <= BOARD_SIZE))
    sun = _seg_dist(CENTER, CENTER, ox, oy, nx, ny) < SUN_RADIUS

    # fleet x planet collision matrix [F, P]
    d = _seg_dist(js["p_x"][None, :], js["p_y"][None, :],
                  ox[:, None], oy[:, None], nx[:, None], ny[:, None])
    hit = (d < js["p_rad"][None, :]) & js["p_valid"][None, :]
    phit_idx = _first_true_idx(hit)            # [F], -1 if none
    phit = phit_idx >= 0

    # fast_sim order: oob -> sun -> planet (first match removes & assigns combat)
    do_oob = fv & oob
    do_sun = fv & ~oob & sun
    do_planet = fv & ~oob & ~sun & phit
    removed = do_oob | do_sun | do_planet
    combat_planet = jnp.where(do_planet, phit_idx, -1)

    f_valid2 = fv & ~removed
    return ({**js, "f_x": nx, "f_y": ny, "f_valid": f_valid2}, combat_planet)


def rotation_sweep(js, combat_planet):
    """Phase 4 (fast_sim 165-194): rotate planets; sweep fleets along the arc.

    No comets here (deferred): every planet rotates if r + rad < 50. Sweep adds
    fleets (still valid after movement) to combat with the first planet whose
    swept segment passes within its radius.
    """
    dx = js["p_ix"] - CENTER
    dy = js["p_iy"] - CENTER
    r = jnp.sqrt(dx * dx + dy * dy)
    rotate = (r + js["p_rad"]) < ROTATION_RADIUS_LIMIT
    ox, oy = js["p_x"], js["p_y"]
    ia = jnp.arctan2(dy, dx)
    ca = ia + js["av"] * js["step"].astype(jnp.float32)
    nx = jnp.where(rotate, CENTER + r * jnp.cos(ca), ox)
    ny = jnp.where(rotate, CENTER + r * jnp.sin(ca), oy)

    moved = (ox != nx) | (oy != ny)
    fv = js["f_valid"]                          # fleets surviving movement
    # planet (swept seg ox,oy->nx,ny) x fleet (at new pos f_x,f_y) matrix [P, F]
    d = _seg_dist(js["f_x"][None, :], js["f_y"][None, :],
                  ox[:, None], oy[:, None], nx[:, None], ny[:, None])
    hit = (d < js["p_rad"][:, None]) & js["p_valid"][:, None] & moved[:, None] & fv[None, :]
    # each fleet -> first planet (slot order) that sweeps it
    hitT = hit.T                                # [F, P]
    swept_planet = _first_true_idx(hitT)        # [F]
    swept = swept_planet >= 0

    combat_planet2 = jnp.where(swept, swept_planet, combat_planet)
    f_valid2 = fv & ~swept
    return ({**js, "p_x": nx, "p_y": ny, "f_valid": f_valid2}, combat_planet2)


def combat(js, combat_planet, num_agents):
    """Phase 5 (fast_sim 233-262): per-planet, top-2 players by incoming ships."""
    cp = combat_planet
    valid_c = cp >= 0
    cp_safe = jnp.where(valid_c, cp, 0)
    # ships_by_player[P, a]
    sbp = jnp.zeros((MAXP, num_agents), jnp.float32)
    for a in range(num_agents):
        contrib = jnp.where(valid_c & (js["f_owner"] == a), js["f_ships"], 0.0)
        sbp = sbp.at[cp_safe, a].add(contrib)

    participants = jnp.sum(sbp > 0, axis=1)              # [P]
    top1 = jnp.max(sbp, axis=1)
    top1_idx = jnp.argmax(sbp, axis=1)
    masked = sbp.at[jnp.arange(MAXP), top1_idx].set(-1.0)
    second = jnp.max(masked, axis=1)
    second = jnp.where(participants >= 2, second, 0.0)

    tie = (participants >= 2) & (top1 == second)
    surv = jnp.where(participants >= 2, top1 - second, top1)
    surv = jnp.where(tie, 0.0, surv)
    surv_owner = jnp.where(surv > 0, top1_idx, -1)

    active = js["p_valid"] & (participants > 0) & (surv > 0)
    same = active & (js["p_owner"] == surv_owner)
    diff = active & (js["p_owner"] != surv_owner)

    p_ships = js["p_ships"]
    p_ships = jnp.where(same, p_ships + surv, p_ships)
    after = p_ships - surv                                # tentative for diff
    captured = diff & (after < 0)
    p_ships = jnp.where(diff, jnp.where(captured, jnp.abs(after), after), p_ships)
    p_owner = jnp.where(captured, surv_owner, js["p_owner"])
    return {**js, "p_ships": p_ships, "p_owner": p_owner}


def terminate(js, num_agents):
    """Phase 6 (fast_sim 264-291): step++, done flag, scores, rewards."""
    step2 = js["step"] + 1
    term = step2 >= (js["episode_steps"] - 2)
    scores = jnp.zeros((num_agents,), jnp.float32)
    alive = jnp.zeros((num_agents,), jnp.bool_)
    for a in range(num_agents):
        po = js["p_valid"] & (js["p_owner"] == a)
        fo = js["f_valid"] & (js["f_owner"] == a)
        scores = scores.at[a].set(
            jnp.sum(jnp.where(po, js["p_ships"], 0.0))
            + jnp.sum(jnp.where(fo, js["f_ships"], 0.0)))
        alive = alive.at[a].set(jnp.any(po) | jnp.any(fo))
    n_alive = jnp.sum(alive.astype(jnp.int32))
    term = term | (n_alive <= 1)
    mx = jnp.max(scores)
    rewards = jnp.where((scores == mx) & (mx > 0), 1.0, -1.0)
    return {**js, "step": step2}, term, scores, rewards


def step(js, action, num_agents):
    """One full turn (comets deferred). Order matches fast_sim exactly.

    action: dict of [MAXP] arrays {launch(bool), target(int slot), frac(float)}.
    Returns (js, done, scores, rewards).
    """
    js = launch(js, action, num_agents)
    js = production(js)
    js, cp = movement(js)
    js, cp = rotation_sweep(js, cp)
    js = combat(js, cp, num_agents)
    return terminate(js, num_agents)
