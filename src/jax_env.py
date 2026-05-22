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


# ---------------------------------------------------------------- phases

def production(js):
    """Phase 2 (fast_sim 135-138): owned planets gain `prod` ships."""
    owned = js["p_valid"] & (js["p_owner"] >= 0)
    p_ships = jnp.where(owned, js["p_ships"] + js["p_prod"], js["p_ships"])
    return {**js, "p_ships": p_ships}


def step(js, joint_actions, num_agents):
    """Full turn — BUILT INCREMENTALLY (see jax_port.md). Not yet complete."""
    raise NotImplementedError("step() phases S1-S6 pending; see jax_port.md")
