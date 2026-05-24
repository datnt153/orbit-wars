"""Rasterize Orbit Wars continuous state -> spatial grid for a CNN/UNet policy.

Lux top-1/2 used grid-CNNs (they learn geometry — distances, the sun, orbits —
automatically, vs our hand-feature transformer which plateaued ~10-17% vs v4 and
even flew fleets into the sun). Orbit Wars is continuous, so we rasterize.

raster_one(js, player) -> (grid[H,W,C], planet_ix[MAXP], pmask[MAXP],
omask[MAXP], gf[G]).  planet_ix = flat cell index per planet (for gather in the
action head). vmap over players then envs for a batch.
"""
import jax
import jax.numpy as jnp

import jax_env

MAXP = jax_env.MAXP
H = W = 50                      # cell = 100/50 = 2.0 board units
CELL = jax_env.BOARD_SIZE / H
CENTER = jax_env.CENTER
SUN_R = jax_env.SUN_RADIUS
ROT_LIMIT = jax_env.ROTATION_RADIUS_LIMIT
C = 11                         # channels (see below)
G = 6                          # global features

# static sun-mask channel: cells whose center is within the sun
_ys, _xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
_cx = (_xs + 0.5) * CELL
_cy = (_ys + 0.5) * CELL
_SUN_GRID = (jnp.hypot(_cx - CENTER, _cy - CENTER) < SUN_R).astype(jnp.float32)


def raster_one(js, player):
    pv = js["p_valid"]
    own = js["p_owner"]
    px, py = js["p_x"], js["p_y"]
    psh, ppr, prad = js["p_ships"], js["p_prod"], js["p_rad"]
    col = jnp.clip((px / CELL).astype(jnp.int32), 0, W - 1)
    row = jnp.clip((py / CELL).astype(jnp.int32), 0, H - 1)
    flat = row * W + col                                  # [MAXP] cell per planet

    is_mine = pv & (own == player)
    is_enemy = pv & (own >= 0) & (own != player)
    is_neu = pv & (own == -1)
    dsun = jnp.hypot(px - CENTER, py - CENTER)
    is_orb = (dsun + prad) < ROT_LIMIT

    grid = jnp.zeros((H * W, C), jnp.float32)

    def scat(g, ch, vals, mask):
        return g.at[flat, ch].add(jnp.where(mask, vals, 0.0))

    one = jnp.ones_like(px)
    grid = scat(grid, 0, one, is_mine)
    grid = scat(grid, 1, one, is_enemy)
    grid = scat(grid, 2, one, is_neu)
    grid = scat(grid, 3, jnp.log1p(psh) / 8.0, pv)
    grid = scat(grid, 4, ppr / 5.0, pv)
    grid = scat(grid, 5, prad / 4.0, pv)
    grid = scat(grid, 6, is_orb.astype(jnp.float32), pv)

    # fleets -> channels 7 (mine) / 8 (enemy) by ship mass
    fv = js["f_valid"]
    fcol = jnp.clip((js["f_x"] / CELL).astype(jnp.int32), 0, W - 1)
    frow = jnp.clip((js["f_y"] / CELL).astype(jnp.int32), 0, H - 1)
    fflat = frow * W + fcol
    f_mine = fv & (js["f_owner"] == player)
    f_enemy = fv & (js["f_owner"] >= 0) & (js["f_owner"] != player)
    fval = jnp.log1p(js["f_ships"]) / 8.0
    grid = grid.at[fflat, 7].add(jnp.where(f_mine, fval, 0.0))
    grid = grid.at[fflat, 8].add(jnp.where(f_enemy, fval, 0.0))

    grid = grid.reshape(H, W, C)
    grid = grid.at[:, :, 9].set(_SUN_GRID)                # static sun-mask
    grid = grid.at[:, :, 10].set(1.0)                     # bias plane

    pmask = pv.astype(jnp.float32)
    omask = is_mine.astype(jnp.float32)
    # globals
    step = js["step"].astype(jnp.float32)
    np4 = jnp.any((own >= 2) & pv).astype(jnp.float32)
    my_t = jnp.sum(jnp.where(is_mine, psh + ppr * 8.0, 0.0)) / 200.0
    op_t = jnp.sum(jnp.where(is_enemy, psh + ppr * 8.0, 0.0)) / 200.0
    my_np = jnp.sum(is_mine.astype(jnp.float32))
    n = jnp.sum(pmask)
    gf = jnp.stack([step / 500.0, np4, my_t, op_t, my_np / 20.0,
                    my_np / jnp.maximum(1.0, n)])
    return grid, flat, pmask, omask, gf


def raster_batch(js_b, num_agents):
    players = jnp.arange(num_agents)

    def per_env(js1):
        return jax.vmap(lambda pl: raster_one(js1, pl))(players)

    return jax.vmap(per_env)(js_b)     # grid[N,A,H,W,C], ix[N,A,MAXP], ...
