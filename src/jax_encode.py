"""S9: encode_state in JAX — env state pytree -> policy features.

Parity port of src/policy_encode.encode_state (the one true feature path, used
both offline and at inference). Single-env, single-perspective; vmap over
players then envs in the rollout. Comet flag = 0 (comets deferred).
"""
import jax
import jax.numpy as jnp

F = 15
G = 8
CENTER = 50.0
ROT_LIMIT = 50.0
_COMET = jnp.array([50.0, 150.0, 250.0, 350.0, 450.0], jnp.float32)


def _adiff(a, b):
    d = jnp.mod(a - b, 2 * jnp.pi)
    return jnp.minimum(d, 2 * jnp.pi - d)


def encode_one(js, player):
    """js: single-env pytree. player: scalar int. -> pf,pmask,omask,gf."""
    pv = js["p_valid"]
    own = js["p_owner"]
    px, py = js["p_x"], js["p_y"]
    pr, psh, ppr = js["p_rad"], js["p_ships"], js["p_prod"]

    # incoming fleet pressure: each fleet -> nearest planet by heading angle
    fv = js["f_valid"]
    fx, fy, fa, fsh, fown = (js["f_x"], js["f_y"], js["f_angle"],
                             js["f_ships"], js["f_owner"])
    ang = jnp.arctan2(py[None, :] - fy[:, None], px[None, :] - fx[:, None])  # [Ff,P]
    e = _adiff(fa[:, None], ang)
    e = jnp.where(pv[None, :], e, 1e9)              # ignore padded planets
    best_j = jnp.argmin(e, axis=1)                  # [Ff]
    best_e = jnp.min(e, axis=1)
    assign = fv & (best_e < 0.5)
    mine_c = jnp.where(assign & (fown == player), fsh, 0.0)
    en_c = jnp.where(assign & (fown >= 0) & (fown != player), fsh, 0.0)
    P = pv.shape[0]
    press_mine = jnp.zeros(P, jnp.float32).at[best_j].add(mine_c)
    press_enemy = jnp.zeros(P, jnp.float32).at[best_j].add(en_c)

    is_mine = (own == player) & pv
    is_enemy = (own >= 0) & (own != player) & pv
    is_neu = (own == -1) & pv
    dsun = jnp.hypot(px - CENTER, py - CENTER)
    is_orb = (dsun + pr) < ROT_LIMIT

    feats = jnp.stack([
        is_mine.astype(jnp.float32),
        is_enemy.astype(jnp.float32),
        is_neu.astype(jnp.float32),
        jnp.log1p(psh) / 8.0,
        ppr / 5.0,
        pr / 4.0,
        px / 100.0,
        py / 100.0,
        dsun / 70.0,
        jnp.zeros(P, jnp.float32),                 # comet flag (deferred)
        is_orb.astype(jnp.float32),
        jnp.log1p(press_mine) / 8.0,
        jnp.log1p(press_enemy) / 8.0,
        is_mine.astype(jnp.float32),
        jnp.ones(P, jnp.float32),
    ], axis=1)                                      # [P, F]
    pmask = pv.astype(jnp.float32)
    pf = feats * pmask[:, None]                     # padded rows -> 0
    omask = is_mine.astype(jnp.float32)

    # globals
    np4 = jnp.any((own >= 2) & pv)
    my_tot = jnp.sum(jnp.where(is_mine, psh + ppr * 8.0, 0.0))
    opp_tot = jnp.sum(jnp.where(is_enemy, psh + ppr * 8.0, 0.0))
    neu_tot = jnp.sum(jnp.where(is_neu, psh, 0.0))
    my_np = jnp.sum(is_mine.astype(jnp.float32))
    n = jnp.sum(pmask)
    step = js["step"].astype(jnp.float32)
    nxt = jnp.min(jnp.where(_COMET > step, _COMET, 500.0))
    gf = jnp.stack([
        step / 500.0,
        np4.astype(jnp.float32),
        my_tot / 200.0, opp_tot / 200.0, neu_tot / 200.0,
        my_np / 20.0,
        (nxt - step) / 100.0,
        my_np / jnp.maximum(1.0, n),
    ])
    return pf, pmask, omask, gf


def encode_batch(js_b, num_agents):
    """js_b: batched pytree [N,...] -> pf[N,A,P,F], pmask[N,A,P], omask, gf[N,A,G]."""
    players = jnp.arange(num_agents)

    def per_env(js1):
        return jax.vmap(lambda pl: encode_one(js1, pl))(players)

    return jax.vmap(per_env)(js_b)
