"""CNN policy PPO trainer (2P self-play vs opponent pool). Tests whether the
grid-CNN beats the transformer's ~10-17%-vs-v4 plateau.

Memory: rasterized grid (50x50x11) is big, so we store ONLY seat-0's grid (the
trained seat) per step -> ~1.8GB at T=32,N=512. seat-1 = opponent sampled from a
pool (PFSP). Reward = territory(prod) shaping + terminal. No teacher-KL (it kept
the transformer diffuse). Sun handled natively by the CNN (sun channel).

Run: .venv/bin/python src/jax_train_cnn.py    (env: N_ENVS N_STEPS MB UPDATES ...)
"""
import functools
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

try:
    import wandb
except ImportError:
    wandb = None

import fast_sim
import jax_env
import jax_raster
import jax_cnn_policy as cnn
import jax_policy                       # log_prob_and_entropy (gate/tgt interface)

ROOT = Path(__file__).resolve().parent.parent
H, Wd, Cc, Gg = jax_raster.H, jax_raster.W, jax_raster.C, jax_raster.G
P = jax_env.MAXP
GAMMA = float(os.environ.get("GAMMA", "0.9997"))
LAM, CLIP, VAL_COEF, MAX_GRAD = 0.95, 0.2, 0.5, 0.5
A = 2                                   # 2P self-play


def load_templates(n_envs):
    md = os.environ.get("MAPS_DIR", "data/maps_2p")
    eps = sorted(glob.glob(str(ROOT / md / "**" / "*.json"), recursive=True))
    states = []
    for p in eps:
        obs0 = json.loads(open(p).read())["steps"][0][0]["observation"]
        s = fast_sim.new_state(obs0, ship_speed=6.0, episode_steps=500)
        s["comets"] = []; s["comet_planet_ids"] = []
        states.append(jax_env.to_jax(s))
    js_list = [states[i % len(states)] for i in range(n_envs)]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *js_list)


def potential(js):
    terr = [jnp.sum(jnp.where(js["p_valid"] & (js["p_owner"] == a), js["p_prod"],
                              0.0), axis=-1) for a in range(A)]
    t = jnp.stack(terr, axis=-1)
    return t - (jnp.sum(t, -1, keepdims=True) - t) / max(1, A - 1)


def save_ckpt(path, params, steps, upd):
    out = {("cnn." + k): np.asarray(v) for k, v in params.items()}
    out.update(_STEPS=np.int64(steps), _UPD=np.int64(upd), _CNN=np.int64(1))
    np.savez(path, **out)


def make_update(N, T, epochs, mb, shape_scale, js0):
    B = N * A
    M = T * N
    n_mb = M // mb
    vstep = jax.vmap(lambda js, a: jax_env.step(js, a, A), in_axes=(0, 0))
    seatL = jnp.array([True, False])

    def actions(params, opp, js, key):
        grid, ix, pmask, omask, gf = jax_raster.raster_batch(js, A)   # [N,A,...]
        gb = grid.reshape(B, H, Wd, Cc); ib = ix.reshape(B, P)
        pmb = pmask.reshape(B, P); gfb = gf.reshape(B, Gg)
        g, t, v = cnn.forward(params, gb, ib, pmb, gfb)
        ga, ta, _ = cnn.forward(opp, gb, ib, pmb, gfb)
        g, t, v = g.reshape(N, A, P), t.reshape(N, A, P, P), v.reshape(N, A)
        ga, ta = ga.reshape(N, A, P), ta.reshape(N, A, P, P)
        om = omask                                              # [N,A,P]
        k1, k2, k3, k4 = jax.random.split(key, 4)
        Ll = jax.random.bernoulli(k1, jax.nn.sigmoid(g)) & (om > 0.5)
        Lt = jax.random.categorical(k2, t, axis=-1)
        Al = jax.random.bernoulli(k3, jax.nn.sigmoid(ga)) & (om > 0.5)
        At = jax.random.categorical(k4, ta, axis=-1)
        sl = seatL[None, :, None]
        launch = jnp.where(sl, Ll, Al); target = jnp.where(sl, Lt, At)
        lp, _e = jax_policy.log_prob_and_entropy(
            g.reshape(B, P), t.reshape(B, P, P), om.reshape(B, P),
            pmb, launch.reshape(B, P).astype(jnp.float32), target.reshape(B, P))
        return (launch, target, lp.reshape(N, A), v, grid[:, 0], ix[:, 0],
                pmask[:, 0], omask[:, 0], gf[:, 0])

    def merge_step(js, launch, target):
        owner = jnp.clip(js["p_owner"], 0, A - 1)[:, None, :]
        ml = jnp.take_along_axis(launch, owner, axis=1)[:, 0, :]
        mt = jnp.take_along_axis(target, owner, axis=1)[:, 0, :]
        owned = (js["p_owner"] >= 0) & js["p_valid"]
        return {"launch": ml & owned, "target": mt,
                "frac": jnp.ones((N, P), jnp.float32)}

    def loss_fn(params, grid, ix, pmask, omask, launch, target, lp_old,
                adv, ret, ent_c):
        g, t, v = cnn.forward(params, grid, ix, pmask, jnp.zeros((grid.shape[0], Gg)))
        lp, ent = jax_policy.log_prob_and_entropy(g, t, omask, pmask, launch, target)
        ratio = jnp.exp(jnp.clip(lp - lp_old, -20, 20))
        s1 = ratio * adv; s2 = jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv
        pl = -jnp.mean(jnp.minimum(s1, s2))
        vl = jnp.mean((v - ret) ** 2)
        em = jnp.mean(ent)
        loss = pl + VAL_COEF * vl - ent_c * em
        kl = jnp.mean(lp_old - lp)
        cf = jnp.mean((jnp.abs(ratio - 1.0) > CLIP).astype(jnp.float32))
        return loss, (pl, vl, em, kl, cf)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                      optax.adam(float(os.environ.get("LR", "1e-4"))))

    @jax.jit
    def update(params, opt_state, opp, js, phi, key, ent_c):
        # gf for loss is recomputed=0 (CNN uses grid; gf minor) — store grid only
        def body(carry, _):
            js, phi, key = carry
            key, ka = jax.random.split(key)
            (launch, target, lp, v, g0, ix0, pm0, om0, gf0) = actions(
                params, opp, js, ka)
            act = merge_step(js, launch, target)
            js2, done, scores, term = vstep(js, act)
            phi2 = potential(js2)
            rew = (phi2 - phi) * shape_scale + jnp.where(done[:, None], term, 0.0)
            done_f = jnp.broadcast_to(done[:, None], (N, A)).astype(jnp.float32)
            nd = jnp.sum(done); nw = jnp.sum(done & (term[:, 0] > 0))
            js2 = jax.tree_util.tree_map(
                lambda c, i: jnp.where(done.reshape((N,) + (1,) * (c.ndim - 1)),
                                       i, c), js2, js0)
            phin = potential(js2)
            # store seat-0 only
            out = (g0, ix0, pm0, om0, launch[:, 0], target[:, 0], lp[:, 0],
                   v[:, 0], rew[:, 0], done_f[:, 0], nd, nw)
            return (js2, phin, key), out
        (js, phi, key), tr = jax.lax.scan(body, (js, phi, key), None, length=T)
        grid, ix, pm, om, launch, target, lp, val, rew, done, nd, nw = tr
        # GAE (seat0) [T,N]
        def gstep(c, x):
            nv, last = c; r, vv, d = x; m = 1.0 - d
            delta = r + GAMMA * nv * m - vv
            last = delta + GAMMA * LAM * m * last
            return (vv, last), last
        z = jnp.zeros((N,))
        _, adv = jax.lax.scan(gstep, (z, z), (rew, val, done), reverse=True)
        ret = adv + val
        ev = 1.0 - jnp.var(ret - val) / (jnp.var(ret) + 1e-8)
        fl = lambda x, *r: x.reshape(M, *r)
        grid = fl(grid, H, Wd, Cc); ix = fl(ix, P); pm = fl(pm, P); om = fl(om, P)
        launch = fl(launch, P); target = fl(target, P)
        lp = fl(lp); adv = fl(adv); ret = fl(ret)
        adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-6)

        def epoch2(c, ek):
            params, opt_state = c
            perm = jax.random.permutation(ek, M)[:n_mb * mb].reshape(n_mb, mb)

            def mbs(cc, ii):
                pr, os_ = cc
                (l, aux), grads = grad_fn(pr, grid[ii], ix[ii], pm[ii], om[ii],
                                          launch[ii], target[ii], lp[ii], adv[ii],
                                          ret[ii], ent_c)
                upd, os2 = opt.update(grads, os_, pr)
                return (optax.apply_updates(pr, upd), os2), aux
            return jax.lax.scan(mbs, (params, opt_state), perm)

        key, *eks = jax.random.split(key, epochs + 1)
        (params, opt_state), auxs = jax.lax.scan(epoch2, (params, opt_state),
                                                 jnp.stack(eks))
        pl, vl, em, kl, cf = [a[-1, -1] for a in auxs]
        return (params, opt_state, js, phi, key,
                (pl, vl, em, kl, cf, ev, jnp.sum(nd), jnp.sum(nw)))

    return update, opt


def main():
    N = int(os.environ.get("N_ENVS", "512"))
    T = int(os.environ.get("N_STEPS", "32"))
    epochs = int(os.environ.get("PPO_EPOCHS", "2"))
    mb = int(os.environ.get("MB", "512"))
    updates = int(os.environ.get("UPDATES", "20"))
    ent = float(os.environ.get("ENT", "0.01"))
    anneal = int(os.environ.get("ANNEAL_UPD", str(max(1, updates // 2))))
    shape_scale = float(os.environ.get("SHAPE_SCALE", "0.05"))
    pool_every = int(os.environ.get("POOL_EVERY", "300"))
    pool_max = int(os.environ.get("POOL_MAX", "6"))
    save_every = int(os.environ.get("SAVE_EVERY", "300"))
    out_path = os.environ.get("OUT", str(ROOT / "data" / "jax_cnn.npz"))

    print(f"device={jax.devices()[0].platform} CNN A={A} N={N} T={T} mb={mb} "
          f"updates={updates} grid={H}x{Wd}x{Cc}", flush=True)
    key = jax.random.PRNGKey(int(os.environ.get("SEED", "0")))
    key, ik = jax.random.split(key)
    params = cnn.init_params(ik)
    pool = [jax.tree_util.tree_map(lambda x: x, params)]
    prng = np.random.default_rng(0)

    use_wandb = os.environ.get("WANDB", "0") == "1" and wandb is not None
    if use_wandb:
        wandb.init(project="orbit-wars", name=f"cnn_{time.strftime('%Y%m%d_%H%M')}",
                   config=dict(N=N, T=T, mb=mb, updates=updates, arch="CNN"))

    js0 = load_templates(N)
    update, opt = make_update(N, T, epochs, mb, shape_scale, js0)
    opt_state = opt.init(params)
    js = jax.tree_util.tree_map(lambda x: x, js0)
    phi = potential(js)
    steps_per = N * T
    t0 = time.time()
    for u in range(updates):
        ent_c = ent * max(0.0, 1.0 - u / max(1, anneal))
        opp = pool[int(prng.integers(len(pool)))]
        tt = time.time()
        params, opt_state, js, phi, key, m = update(
            params, opt_state, opp, js, phi, key, jnp.float32(ent_c))
        jax.block_until_ready(params)
        dt = time.time() - tt
        pl, vl, em, kl, cf, ev, nd, nw = [float(x) for x in m]
        awr = nw / nd if nd > 0 else 0.0
        steps = (u + 1) * steps_per
        if u == 0:
            print(f"[u0 compile {dt:.0f}s]", flush=True)
        print(f"u{u:04d} {steps_per/dt:6.0f}sps {steps/1e6:5.1f}M | pl {pl:+.3f} "
              f"vl {vl:6.2f} ent {em:.3f} kl {kl:+.4f} cf {cf:.3f} ev {ev:+.2f} "
              f"awr {awr:.2f} |pool|={len(pool)} ({int(nd)}g)", flush=True)
        if use_wandb:
            wandb.log({"sps": steps_per / dt, "policy_loss": pl, "value_loss": vl,
                       "entropy": em, "kl": kl, "clip_frac": cf,
                       "explained_variance": ev, "awr": awr}, step=steps)
        if pool_every and (u + 1) % pool_every == 0:
            pool.append(jax.tree_util.tree_map(lambda x: x, params))
            if len(pool) > pool_max:
                pool.pop(1)
        if save_every and ((u + 1) % save_every == 0 or u == updates - 1):
            save_ckpt(out_path, params, steps, u + 1)
            print(f"[saved {Path(out_path).name} @ {steps/1e6:.1f}M]", flush=True)
    if use_wandb:
        wandb.finish()
    print(f"DONE {updates*steps_per:,} steps in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
