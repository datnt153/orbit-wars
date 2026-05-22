"""S10: PPO self-play training loop in JAX (jitted end-to-end).

Anchor curriculum (the validated PyTorch recipe core): seat 0 = learner,
seat>=1 = FROZEN anchor; only seat 0 is trained. Each train_update does
rollout (lax.scan) + GAE + K epochs of minibatch PPO, all on device — this is
what removes bottleneck F2. Reward = potential-based material shaping + terminal.

Logs the 1st-place's health metrics: explained_variance, approx_kl, clip_frac,
plus awr (learner win-rate vs anchor). Parity of the moving parts (env / encode /
forward / logp) is verified in test_jax_*.py.

Run: .venv/bin/python src/jax_train.py [resume.npz]
Env: N_ENVS N_STEPS UPDATES MB PPO_EPOCHS LR ENT SHAPE_SCALE
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

import fast_sim
import jax_env
import jax_encode
import jax_policy

ROOT = Path(__file__).resolve().parent.parent
F, G, P = jax_encode.F, jax_encode.G, jax_env.MAXP
GAMMA, LAM, CLIP, VAL_COEF, MAX_GRAD = 0.999, 0.95, 0.2, 0.5, 0.5


def load_templates(n_envs, num_agents):
    pat = ROOT / "data" / "bovard" / "maps128"
    eps = sorted(glob.glob(str(pat / "**" / "*.json"), recursive=True))
    if not eps:
        eps = sorted(glob.glob(str(ROOT / "data" / "bovard" / "**" / "*.json"),
                               recursive=True))
    states = []
    for p in eps:
        obs0 = json.loads(open(p).read())["steps"][0][0]["observation"]
        s = fast_sim.new_state(obs0, ship_speed=6.0, episode_steps=500)
        s["comets"] = []
        s["comet_planet_ids"] = []
        states.append(jax_env.to_jax(s))
    js_list = [states[i % len(states)] for i in range(n_envs)]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *js_list)


def potential(js, A):
    """Φ_a = material_a − mean_{b≠a} material_b ; material = Σ(ships+prod·8)."""
    mats = [jnp.sum(jnp.where(js["p_valid"] & (js["p_owner"] == a),
                              js["p_ships"] + js["p_prod"] * 8.0, 0.0), axis=-1)
            for a in range(A)]
    mat = jnp.stack(mats, axis=-1)                      # [N,A]
    total = jnp.sum(mat, axis=-1, keepdims=True)
    avg_opp = (total - mat) / max(1, A - 1)
    return mat - avg_opp


def make_train_update(num_agents, n_envs, n_steps, epochs, mb, shape_scale):
    A, N, T = num_agents, n_envs, n_steps
    B = N * A
    M = T * N * A
    n_mb = M // mb
    vstep = jax.vmap(lambda js, a: jax_env.step(js, a, A), in_axes=(0, 0))
    seat_learner = (jnp.arange(A) == 0)                 # [A] only seat0 trained

    def fwd(params, pf, pmask, gf):
        return jax_policy.forward(params, pf.reshape(B, P, F),
                                  pmask.reshape(B, P), gf.reshape(B, G))

    def gae(rew, val, done):
        def step(carry, x):
            next_v, last = carry
            r, v, d = x
            m = 1.0 - d
            delta = r + GAMMA * next_v * m - v
            last = delta + GAMMA * LAM * m * last
            return (v, last), last
        z = jnp.zeros((N, A))
        _, adv = jax.lax.scan(step, (z, z), (rew, val, done), reverse=True)
        return adv, adv + val

    def loss_fn(params, pf, pmask, omask, gf, launch, target, lp_old,
                adv, ret, tmask, ent_coef_t):
        g, t, v = jax_policy.forward(params, pf, pmask, gf)
        lp, ent = jax_policy.log_prob_and_entropy(g, t, omask, pmask,
                                                  launch, target)
        ratio = jnp.exp(jnp.clip(lp - lp_old, -20, 20))
        surr1 = ratio * adv
        surr2 = jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv
        denom = jnp.maximum(jnp.sum(tmask), 1.0)
        pl = -jnp.sum(jnp.minimum(surr1, surr2) * tmask) / denom
        vl = jnp.sum((v - ret) ** 2 * tmask) / denom
        em = jnp.sum(ent * tmask) / denom
        loss = pl + VAL_COEF * vl - ent_coef_t * em
        kl = jnp.sum((lp_old - lp) * tmask) / denom
        cf = jnp.sum((jnp.abs(ratio - 1.0) > CLIP).astype(jnp.float32) * tmask) / denom
        return loss, (pl, vl, em, kl, cf)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def train_update(params, opt_state, anchor, js, phi, key, ent_coef_t):
        body = make_body(params, anchor, js)
        (js, phi, key), traj = jax.lax.scan(body, (js, phi, key), None, length=T)
        pf, pmask, omask, gf, launch, target, lp, val, rew, done, nd, nw = traj
        tmask = jnp.broadcast_to(seat_learner[None, None, :], (T, N, A))
        tmask = tmask.astype(jnp.float32)
        adv, ret = gae(rew, val, done)
        # masked stats (no boolean indexing under jit) — weight by tmask
        mf = tmask.reshape(-1)
        dm = jnp.sum(mf) + 1e-8
        rf, vf = ret.reshape(-1), val.reshape(-1)
        rmean = jnp.sum(rf * mf) / dm
        rvar = jnp.sum(((rf - rmean) ** 2) * mf) / dm
        diff = rf - vf
        dmean = jnp.sum(diff * mf) / dm
        dvar = jnp.sum(((diff - dmean) ** 2) * mf) / dm
        ev = 1.0 - dvar / (rvar + 1e-8)
        # flatten
        def fl(x, *r):
            return x.reshape(M, *r)
        pf, pmask, omask, gf = fl(pf, P, F), fl(pmask, P), fl(omask, P), fl(gf, G)
        launch, target = fl(launch, P), fl(target, P)
        lp, ret, tmask = fl(lp), fl(ret), fl(tmask)
        adv = fl(adv)
        amean = jnp.sum(adv * mf) / dm
        avar = jnp.sum(((adv - amean) ** 2) * mf) / dm
        adv = (adv - amean) / (jnp.sqrt(avar) + 1e-6)

        def epoch(carry, ek):
            params, opt_state = carry
            perm = jax.random.permutation(ek, M)[:n_mb * mb].reshape(n_mb, mb)

            def mb_step(c, ib):
                params, opt_state = c
                (loss, aux), grads = grad_fn(
                    params, pf[ib], pmask[ib], omask[ib], gf[ib], launch[ib],
                    target[ib], lp[ib], adv[ib], ret[ib], tmask[ib], ent_coef_t)
                upd, opt_state = opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, upd)
                return (params, opt_state), aux

            (params, opt_state), auxs = jax.lax.scan(mb_step, (params, opt_state), perm)
            return (params, opt_state), auxs

        key, *eks = jax.random.split(key, epochs + 1)
        (params, opt_state), auxs = jax.lax.scan(
            epoch, (params, opt_state), jnp.stack(eks))
        pl, vl, em, kl, cf = [a[-1, -1] for a in auxs]   # last mb of last epoch
        metrics = (pl, vl, em, kl, cf, ev, jnp.sum(nd), jnp.sum(nw))
        return params, opt_state, js, phi, key, metrics

    # build closures needing js0 (initial templates) at call time
    holder = {}

    def make_body(params, anchor, js):
        js0 = holder["js0"]
        def body(carry, _):
            js, phi, key = carry
            pf, pmask, omask, gf = jax_encode.encode_batch(js, A)
            g, t, v = fwd(params, pf, pmask, gf)
            ga, ta, _ = fwd(anchor, pf, pmask, gf)
            g, t, v = g.reshape(N, A, P), t.reshape(N, A, P, P), v.reshape(N, A)
            ga, ta = ga.reshape(N, A, P), ta.reshape(N, A, P, P)
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            om = omask > 0.5
            Ll = jax.random.bernoulli(k1, jax.nn.sigmoid(g)) & om
            Lt = jax.random.categorical(k2, t, axis=-1)
            Al = jax.random.bernoulli(k3, jax.nn.sigmoid(ga)) & om
            At = jax.random.categorical(k4, ta, axis=-1)
            sl = seat_learner[None, :, None]
            launch = jnp.where(sl, Ll, Al)
            target = jnp.where(sl, Lt, At)
            lp, _e = jax_policy.log_prob_and_entropy(
                g.reshape(B, P), t.reshape(B, P, P), omask.reshape(B, P),
                pmask.reshape(B, P), launch.reshape(B, P).astype(jnp.float32),
                target.reshape(B, P))
            lp = lp.reshape(N, A)
            owner = jnp.clip(js["p_owner"], 0, A - 1)[:, None, :]
            m_launch = jnp.take_along_axis(launch, owner, axis=1)[:, 0, :]
            m_target = jnp.take_along_axis(target, owner, axis=1)[:, 0, :]
            owned = (js["p_owner"] >= 0) & js["p_valid"]
            act = {"launch": m_launch & owned, "target": m_target,
                   "frac": jnp.ones((N, P), jnp.float32)}
            js2, done, scores, term = vstep(js, act)
            phi2 = potential(js2, A)
            rew = (phi2 - phi) * shape_scale + jnp.where(done[:, None], term, 0.0)
            done_f = jnp.broadcast_to(done[:, None], (N, A)).astype(jnp.float32)
            nd = jnp.sum(done)
            nw = jnp.sum(done & (term[:, 0] > 0))
            js2 = jax.tree_util.tree_map(
                lambda c, i: jnp.where(
                    done.reshape((N,) + (1,) * (c.ndim - 1)), i, c), js2, js0)
            phi_next = potential(js2, A)
            out = (pf, pmask, omask, gf, launch.astype(jnp.float32), target,
                   lp, v, rew, done_f, nd, nw)
            return (js2, phi_next, key), out
        return body

    opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                      optax.adam(float(os.environ.get("LR", "1e-4"))))
    jitted = jax.jit(train_update)
    return jitted, opt, holder


def main():
    A = int(os.environ.get("NUM_AGENTS", "2"))
    N = int(os.environ.get("N_ENVS", "512"))
    T = int(os.environ.get("N_STEPS", "64"))
    epochs = int(os.environ.get("PPO_EPOCHS", "4"))
    mb = int(os.environ.get("MB", "1024"))
    updates = int(os.environ.get("UPDATES", "20"))
    ent = float(os.environ.get("ENT", "0.005"))
    shape_scale = float(os.environ.get("SHAPE_SCALE", "0.01"))
    resume = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "ppo_best.npz")

    print(f"device={jax.devices()[0].platform} A={A} N={N} T={T} mb={mb} "
          f"epochs={epochs} updates={updates} batch={N*A*T}")
    params = jax_policy.load_params(resume)
    anchor = jax.tree_util.tree_map(lambda x: x, params)   # frozen copy
    print(f"resume={Path(resume).name}")

    train_update, opt, holder = make_train_update(A, N, T, epochs, mb, shape_scale)
    opt_state = opt.init(params)
    js0 = load_templates(N, A)
    holder["js0"] = js0
    js = jax.tree_util.tree_map(lambda x: x, js0)
    phi = potential(js, A)
    key = jax.random.PRNGKey(0)

    steps_per = N * T
    t_start = time.time()
    for u in range(updates):
        ent_t = ent * max(0.0, 1.0 - u / max(1, updates))
        t0 = time.time()
        params, opt_state, js, phi, key, metr = train_update(
            params, opt_state, anchor, js, phi, key, jnp.float32(ent_t))
        jax.block_until_ready(params)
        dt = time.time() - t0
        pl, vl, em, kl, cf, ev, nd, nw = [float(x) for x in metr]
        awr = nw / nd if nd > 0 else 0.0
        sps = steps_per / dt
        if u == 0:
            print(f"[u0 compile+run {dt:.1f}s]")
        print(f"u{u:03d} {sps:8.0f} sps | pl {pl:+.3f} vl {vl:6.2f} "
              f"ent {em:.3f} kl {kl:+.4f} cf {cf:.3f} ev {ev:+.2f} "
              f"awr {awr:.2f} ({int(nd)}g)")

    tot = time.time() - t_start
    total_steps = updates * steps_per
    print(f"\nTOTAL {total_steps:,} env-steps in {tot:.1f}s -> "
          f"{total_steps/tot:,.0f} sps (incl compile)")
    # steady SPS excluding first (compile) update
    print(f"finite params: {bool(jnp.isfinite(jax.flatten_util.ravel_pytree(params)[0]).all())}")


if __name__ == "__main__":
    main()
