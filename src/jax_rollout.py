"""S8: vmap(batch) + lax.scan(time) rollout — the whole env loop as ONE XLA
graph on device. This is what kills bottleneck F2 (no per-step host round-trip).

Random-action policy for now (env throughput isolation). A JAX policy (S9)
plugs in by replacing `random_actions` with a network forward+sample.

Usage:
  .venv/bin/python src/jax_rollout.py [N_ENVS] [N_STEPS]
"""
import functools
import glob
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import fast_sim
import jax_env
import jax_encode
import jax_policy

ROOT = Path(__file__).resolve().parent.parent


def load_batch(n_envs, num_agents=4):
    """Build a batched env pytree from bovard initial states (tiled to n_envs)."""
    eps = sorted(glob.glob(str(ROOT / "data" / "bovard" / "**" / "*.json"),
                           recursive=True))
    states = []
    for p in eps[:max(1, min(len(eps), n_envs))]:
        obs0 = json.loads(open(p).read())["steps"][0][0]["observation"]
        s = fast_sim.new_state(obs0, ship_speed=6.0, episode_steps=500)
        s["comets"] = []
        s["comet_planet_ids"] = []
        states.append(jax_env.to_jax(s))
    js_list = [states[i % len(states)] for i in range(n_envs)]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *js_list)


def random_actions(js_batch, key, num_agents):
    N = js_batch["p_valid"].shape[0]
    P = jax_env.MAXP
    k1, k2, k3 = jax.random.split(key, 3)
    owned = js_batch["p_valid"] & (js_batch["p_owner"] >= 0) \
        & (js_batch["p_owner"] < num_agents)
    launch = jax.random.bernoulli(k1, 0.4, (N, P)) & owned
    target = jax.random.randint(k2, (N, P), 0, P)
    frac = jax.random.uniform(k3, (N, P), minval=0.3, maxval=1.0)
    return {"launch": launch, "target": target, "frac": frac}


@functools.partial(jax.jit, static_argnums=(2, 3))
def rollout(js_batch, key, n_steps, num_agents):
    vstep = jax.vmap(lambda js, a: jax_env.step(js, a, num_agents),
                     in_axes=(0, 0))

    def body(carry, _):
        js, key = carry
        key, ka = jax.random.split(key)
        act = random_actions(js, ka, num_agents)
        js, done, scores, rew = vstep(js, act)
        return (js, key), rew.sum()

    (js, key), rews = jax.lax.scan(body, (js_batch, key), None, length=n_steps)
    return js, rews


def policy_actions(params, js_batch, key, num_agents):
    """Self-play: encode per seat -> forward -> sample -> merge per-planet owner.

    Each planet's action comes from its OWNER's policy seat. frac=1.0 (PPO sends
    all ships, matching sf=ones)."""
    N = js_batch["p_valid"].shape[0]
    P, A = jax_env.MAXP, num_agents
    pf, pmask, omask, gf = jax_encode.encode_batch(js_batch, A)   # [N,A,P,*]
    B = N * A
    gate, tgt, _ = jax_policy.forward(
        params, pf.reshape(B, P, jax_encode.F), pmask.reshape(B, P),
        gf.reshape(B, jax_encode.G))
    gate = gate.reshape(N, A, P)
    tgt = tgt.reshape(N, A, P, P)

    kg, kt = jax.random.split(key)
    launch_seat = jax.random.bernoulli(kg, jax.nn.sigmoid(gate)) & (omask > 0.5)
    tgt_idx = jax.random.categorical(kt, tgt, axis=-1)            # [N,A,P]

    owner = jnp.clip(js_batch["p_owner"], 0, A - 1)               # [N,P]
    oidx = owner[:, None, :]                                      # [N,1,P]
    m_launch = jnp.take_along_axis(launch_seat, oidx, axis=1)[:, 0, :]
    m_target = jnp.take_along_axis(tgt_idx, oidx, axis=1)[:, 0, :]
    owned = (js_batch["p_owner"] >= 0) & js_batch["p_valid"]
    return {"launch": m_launch & owned, "target": m_target,
            "frac": jnp.ones((N, P), jnp.float32)}


@functools.partial(jax.jit, static_argnums=(3, 4))
def selfplay_rollout(params, js_batch, key, n_steps, num_agents):
    vstep = jax.vmap(lambda js, a: jax_env.step(js, a, num_agents),
                     in_axes=(0, 0))

    def body(carry, _):
        js, key = carry
        key, ka = jax.random.split(key)
        act = policy_actions(params, js, ka, num_agents)
        js, done, scores, rew = vstep(js, act)
        return (js, key), rew.sum()

    (js, key), rews = jax.lax.scan(body, (js_batch, key), None, length=n_steps)
    return js, rews


def _bench(fn, *args, reps=3):
    t0 = time.time()
    out = fn(*args)
    jax.block_until_ready(out)
    t_compile = time.time() - t0
    t0 = time.time()
    for _ in range(reps):
        out = fn(*args)
        jax.block_until_ready(out)
    return t_compile, (time.time() - t0) / reps, out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    selfplay = "--selfplay" in sys.argv
    n_envs = int(args[0]) if len(args) > 0 else 512
    n_steps = int(args[1]) if len(args) > 1 else 200
    num_agents = 4
    print(f"device={jax.devices()[0].platform}  N_ENVS={n_envs}  "
          f"N_STEPS={n_steps}  mode={'SELFPLAY(policy)' if selfplay else 'random'}")
    js = load_batch(n_envs, num_agents)
    key = jax.random.PRNGKey(0)

    if selfplay:
        ckpt = next((a for a in args if a.endswith(".npz")),
                    str(ROOT / "data" / "ppo_best.npz"))
        params = jax_policy.load_params(ckpt)
        print(f"policy={Path(ckpt).name}")
        tc, dt, (js2, rews) = _bench(selfplay_rollout, params, js, key,
                                     n_steps, num_agents)
    else:
        tc, dt, (js2, rews) = _bench(rollout, js, key, n_steps, num_agents)

    sps = n_envs * n_steps / dt
    print(f"compile+first: {tc:.2f}s")
    print(f"steady: {dt * 1000:.1f} ms/rollout  ->  {sps:,.0f} env-steps/s "
          f"({n_envs}x{n_steps})")
    print(f"rews finite={bool(jnp.isfinite(rews).all())}")


if __name__ == "__main__":
    main()
