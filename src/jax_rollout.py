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


def main():
    n_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    num_agents = 4
    print(f"device={jax.devices()[0].platform}  N_ENVS={n_envs}  "
          f"N_STEPS={n_steps}")
    js = load_batch(n_envs, num_agents)
    key = jax.random.PRNGKey(0)

    t0 = time.time()
    js2, rews = rollout(js, key, n_steps, num_agents)
    jax.block_until_ready((js2, rews))
    t_compile = time.time() - t0
    print(f"compile+first run: {t_compile:.2f}s")

    # timed runs (cached)
    reps = 3
    t0 = time.time()
    for _ in range(reps):
        js2, rews = rollout(js, key, n_steps, num_agents)
        jax.block_until_ready((js2, rews))
    dt = (time.time() - t0) / reps
    sps = n_envs * n_steps / dt
    print(f"steady: {dt * 1000:.1f} ms/rollout  ->  {sps:,.0f} env-steps/s "
          f"({n_envs}x{n_steps})")
    print(f"rews finite={bool(jnp.isfinite(rews).all())}")


if __name__ == "__main__":
    main()
