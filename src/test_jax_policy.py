"""Forward parity: JAX jax_policy.forward vs PyTorch ActorCritic, same weights.

Run: .venv/bin/python src/test_jax_policy.py [ckpt.npz]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import jax.numpy as jnp

import jax_policy
from ppo_train import ActorCritic
from policy_encode import MAXP, F, G

ROOT = Path(__file__).resolve().parent.parent


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "ppo_best59.npz")
    z = np.load(ckpt)

    # PyTorch net
    net = ActorCritic()
    net.load_state_dict({k: torch.tensor(z[k]) for k in z.files
                         if not k.startswith("_")})
    net.eval()
    # JAX params
    p = jax_policy.load_params(ckpt)

    rng = np.random.default_rng(0)
    B = 16
    pf = rng.standard_normal((B, MAXP, F)).astype(np.float32)
    pmask = np.zeros((B, MAXP), np.float32)
    for b in range(B):
        n = int(rng.integers(8, MAXP))      # 8..47 real planets
        pmask[b, :n] = 1.0
    gf = rng.standard_normal((B, G)).astype(np.float32)

    with torch.no_grad():
        g_t, t_t, v_t = net(torch.tensor(pf), torch.tensor(pmask), torch.tensor(gf))
    g_j, t_j, v_j = jax_policy.forward(p, jnp.asarray(pf), jnp.asarray(pmask),
                                       jnp.asarray(gf))
    g_t, t_t, v_t = g_t.numpy(), t_t.numpy(), v_t.numpy()
    g_j, t_j, v_j = np.asarray(g_j), np.asarray(t_j), np.asarray(v_j)

    # tgt: compare only unmasked logits (masked = -inf constant in both)
    finite = t_t > (jax_policy.NEG / 2)
    dg = np.abs(g_t - g_j).max()
    dv = np.abs(v_t - v_j).max()
    dt = np.abs(t_t[finite] - t_j[finite]).max() if finite.any() else 0.0

    ok = dg < 1e-3 and dv < 1e-3 and dt < 1e-3
    print(f"ckpt={Path(ckpt).name}")
    print(f"[gate]  maxdiff={dg:.2e}")
    print(f"[tgt]   maxdiff={dt:.2e}  (unmasked logits)")
    print(f"[value] maxdiff={dv:.2e}")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
