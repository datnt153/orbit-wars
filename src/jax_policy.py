"""S9: ActorCritic entity-transformer forward in pure JAX.

Replicates src/ppo_train.py ActorCritic.forward bit-for-bit (incl. PyTorch
MultiheadAttention with key_padding_mask) so the SAME ppo_*.npz weights run
inside the JAX rollout. Verified vs PyTorch in test_jax_policy.py.

Weights: the npz keys are PyTorch state_dict names (Linear.weight is [out,in]).
"""
import numpy as np
import jax
import jax.numpy as jnp

D = 64
HEADS = 4
HEAD_DIM = D // HEADS
NEG = float(np.finfo(np.float32).min)


def load_params(path):
    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in z.files if not k.startswith("_")}


def _linear(x, p, name):
    return x @ p[name + ".weight"].T + p[name + ".bias"]


def _layernorm(x, p, name, eps=1e-5):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    xn = (x - mu) / jnp.sqrt(var + eps)
    return xn * p[name + ".weight"] + p[name + ".bias"]


def _mha(h, kpm, p, name):
    """PyTorch nn.MultiheadAttention(batch_first), self-attention h,h,h.
    h: [B, S, D]; kpm: [B, S] bool (True = padded key to mask)."""
    B, S, _ = h.shape
    qkv = h @ p[name + ".in_proj_weight"].T + p[name + ".in_proj_bias"]  # [B,S,3D]
    q, k, v = jnp.split(qkv, 3, axis=-1)

    def split_heads(t):
        return t.reshape(B, S, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # [B,H,S,Dh]

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    attn = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(HEAD_DIM)            # [B,H,S,S]
    mask = kpm[:, None, None, :]                                        # mask keys
    attn = jnp.where(mask, NEG, attn)
    attn = jax.nn.softmax(attn, axis=-1)
    out = attn @ v                                                      # [B,H,S,Dh]
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    return out @ p[name + ".out_proj.weight"].T + p[name + ".out_proj.bias"]


def forward(p, pf, pmask, gf):
    """pf [B,S,F], pmask [B,S] (1=real), gf [B,G] -> gate[B,S], tgt[B,S,S], v[B]."""
    x = _linear(pf, p, "inp") + _linear(gf, p, "gemb")[:, None, :]
    kpm = pmask < 0.5
    for i in (0, 1):
        b = f"blocks.{i}"
        h = _layernorm(x, p, b + ".n1")
        x = x + _mha(h, kpm, p, b + ".att")
        h2 = _layernorm(x, p, b + ".n2")
        ff = jax.nn.relu(_linear(h2, p, b + ".ff.0"))
        ff = _linear(ff, p, b + ".ff.2")
        x = x + ff

    gate = jax.nn.relu(_linear(x, p, "gate.0"))
    gate = _linear(gate, p, "gate.2")[..., 0]                          # [B,S]

    q = _linear(x, p, "tq")
    k = _linear(x, p, "tk")
    tgt = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(D)                     # [B,S,S]
    tgt = jnp.where(kpm[:, None, :], NEG, tgt)
    S = pf.shape[1]
    eye = jnp.eye(S, dtype=bool)
    tgt = jnp.where(eye[None], NEG, tgt)

    m = pmask[..., None]
    pooled = jnp.sum(x * m, axis=1) / jnp.clip(jnp.sum(m, axis=1), 1.0, None)
    val = jax.nn.relu(_linear(pooled, p, "v_head.0"))
    val = _linear(val, p, "v_head.2")[..., 0]                          # [B]
    return gate, tgt, val
