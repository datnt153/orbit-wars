"""CNN policy (Lux top-1/2 architecture, adapted to Orbit Wars).

Rasterized grid -> conv core (learns geometry: sun, distances, orbits) -> GATHER
per-planet embeddings at planet cells -> same gate / target-pointer / value heads
as the transformer (so it drops into jax_train). ~300K params (vs transformer 85K).

forward(params, grid[B,H,W,C], planet_ix[B,MAXP], pmask[B,MAXP], gf[B,G])
  -> gate[B,MAXP], tgt[B,MAXP,MAXP], value[B]   (same interface as jax_policy)
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import jax_raster

H, W, C, G = jax_raster.H, jax_raster.W, jax_raster.C, jax_raster.G
MAXP = jax_raster.MAXP
D = 64            # conv hidden / model dim
KBLOCKS = 4       # residual conv blocks (each 2x 3x3 conv)
NEG = float(np.finfo(np.float32).min)
_DN = ("NHWC", "HWIO", "NHWC")


def _conv(x, w, b):
    y = lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=_DN)
    return y + b


def init_params(key):
    ks = iter(jax.random.split(key, 64))

    def kern(kh, kw, cin, cout):
        lim = (1.0 / (kh * kw * cin)) ** 0.5
        return (jax.random.uniform(next(ks), (kh, kw, cin, cout), minval=-lim,
                                   maxval=lim), jnp.zeros((cout,)))

    def lin(out, inp):
        lim = (1.0 / inp) ** 0.5
        return (jax.random.uniform(next(ks), (out, inp), minval=-lim, maxval=lim),
                jnp.zeros((out,)))

    p = {}
    p["proj.w"], p["proj.b"] = kern(1, 1, C, D)
    p["gemb.w"], p["gemb.b"] = lin(D, G)
    for i in range(KBLOCKS):
        p[f"b{i}.0.w"], p[f"b{i}.0.b"] = kern(3, 3, D, D)
        p[f"b{i}.1.w"], p[f"b{i}.1.b"] = kern(3, 3, D, D)
    p["gate.0.w"], p["gate.0.b"] = lin(D, D)
    p["gate.2.w"], p["gate.2.b"] = lin(1, D)
    p["tq.w"], p["tq.b"] = lin(D, D)
    p["tk.w"], p["tk.b"] = lin(D, D)
    p["v.0.w"], p["v.0.b"] = lin(D, D)
    p["v.2.w"], p["v.2.b"] = lin(1, D)
    return p


def _dense(x, p, name):
    return x @ p[name + ".w"].T + p[name + ".b"]


def forward(p, grid, planet_ix, pmask, gf):
    B = grid.shape[0]
    x = jax.nn.relu(_conv(grid, p["proj.w"], p["proj.b"]))     # [B,H,W,D]
    g = jax.nn.relu(_dense(gf, p, "gemb"))                     # [B,D]
    x = x + g[:, None, None, :]
    for i in range(KBLOCKS):
        h = jax.nn.relu(_conv(x, p[f"b{i}.0.w"], p[f"b{i}.0.b"]))
        h = _conv(h, p[f"b{i}.1.w"], p[f"b{i}.1.b"])
        x = jax.nn.relu(x + h)                                 # residual
    fmap = x.reshape(B, H * W, D)
    idx = jnp.broadcast_to(planet_ix[:, :, None], (B, MAXP, D))
    emb = jnp.take_along_axis(fmap, idx, axis=1)               # [B,MAXP,D]

    gh = jax.nn.relu(_dense(emb, p, "gate.0"))
    gate = _dense(gh, p, "gate.2")[..., 0]                     # [B,MAXP]
    q = _dense(emb, p, "tq")
    k = _dense(emb, p, "tk")
    tgt = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(D)
    kpm = pmask < 0.5
    tgt = jnp.where(kpm[:, None, :], NEG, tgt)
    tgt = jnp.where(jnp.eye(MAXP, dtype=bool)[None], NEG, tgt)

    pooled = jnp.mean(fmap, axis=1)                            # [B,D]
    vh = jax.nn.relu(_dense(pooled, p, "v.0"))
    val = _dense(vh, p, "v.2")[..., 0]                         # [B]
    return gate, tgt, val
