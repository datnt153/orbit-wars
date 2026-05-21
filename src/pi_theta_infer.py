"""P1 — pure-numpy π_θ inference (NO torch → no import cost/timeout in
the Kaggle bundle; same zero-deps pattern as the GBC-trees agent).

MUST bit-match the torch model (src/policy_train.py PiTheta) or we get
silent train/inference skew. `__main__` runs a parity test vs torch and
a CPU speed test — verify before integrating (cf. the no-op-rollout bug:
trust nothing, parity-test everything).

`decode_moves(obs, player)` → [[from_id, angle, ships], ...] : the move
π_θ would play, for use as a FAST rollout opponent replacing v4 (~9ms →
target ≪1ms).
"""
from __future__ import annotations

import numpy as np

from policy_encode import encode_state

_W = None
_META = None


def load(path="data/pi_theta_w.npz"):
    global _W, _META
    z = np.load(path)
    _W = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("_")}
    _META = (int(z["_F"]), int(z["_G"]), int(z["_D"]))
    return _W, _META


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)            # biased (torch LayerNorm)
    return (x - m) / np.sqrt(v + eps) * w + b


def _lin(x, w, b):
    return x @ w.T + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _mha(h, w, pfx, kpm, heads=4):
    # torch nn.MultiheadAttention(batch_first=True), packed in_proj.
    S, D = h.shape
    dh = D // heads
    ipw = w[f"{pfx}.in_proj_weight"]
    ipb = w[f"{pfx}.in_proj_bias"]
    q = h @ ipw[:D].T + ipb[:D]
    k = h @ ipw[D:2 * D].T + ipb[D:2 * D]
    v = h @ ipw[2 * D:].T + ipb[2 * D:]
    q = q.reshape(S, heads, dh).transpose(1, 0, 2)        # [H,S,dh]
    k = k.reshape(S, heads, dh).transpose(1, 0, 2)
    v = v.reshape(S, heads, dh).transpose(1, 0, 2)
    sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh)         # [H,S,S]
    sc = np.where(kpm[None, None, :], -np.inf, sc)        # mask pad keys
    a = _softmax(sc) @ v                                  # [H,S,dh]
    a = a.transpose(1, 0, 2).reshape(S, D)
    return _lin(a, w[f"{pfx}.out_proj.weight"], w[f"{pfx}.out_proj.bias"])


def forward(pf, pmask, gf):
    """pf[P,F] pmask[P] gf[G] → gate[P], tgt[P,P], frac[P]."""
    w = _W
    x = _lin(pf, w["inp.weight"], w["inp.bias"]) + _lin(
        gf, w["gemb.weight"], w["gemb.bias"])
    kpm = pmask < 0.5
    for bi in (0, 1):
        p = f"blocks.{bi}"
        h = _ln(x, w[f"{p}.n1.weight"], w[f"{p}.n1.bias"])
        x = x + _mha(h, w, f"{p}.att", kpm)
        h2 = _ln(x, w[f"{p}.n2.weight"], w[f"{p}.n2.bias"])
        ff = np.maximum(0.0, _lin(h2, w[f"{p}.ff.0.weight"],
                                  w[f"{p}.ff.0.bias"]))
        x = x + _lin(ff, w[f"{p}.ff.2.weight"], w[f"{p}.ff.2.bias"])
    gh = np.maximum(0.0, _lin(x, w["gate.0.weight"], w["gate.0.bias"]))
    gate = _lin(gh, w["gate.2.weight"], w["gate.2.bias"]).squeeze(-1)
    fh = np.maximum(0.0, _lin(x, w["frac.0.weight"], w["frac.0.bias"]))
    frac = 1.0 / (1.0 + np.exp(
        -_lin(fh, w["frac.2.weight"], w["frac.2.bias"]).squeeze(-1)))
    q = _lin(x, w["tq.weight"], w["tq.bias"])
    k = _lin(x, w["tk.weight"], w["tk.bias"])
    D = q.shape[-1]
    tgt = (q @ k.T) / np.sqrt(D)
    tgt[:, kpm] = -np.inf
    np.fill_diagonal(tgt, -np.inf)
    return gate, tgt, frac


def decode_moves(obs, player, gate_thr=0.0):
    """π_θ's joint move for `player` at `obs` (rollout-opponent use)."""
    if _W is None:
        load()
    pf, pmask, omask, gf, pids = encode_state(obs, player)
    gate, tgt, frac = forward(pf, pmask, gf)
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    pos = {int(p[0]): (p[2], p[3]) for p in planets}
    ships_of = {int(p[0]): p[5] for p in planets}
    moves = []
    import math
    for r in np.where(omask > 0.5)[0]:
        if gate[r] <= gate_thr:
            continue
        tr = int(np.argmax(tgt[r]))
        sid, tid = pids[r], pids[tr]
        if sid < 0 or tid < 0:
            continue
        n = int(round(frac[r] * ships_of.get(sid, 0)))
        if n <= 0:
            continue
        sx, sy = pos[sid]
        tx, ty = pos[tid]
        moves.append([sid, math.atan2(ty - sy, tx - sx), n])
    return moves


if __name__ == "__main__":
    import json, os, time, sys
    sys.path.insert(0, "src")
    load()
    print(f"loaded {sum(v.size for v in _W.values())} params  meta={_META}")

    # --- PARITY vs torch (must match or train/inference skew) ---
    import torch
    from policy_train import PiTheta
    F, G, Dm = _META
    net = PiTheta(F, G)
    sd = {}
    z = np.load("data/pi_theta_w.npz")
    for k in z.files:
        if k.startswith("_"):
            continue
        sd[k] = torch.tensor(z[k])
    net.load_state_dict(sd)
    net.eval()

    d = "data/bovard/2026-05-04/episodes/episodes"
    rep = json.load(open(os.path.join(d, os.listdir(d)[0])))
    maxg = maxt = maxf = 0.0
    ts = []
    for st in rep["steps"][20:60]:
        obs = st[0]["observation"]
        pf, pm, om, gf, _ = encode_state(obs, 0)
        t0 = time.perf_counter()
        g_np, t_np, f_np = forward(pf, pm, gf)
        ts.append(time.perf_counter() - t0)
        with torch.no_grad():
            g_t, t_t, f_t = net(torch.tensor(pf)[None],
                                torch.tensor(pm)[None],
                                torch.tensor(gf)[None])
        g_t = g_t[0].numpy(); f_t = f_t[0].numpy()
        t_t = t_t[0].numpy()
        fin = np.isfinite(t_np) & np.isfinite(t_t)
        maxg = max(maxg, np.abs(g_np - g_t).max())
        maxf = max(maxf, np.abs(f_np - f_t).max())
        maxt = max(maxt, np.abs(t_np[fin] - t_t[fin]).max())
    ts.sort()
    print(f"PARITY numpy vs torch  max|Δ| gate={maxg:.2e} "
          f"target={maxt:.2e} frac={maxf:.2e}  "
          f"({'OK' if max(maxg,maxt,maxf) < 1e-3 else 'FAIL — skew!'})")
    print(f"forward() CPU: mean={1000*sum(ts)/len(ts):.3f}ms "
          f"p90={1000*ts[int(len(ts)*0.9)]:.3f}ms  (v4 was ~9ms)")
    # full decode_moves timing (encode + forward + decode)
    ts2 = []
    for st in rep["steps"][20:60]:
        obs = st[0]["observation"]
        t0 = time.perf_counter()
        decode_moves(obs, 0)
        ts2.append(time.perf_counter() - t0)
    ts2.sort()
    print(f"decode_moves CPU: mean={1000*sum(ts2)/len(ts2):.3f}ms "
          f"p90={1000*ts2[int(len(ts2)*0.9)]:.3f}ms")
