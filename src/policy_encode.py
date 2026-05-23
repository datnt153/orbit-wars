"""P1 — shared entity encoder + offline dataset builder for π_θ.

CRITICAL: `encode_state` is used BOTH to build the offline training set
AND at inference inside the rollout. One code path → no train/inference
skew (a classic silent BC killer, cf. the no-op-rollout bug).

Entity-based (Orbit Wars = continuous-space planets/fleets, no 2D grid
like Lux). Small, hand-built inductive-bias feature set — we have no
scale, so bias > capacity (discussion + Lux both stress this).

Per-state arrays (planets padded to MAXP, masked):
  pf   [MAXP, F]   per-planet features
  pmask[MAXP]      1 = real planet
  omask[MAXP]      1 = planet owned by `player` (a decision source)
  gf   [G]         global context (broadcast)
Labels (aligned to planet rows; valid where omask=1):
  y_launch[MAXP]   0/1   trained on ALL owned slots (gate head)
  y_target[MAXP]   int   planet-row index aimed at, -1 if no launch
  y_frac  [MAXP]   float ships/garrison, valid only where y_launch=1
(Lux transfer A: target+frac heads supervised ONLY on launch-positive
slots; gate head on all owned slots.)
"""
from __future__ import annotations

import math

import numpy as np

MAXP = 48                       # 20-40 planets + ≤4 transient comets, padded
F = 21                          # 15 base + 6 relational (nearest dists, threat, takeable, net_press)
G = 8                           # global feature dim
_PID, _OWN, _X, _Y, _R, _SHIPS, _PROD = range(7)
CENTER = 50.0
ROT_LIMIT = 50.0
_COMET_STEPS = (50, 150, 250, 350, 450)


def _ang(px, py, qx, qy):
    return math.atan2(qy - py, qx - px)


def _adiff(a, b):
    d = (a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def encode_state(obs, player):
    """obs dict/namespace → (pf[MAXP,F], pmask[MAXP], omask[MAXP], gf[G],
    planet_ids[list]). planet_ids maps row→engine planet id (for decoding
    a target row back to a move at inference)."""
    g = (obs.get if isinstance(obs, dict)
         else lambda k, d=None: getattr(obs, k, d))
    planets = list(g("planets", []) or [])[:MAXP]
    fleets = list(g("fleets", []) or [])
    comet_ids = set(g("comet_planet_ids", []) or [])
    step = int(g("step", 0) or 0)
    np_agents = 4 if any(
        int(p[_OWN]) >= 2 for p in planets if int(p[_OWN]) >= 0
    ) else 2

    n = len(planets)
    pf = np.zeros((MAXP, F), np.float32)
    pmask = np.zeros(MAXP, np.float32)
    omask = np.zeros(MAXP, np.float32)
    planet_ids = [-1] * MAXP

    # incoming fleet pressure per planet (by relation to `player`)
    press_mine = [0.0] * n
    press_enemy = [0.0] * n
    for f in fleets:
        fa = f[4]
        fx, fy = f[2], f[3]
        best_j, best_e = -1, 1e9
        for j, p in enumerate(planets):
            e = _adiff(fa, _ang(fx, fy, p[_X], p[_Y]))
            if e < best_e:
                best_e, best_j = e, j
        if best_j >= 0 and best_e < 0.5:
            if int(f[1]) == player:
                press_mine[best_j] += f[6]
            elif int(f[1]) >= 0:
                press_enemy[best_j] += f[6]

    # relational precompute (match jax_encode F15-20): nearest planet by relation
    PX = np.array([p[_X] for p in planets], np.float64)
    PY = np.array([p[_Y] for p in planets], np.float64)
    POW = np.array([int(p[_OWN]) for p in planets])
    dmat = np.sqrt((PX[:, None] - PX[None, :]) ** 2
                   + (PY[:, None] - PY[None, :]) ** 2) if n else np.zeros((0, 0))
    eye = np.eye(n, dtype=bool)

    def _nearest(maskj):
        if n == 0:
            return np.zeros(0)
        dd = np.where((~eye) & maskj[None, :], dmat, 1e9)
        return dd.min(axis=1)

    d_enemy_a = np.minimum(_nearest((POW >= 0) & (POW != player)), 100.0) / 70.0
    d_neutral_a = np.minimum(_nearest(POW == -1), 100.0) / 70.0
    d_mine_a = np.minimum(_nearest(POW == player), 100.0) / 70.0

    my_tot = opp_tot = neu_tot = 0.0
    my_np = 0
    for i, p in enumerate(planets):
        own = int(p[_OWN])
        ships = float(p[_SHIPS])
        if own == player:
            my_tot += ships + p[_PROD] * 8
            my_np += 1
        elif own == -1:
            neu_tot += ships
        else:
            opp_tot += ships + p[_PROD] * 8
        dx, dy = p[_X] - CENTER, p[_Y] - CENTER
        dsun = math.hypot(dx, dy)
        is_orb = 1.0 if (dsun + p[_R] < ROT_LIMIT) else 0.0
        pf[i] = (
            1.0 if own == player else 0.0,
            1.0 if (own >= 0 and own != player) else 0.0,
            1.0 if own == -1 else 0.0,
            math.log1p(ships) / 8.0,
            p[_PROD] / 5.0,
            p[_R] / 4.0,
            p[_X] / 100.0,
            p[_Y] / 100.0,
            dsun / 70.0,
            1.0 if p[_PID] in comet_ids else 0.0,
            is_orb,
            math.log1p(press_mine[i]) / 8.0,
            math.log1p(press_enemy[i]) / 8.0,
            1.0 if own == player else 0.0,
            1.0,
            float(d_enemy_a[i]), float(d_neutral_a[i]), float(d_mine_a[i]),
            min(max(press_enemy[i] / (ships + 1.0), 0.0), 3.0) / 3.0,   # threat
            min(max(press_mine[i] / (ships + 1.0), 0.0), 3.0) / 3.0,    # takeable
            math.tanh((press_mine[i] - press_enemy[i]) / 10.0),        # net_press
        )
        pmask[i] = 1.0
        planet_ids[i] = int(p[_PID])
        if own == player:
            omask[i] = 1.0

    nxt = next((c for c in _COMET_STEPS if c > step), 500)
    sc = 200.0
    gf = np.array([
        step / 500.0,
        1.0 if np_agents == 4 else 0.0,
        my_tot / sc, opp_tot / sc, neu_tot / sc,
        my_np / 20.0,
        (nxt - step) / 100.0,
        my_np / max(1, n),
    ], np.float32)
    return pf, pmask, omask, gf, planet_ids


def _sym(planets, fleets, mode):
    """4-fold board mirror (Orbit Wars symmetry; 2nd-place-Lux transfer).
    mode 0=id, 1=mirror-x, 2=mirror-y, 3=180°. Planet LIST ORDER kept →
    target-index / launch / ship_frac labels stay valid (move angle is
    re-derived from geometry at decode time, so it's not stored)."""
    if mode == 0:
        return planets, fleets
    fx = mode in (1, 3)        # flip x
    fy = mode in (2, 3)        # flip y
    P = []
    for p in planets:
        q = list(p)
        if fx:
            q[2] = 100.0 - q[2]
        if fy:
            q[3] = 100.0 - q[3]
        P.append(q)
    Fl = []
    for f in fleets:
        q = list(f)
        if fx:
            q[2] = 100.0 - q[2]
        if fy:
            q[3] = 100.0 - q[3]
        a = q[4]
        if fx:
            a = math.pi - a
        if fy:
            a = -a
        q[4] = math.atan2(math.sin(a), math.cos(a))
        Fl.append(q)
    return P, Fl


def build_dataset(out_path, days=None, min_rating=1500.0, max_per_day=120,
                  symmetry=True):
    """Encode every strong-player decision into padded arrays + labels.
    symmetry=True → ×4 via 4-fold board mirror augmentation."""
    from policy_dataset import extract

    PF, PM, OM, GF = [], [], [], []
    YL, YT, YF = [], [], []
    n = 0
    modes = (0, 1, 2, 3) if symmetry else (0,)
    for ps in extract(days=days, min_rating=min_rating,
                       max_per_day=max_per_day):
      for _m in modes:
        spl, sfl = _sym(ps.planets, ps.fleets, _m)
        obs = {"planets": spl, "fleets": sfl,
               "comet_planet_ids": [], "step": ps.step}
        pf, pm, om, gf, pids = encode_state(obs, ps.player)
        id2row = {pid: r for r, pid in enumerate(pids) if pid >= 0}
        yl = np.zeros(MAXP, np.float32)
        yt = np.full(MAXP, -1, np.int64)
        yf = np.zeros(MAXP, np.float32)
        for k, src_pid in enumerate(ps.src_planet_ids):
            r = id2row.get(src_pid)
            if r is None:
                continue
            if ps.launch[k]:
                yl[r] = 1.0
                ti = ps.target_idx[k]               # row in ps.planets
                yt[r] = ti if 0 <= ti < MAXP else -1
                yf[r] = max(1e-3, min(1.0, ps.ship_frac[k]))
        PF.append(pf); PM.append(pm); OM.append(om); GF.append(gf)
        YL.append(yl); YT.append(yt); YF.append(yf)
        n += 1

    np.savez_compressed(
        out_path,
        pf=np.stack(PF), pmask=np.stack(PM), omask=np.stack(OM),
        gf=np.stack(GF), y_launch=np.stack(YL), y_target=np.stack(YT),
        y_frac=np.stack(YF),
    )
    return n


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/pi_theta_ds.npz"
    n = build_dataset(out, max_per_day=120)
    d = np.load(out)
    om = d["omask"]; yl = d["y_launch"]
    owned = om.sum()
    launches = (yl * om).sum()
    print(f"states={n}  arrays→ {out}")
    print(f"  owned slots={int(owned)}  launches={int(launches)} "
          f"({launches/max(1,owned):.1%})")
    print(f"  pf={d['pf'].shape} gf={d['gf'].shape} "
          f"size={d['pf'].nbytes/1e6:.1f}MB(pf)")
    # sanity: target rows must be valid planet rows where launched
    bad = 0
    yt = d["y_target"]; pm = d["pmask"]
    for s in range(min(n, 5000)):
        for r in np.where((yl[s] * om[s]) > 0)[0]:
            t = yt[s, r]
            if t < 0 or pm[s, t] == 0:
                bad += 1
    print(f"  invalid target rows (sample 5k): {bad}")
