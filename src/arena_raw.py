"""Fast raw-policy arena via ow_sim — head-to-head between two weight sets.

Both policies share the entity-attention ActorCritic arch (reused from
ppo_train). Stochastic sampling, identical to the training rollout. Maps
are round-robin'd from bovard initial states; seats are swapped half the
games to remove side bias. This isolates RAW policy strength (no v4
rollout wrapper) — answers "has PPO surpassed the BC π_θ it forked from".

Usage:
    arena_raw.py A.npz B.npz [games_per_side] [n_envs]
Reports A win-rate vs B (draws shown separately), summed over both seats.
"""
import glob
import json
import sys

import numpy as np
import torch
import torch.nn.functional as Fnn

from ppo_train import ActorCritic, MAXP, F, G, DEV, ROOT
import ow_sim


def load_net(path):
    z = np.load(path)
    sd = {k: torch.tensor(z[k]) for k in z.files if not k.startswith("_")}
    net = ActorCritic().to(DEV)
    net.load_state_dict(sd)
    net.eval()
    steps = int(z["_STEPS"]) if "_STEPS" in z.files else -1
    return net, steps


def load_templates(n):
    eps = sorted(glob.glob(str(ROOT / "data" / "bovard" / "2026-05-04" /
                               "episodes" / "episodes" / "*.json")))
    out = []
    for p in eps[:n]:
        obs0 = json.loads(open(p).read())["steps"][0][0]["observation"]
        out.append(ow_sim.State(obs0, 6.0, 500))
    return out


@torch.no_grad()
def sample_seat(net, pf, pm, om, gf):
    E = pf.shape[0]
    g_l, t_l, _ = net(pf, pm, gf)
    gp = torch.sigmoid(g_l).clamp(1e-6, 1 - 1e-6)
    launch = torch.bernoulli(gp) * om
    probs = Fnn.softmax(t_l, dim=-1)              # (E, MAXP, MAXP)
    tgt = torch.multinomial(probs.reshape(E * MAXP, MAXP), 1).reshape(E, MAXP)
    return launch, tgt


def run(netA, netB, templates, n_envs, target_games, seatA):
    pool = ow_sim.EnvPool(templates[0], n_envs)
    for e in range(n_envs):
        pool.reset_one(e, templates[e % len(templates)])
    seatB = 1 - seatA
    wins = [0, 0, 0]   # A, B, draw
    games = 0
    while games < target_games:
        pf, pm, om, gf, _ = pool.observe_batch(2)
        pf = torch.from_numpy(pf.reshape(n_envs, 2, MAXP, F).copy()).to(DEV)
        pm = torch.from_numpy(pm.reshape(n_envs, 2, MAXP).copy()).to(DEV)
        om = torch.from_numpy(om.reshape(n_envs, 2, MAXP).copy()).to(DEV)
        gf = torch.from_numpy(gf.reshape(n_envs, 2, G).copy()).to(DEV)
        lA, tA = sample_seat(netA, pf[:, seatA], pm[:, seatA], om[:, seatA], gf[:, seatA])
        lB, tB = sample_seat(netB, pf[:, seatB], pm[:, seatB], om[:, seatB], gf[:, seatB])
        launch = torch.zeros(n_envs, 2, MAXP, device=DEV)
        tgt = torch.zeros(n_envs, 2, MAXP, dtype=torch.long, device=DEV)
        launch[:, seatA] = lA; launch[:, seatB] = lB
        tgt[:, seatA] = tA;    tgt[:, seatB] = tB
        launch_np = np.ascontiguousarray(launch.cpu().numpy().astype(np.float32))
        tgt_np = np.ascontiguousarray(tgt.cpu().numpy().astype(np.int64))
        sf_np = np.ones_like(launch_np, dtype=np.float32)
        pool.step_from_samples(launch_np, tgt_np, sf_np, 2)
        dm = pool.done_mask()
        rws = pool.rewards()
        for e in range(n_envs):
            if dm[e] and len(rws[e]) >= 2:
                rA = rws[e][seatA]
                wins[0 if rA > 0 else (1 if rA < 0 else 2)] += 1
                games += 1
                pool.reset_one(e, templates[(e + games) % len(templates)])
    return wins


def main():
    a_path = sys.argv[1]
    b_path = sys.argv[2]
    per_side = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    n_envs = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    netA, sA = load_net(a_path)
    netB, sB = load_net(b_path)
    templates = load_templates(n_envs)
    print(f"A = {a_path} ({sA/1e6:.1f}M steps)")
    print(f"B = {b_path} ({sB/1e6:.1f}M steps)")
    print(f"maps={len(templates)} per_side={per_side} dev={DEV}", flush=True)

    # seatA=0 then seatA=1, summed (removes side bias).
    w0 = run(netA, netB, templates, n_envs, per_side, 0)
    w1 = run(netA, netB, templates, n_envs, per_side, 1)
    A = w0[0] + w1[0]; B = w0[1] + w1[1]; D = w0[2] + w1[2]
    tot = A + B + D
    decisive = A + B
    wr = A / decisive if decisive else 0.0
    print(f"seat0 A/B/draw = {w0}")
    print(f"seat1 A/B/draw = {w1}")
    print(f"TOTAL  A={A}  B={B}  draw={D}  (n={tot})")
    print(f"A win-rate (draws excluded) = {wr:.3f}  "
          f"[A={A}/{decisive}]")


if __name__ == "__main__":
    main()
