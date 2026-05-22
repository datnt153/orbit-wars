"""R4 — PPO self-play on ow_sim.EnvPool (vectorized).

Entity-attention actor-critic. Sample + log-prob recompute fully tensor-
based across (envs × players). Action construction is the only Python
loop (to format moves for Rust env). Sparse ±1 terminal reward, GAE.
Saves checkpoint .npz (numpy-friendly → pure-numpy inference bundle).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ow_sim
from policy_encode import encode_state, MAXP, F, G

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):
    def __init__(self, fdim=F, gdim=G, d=64, heads=4, layers=2):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(fdim, d)
        self.gemb = nn.Linear(gdim, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d),
                "att": nn.MultiheadAttention(d, heads, batch_first=True),
                "n2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(),
                                     nn.Linear(2*d, d)),
            }))
        self.gate = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.tq = nn.Linear(d, d)
        self.tk = nn.Linear(d, d)
        self.v_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, pf, pmask, gf):
        x = self.inp(pf) + self.gemb(gf).unsqueeze(1)
        kpm = pmask < 0.5
        for b in self.blocks:
            h = b["n1"](x)
            a, _ = b["att"](h, h, h, key_padding_mask=kpm, need_weights=False)
            x = x + a
            x = x + b["ff"](b["n2"](x))
        gate = self.gate(x).squeeze(-1)
        q = self.tq(x); k = self.tk(x)
        tgt = (q @ k.transpose(1, 2)) / (self.d ** 0.5)
        neg = torch.finfo(tgt.dtype).min
        tgt = tgt.masked_fill(kpm.unsqueeze(1), neg)
        eye = torch.eye(pf.shape[1], device=pf.device, dtype=torch.bool)
        tgt = tgt.masked_fill(eye.unsqueeze(0), neg)
        m = pmask.unsqueeze(-1)
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        value = self.v_head(pooled).squeeze(-1)
        return gate, tgt, value


def encode_batch(states_dicts, num_agents):
    """Stack encode_state outputs for all (env, player) → tensors."""
    n_envs = len(states_dicts)
    pf = np.zeros((n_envs, num_agents, MAXP, F), np.float32)
    pm = np.zeros((n_envs, num_agents, MAXP), np.float32)
    om = np.zeros((n_envs, num_agents, MAXP), np.float32)
    gf = np.zeros((n_envs, num_agents, G), np.float32)
    pids_all = []
    for e, sd in enumerate(states_dicts):
        env_pids = None
        for p in range(num_agents):
            obs = {"planets": sd["planets"], "fleets": sd["fleets"],
                   "comet_planet_ids": [], "step": int(sd.get("step", 0))}
            ef, em, eo, eg, pids = encode_state(obs, p)
            pf[e, p] = ef; pm[e, p] = em; om[e, p] = eo; gf[e, p] = eg
            if env_pids is None: env_pids = pids
        pids_all.append(env_pids)
    return pf, pm, om, gf, pids_all


def joint_actions_from_samples(launch_np, target_np, pids_all, states_dicts,
                                num_agents):
    """Build EnvPool joint_actions_batch from sampled (launch, target)."""
    out = []
    for e, sd in enumerate(states_dicts):
        planets = sd["planets"]
        src_to_idx = {int(p[0]): i for i, p in enumerate(planets)}
        env_acts = []
        for p in range(num_agents):
            moves = []
            for r in range(MAXP):
                if launch_np[e, p, r] < 0.5: continue
                t = int(target_np[e, p, r])
                if t < 0 or t >= MAXP: continue
                src_id = pids_all[e][r]; tgt_id = pids_all[e][t]
                if src_id < 0 or tgt_id < 0: continue
                if src_id not in src_to_idx or tgt_id not in src_to_idx: continue
                sp = planets[src_to_idx[src_id]]
                tp = planets[src_to_idx[tgt_id]]
                ships = int(sp[5])
                if ships <= 0: continue
                angle = math.atan2(tp[3]-sp[3], tp[2]-sp[2])
                moves.append([int(src_id), float(angle), ships])  # all-in
            env_acts.append(moves)
        out.append(env_acts)
    return out


def log_prob_and_entropy(gate_logits, tgt_logits, omask, pmask,
                          launch, target):
    """Vectorized log-prob & entropy of sampled (launch, target) under the
    current policy. Shapes:
      gate_logits [B, P]   tgt_logits [B, P, P]
      omask, pmask, launch, target [B, P]  (target is int64; -1 unused)
    Returns lp [B], ent [B].
    """
    eps = 1e-6
    gp = torch.sigmoid(gate_logits).clamp(eps, 1-eps)
    bern_lp = launch * torch.log(gp) + (1-launch) * torch.log(1-gp)
    bern_lp = bern_lp * omask
    bern_ent = -(gp * torch.log(gp) + (1-gp) * torch.log(1-gp)) * omask
    log_softmax_t = Fnn.log_softmax(tgt_logits, dim=-1)
    # Gather log-prob of chosen target for each source row.
    tgt_safe = target.clamp(min=0).unsqueeze(-1)
    target_lp = log_softmax_t.gather(-1, tgt_safe).squeeze(-1)
    # Only count for launched slots
    launch_target = launch * omask
    tgt_term = target_lp * launch_target
    # entropy of target dist for launched rows
    probs_t = log_softmax_t.exp()
    tgt_ent_full = -(probs_t * log_softmax_t).sum(-1)
    tgt_ent = tgt_ent_full * launch_target
    lp = bern_lp.sum(-1) + tgt_term.sum(-1)
    ent = bern_ent.sum(-1) + tgt_ent.sum(-1)
    return lp, ent


def main():
    n_envs   = int(os.environ.get("N_ENVS", "32"))
    n_steps  = int(os.environ.get("N_STEPS", "64"))
    epochs   = int(os.environ.get("PPO_EPOCHS", "4"))
    mb       = int(os.environ.get("MB", "512"))
    lr       = float(os.environ.get("LR", "3e-4"))
    gamma    = 0.999
    lam      = 0.95
    clip     = 0.2
    ent_coef = float(os.environ.get("ENT", "0.005"))  # ANNEALED linearly → 0
    val_coef = 0.5
    max_grad = 0.5
    n_updates = int(os.environ.get("UPDATES", "50"))
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/ppo_w.npz"
    log_every = max(1, n_updates // 20)
    num_agents = 2

    # --- league / anchor opponent (fixes pure-self-play entropy drift) ---
    anchor_frac = float(os.environ.get("ANCHOR_FRAC", "0.5"))
    n_anchor = int(round(n_envs * anchor_frac))   # envs [0:n_anchor]: learner(p0) vs anchor(p1)
    n_maps   = int(os.environ.get("N_MAPS", "64"))
    promote_thr = float(os.environ.get("PROMOTE_THR", "0.55"))
    promote_min = int(os.environ.get("PROMOTE_MIN_GAMES", "150"))

    # Multi-map templates (round-robin) for generalization — single map overfits.
    eps_glob = sorted((ROOT / "data" / "bovard").glob("*/episodes/episodes/*.json"))
    if not eps_glob:
        eps_glob = sorted((ROOT / "data" / "bovard").glob("**/*.json"))
    templates = []
    for p in eps_glob[:n_maps]:
        o = json.loads(p.read_text())["steps"][0][0]["observation"]
        templates.append(ow_sim.State(o, 6.0, 500))
    if not templates:
        raise SystemExit("no bovard maps under data/bovard/")
    template = templates[0]
    pool = ow_sim.EnvPool(template, n_envs)
    for e in range(n_envs):
        pool.reset_one(e, templates[e % len(templates)])
    shape_scale = float(os.environ.get("SHAPE", "0.01"))  # dense reward scale

    net = ActorCritic().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    resume = os.environ.get("RESUME", "")
    start_steps = 0
    if resume and Path(resume).exists():
        z = np.load(resume)
        sd = {k: torch.tensor(z[k]) for k in z.files if not k.startswith("_")}
        net.load_state_dict(sd)
        start_steps = int(z.get("_STEPS", 0))
        # restore Adam moments (resume stability — fresh Adam destabilises PPO)
        opt_pt = resume + ".opt.pt"
        if Path(opt_pt).exists():
            try:
                ck = torch.load(opt_pt, map_location=DEV)
                opt.load_state_dict(ck["opt"])
                print(f"restored Adam state from {opt_pt}", flush=True)
            except Exception as e:
                print(f"Adam restore failed ({e}); fresh optimizer", flush=True)
        print(f"resumed from {resume}  prior_steps={start_steps}", flush=True)

    # Frozen anchor opponent (starts == resume / best-so-far policy).
    anchor_path = os.environ.get("ANCHOR_PATH", resume)
    anchor_net = ActorCritic().to(DEV)
    if anchor_path and Path(anchor_path).exists():
        za = np.load(anchor_path)
        anchor_net.load_state_dict({k: torch.tensor(za[k]) for k in za.files
                                    if not k.startswith("_")})
    else:
        anchor_net.load_state_dict(net.state_dict())
    anchor_net.eval()
    for pp in anchor_net.parameters():
        pp.requires_grad_(False)
    best_path = os.environ.get("BEST_PATH", "data/ppo_best.npz")

    print(f"Device {DEV} | params {sum(p.numel() for p in net.parameters())} | "
          f"envs={n_envs} (anchor={n_anchor}) maps={len(templates)} "
          f"steps={n_steps} updates={n_updates} mb={mb} ent={ent_coef}->0",
          flush=True)

    # Optional Wandb logging — set WANDB=1 to enable.
    use_wandb = os.environ.get("WANDB", "0") == "1"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            _wandb.init(
                project=os.environ.get("WANDB_PROJECT", "orbit-wars"),
                name=os.environ.get("WANDB_NAME", None),
                config={
                    "n_envs": n_envs, "n_steps": n_steps, "n_updates": n_updates,
                    "mb": mb, "lr": lr, "gamma": gamma, "lam": lam,
                    "clip": clip, "ent_coef": ent_coef, "val_coef": val_coef,
                    "shape_scale": shape_scale, "resume_steps": start_steps,
                    "params": sum(p.numel() for p in net.parameters()),
                    "device": DEV,
                },
            )
            wandb = _wandb
            print("wandb enabled", flush=True)
        except Exception as e:
            print(f"wandb disabled (init failed: {e})", flush=True)
            wandb = None

    # Buffers: (T, E, P, ...)
    sh = (n_steps, n_envs, num_agents)
    buf_pf = torch.zeros(sh + (MAXP, F), device=DEV)
    buf_pm = torch.zeros(sh + (MAXP,), device=DEV)
    buf_om = torch.zeros(sh + (MAXP,), device=DEV)
    buf_gf = torch.zeros(sh + (G,), device=DEV)
    buf_launch = torch.zeros(sh + (MAXP,), device=DEV)
    buf_target = torch.full(sh + (MAXP,), -1, dtype=torch.long, device=DEV)
    buf_lp = torch.zeros(sh, device=DEV)
    buf_val = torch.zeros(sh, device=DEV)
    buf_rew = torch.zeros(sh, device=DEV)
    buf_done = torch.zeros(sh, device=DEV)

    # Train only on learner-controlled slots: p0 everywhere; p1 only in
    # self-play envs (anchor controls p1 in envs [0:n_anchor]).
    slot_mask = torch.ones(n_envs, num_agents, device=DEV)
    if n_anchor > 0:
        slot_mask[:n_anchor, 1] = 0.0

    @torch.no_grad()
    def _sample(net_, pf_, pm_, om_, gf_):
        bn = pf_.shape[0]
        g_l, t_l, _ = net_(pf_, pm_, gf_)
        gp_ = torch.sigmoid(g_l).clamp(1e-6, 1 - 1e-6)
        lc = torch.bernoulli(gp_) * om_
        pr = Fnn.softmax(t_l, dim=-1).reshape(bn * MAXP, MAXP)
        tg = torch.multinomial(pr, 1).reshape(bn, MAXP)
        return lc, tg

    t0 = time.time()
    total_env_steps = 0
    ep_rewards = []
    anchor_results = []   # 1=learner(p0) beat anchor, 0=lost, in anchor envs

    prev_diffs = np.array(pool.diff_vs_avg_opp(num_agents), dtype=np.float64)
    for update in range(n_updates):
        # ---------- rollout (Rust-fast: observe_batch + step_from_samples) ----------
        for t in range(n_steps):
            pf_np, pm_np, om_np, gf_np, _pids = pool.observe_batch(num_agents)
            B = n_envs * num_agents
            pf_t = torch.from_numpy(pf_np.reshape(B, MAXP, F).copy()).to(DEV)
            pm_t = torch.from_numpy(pm_np.reshape(B, MAXP).copy()).to(DEV)
            om_t = torch.from_numpy(om_np.reshape(B, MAXP).copy()).to(DEV)
            gf_t = torch.from_numpy(gf_np.reshape(B, G).copy()).to(DEV)
            with torch.no_grad():
                g_l, t_l, v = net(pf_t, pm_t, gf_t)
                gp = torch.sigmoid(g_l).clamp(1e-6, 1-1e-6)
                launch = (torch.bernoulli(gp) * om_t)
                probs_t = Fnn.softmax(t_l, dim=-1)
                flat = probs_t.reshape(B*MAXP, MAXP)
                tgt = torch.multinomial(flat, 1).reshape(B, MAXP)
                lp, _ = log_prob_and_entropy(g_l, t_l, om_t, pm_t, launch, tgt)
            launch_b = launch.reshape(n_envs, num_agents, MAXP)
            tgt_b = tgt.reshape(n_envs, num_agents, MAXP)
            # Anchor opponent acts as p1 in envs [0:n_anchor] (frozen, no grad).
            if n_anchor > 0:
                pf_e = pf_t.reshape(n_envs, num_agents, MAXP, F)[:n_anchor, 1]
                pm_e = pm_t.reshape(n_envs, num_agents, MAXP)[:n_anchor, 1]
                om_e = om_t.reshape(n_envs, num_agents, MAXP)[:n_anchor, 1]
                gf_e = gf_t.reshape(n_envs, num_agents, G)[:n_anchor, 1]
                a_lc, a_tg = _sample(anchor_net, pf_e, pm_e, om_e, gf_e)
                launch_b[:n_anchor, 1] = a_lc
                tgt_b[:n_anchor, 1] = a_tg
            launch_np = np.ascontiguousarray(launch_b.cpu().numpy().astype(np.float32))
            tgt_np = np.ascontiguousarray(tgt_b.cpu().numpy().astype(np.int64))
            sf_np = np.ones_like(launch_np, dtype=np.float32)
            pool.step_from_samples(launch_np, tgt_np, sf_np, num_agents)
            # Dense shaped reward = Δ(my_total − avg_opp) * scale, in [-eps,+eps]
            new_diffs = np.array(pool.diff_vs_avg_opp(num_agents), dtype=np.float64)
            shaped = (new_diffs - prev_diffs) * shape_scale  # [n_envs, num_agents]
            prev_diffs = new_diffs
            buf_pf[t]     = pf_t.reshape(n_envs, num_agents, MAXP, F)
            buf_pm[t]     = pm_t.reshape(n_envs, num_agents, MAXP)
            buf_om[t]     = om_t.reshape(n_envs, num_agents, MAXP)
            buf_gf[t]     = gf_t.reshape(n_envs, num_agents, G)
            buf_launch[t] = launch_b
            buf_target[t] = tgt_b
            buf_lp[t]     = lp.reshape(n_envs, num_agents)
            buf_val[t]    = v.reshape(n_envs, num_agents)
            # base reward = shaped step reward
            for e in range(n_envs):
                for p in range(num_agents):
                    buf_rew[t, e, p] = float(shaped[e, p])
            dm = pool.done_mask()
            rws = pool.rewards()
            for e in range(n_envs):
                if dm[e]:
                    if len(rws[e]) >= num_agents:
                        for p in range(num_agents):
                            buf_rew[t, e, p] += float(rws[e][p])   # +terminal
                            buf_done[t, e, p] = 1.0
                        ep_rewards.append(int(rws[e][0]))
                        if e < n_anchor:   # learner(p0) vs frozen anchor(p1)
                            anchor_results.append(1 if rws[e][0] > 0 else 0)
                    pool.reset_one(e, templates[(e + len(ep_rewards)) % len(templates)])
                    prev_diffs[e] = pool.diff_vs_avg_opp(num_agents)[e]
            total_env_steps += n_envs

        # ---------- GAE ----------
        adv = torch.zeros_like(buf_val)
        ret = torch.zeros_like(buf_val)
        last_gae = torch.zeros((n_envs, num_agents), device=DEV)
        next_value = torch.zeros((n_envs, num_agents), device=DEV)
        for t in reversed(range(n_steps)):
            mask = 1.0 - buf_done[t]
            delta = buf_rew[t] + gamma * next_value * mask - buf_val[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
            ret[t] = adv[t] + buf_val[t]
            next_value = buf_val[t]

        # flatten
        flat = lambda x, *rest: x.reshape(n_steps*n_envs*num_agents, *rest)
        pf_F = flat(buf_pf, MAXP, F)
        pm_F = flat(buf_pm, MAXP)
        om_F = flat(buf_om, MAXP)
        gf_F = flat(buf_gf, G)
        launch_F = flat(buf_launch, MAXP)
        target_F = flat(buf_target, MAXP)
        lp_old = flat(buf_lp)
        adv_F = flat(adv); ret_F = flat(ret)
        tmask_F = slot_mask.unsqueeze(0).expand(n_steps, n_envs, num_agents).reshape(-1)
        _m = tmask_F > 0.5
        adv_F = (adv_F - adv_F[_m].mean()) / (adv_F[_m].std() + 1e-6)
        N = pf_F.shape[0]
        ent_coef_t = ent_coef * max(0.0, 1.0 - update / max(1, n_updates))  # anneal → 0

        last_pl = last_vl = last_ent = last_kl = 0.0
        for _ep in range(epochs):
            perm = torch.randperm(N, device=DEV)
            for s in range(0, N, mb):
                ib = perm[s:s+mb]
                tb = tmask_F[ib]
                denom = tb.sum().clamp_min(1.0)
                g_l, t_l, v = net(pf_F[ib], pm_F[ib], gf_F[ib])
                lp_new, ent = log_prob_and_entropy(
                    g_l, t_l, om_F[ib], pm_F[ib],
                    launch_F[ib], target_F[ib])
                ratio = torch.exp((lp_new - lp_old[ib]).clamp(-20, 20))
                surr1 = ratio * adv_F[ib]
                surr2 = torch.clamp(ratio, 1-clip, 1+clip) * adv_F[ib]
                policy_loss = -(torch.min(surr1, surr2) * tb).sum() / denom
                value_loss = (Fnn.mse_loss(v, ret_F[ib], reduction="none")
                              * tb).sum() / denom
                ent_mean = (ent * tb).sum() / denom
                loss = policy_loss + val_coef * value_loss - ent_coef_t * ent_mean
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad)
                opt.step()
                last_pl = policy_loss.item(); last_vl = value_loss.item()
                last_ent = ent_mean.item()
                last_kl = (((lp_old[ib] - lp_new) * tb).sum() / denom).item()

        elapsed = time.time() - t0
        sps = total_env_steps / elapsed
        recent = ep_rewards[-50:] if ep_rewards else [0]
        win_rate = sum(1 for r in recent if r > 0) / max(1, len(recent))

        # League: promote anchor once learner clearly beats it (curriculum).
        win_window = anchor_results[-promote_min:]
        awr = sum(win_window) / len(win_window) if win_window else 0.0
        promoted = False
        if len(anchor_results) >= promote_min and awr > promote_thr:
            anchor_net.load_state_dict(net.state_dict())
            anchor_net.eval()
            for pp in anchor_net.parameters():
                pp.requires_grad_(False)
            bsd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
            np.savez(best_path, _F=np.int64(F), _G=np.int64(G), _D=np.int64(64),
                     _UPD=np.int64(update + 1),
                     _STEPS=np.int64(total_env_steps + start_steps), **bsd)
            anchor_results.clear()
            promoted = True
            print(f"[league] PROMOTED anchor awr={awr:.2f} -> {best_path} "
                  f"(update {update})", flush=True)

        if update % log_every == 0 or update == n_updates - 1:
            print(f"upd{update:03d} steps={total_env_steps:>8} sps={sps:>6.0f} "
                  f"eps={len(ep_rewards)} wr={win_rate:.2f} awr={awr:.2f} "
                  f"pl={last_pl:.3f} vl={last_vl:.3f} ent={last_ent:.2f} "
                  f"entc={ent_coef_t:.4f} kl={last_kl:+.3f}", flush=True)
        if wandb is not None:
            wandb.log({
                "update": update,
                "env_steps": total_env_steps + start_steps,
                "sps": sps,
                "episodes": len(ep_rewards),
                "win_rate_p0_recent50": win_rate,
                "win_rate_vs_anchor": awr,
                "anchor_promoted": int(promoted),
                "ent_coef": ent_coef_t,
                "policy_loss": last_pl, "value_loss": last_vl,
                "entropy": last_ent, "kl": last_kl,
            }, step=total_env_steps + start_steps)

        # save every 20 updates (or last) so daily-pull always has a fresh ckpt
        if (update + 1) % 20 == 0 or update == n_updates - 1:
            sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
            np.savez(out_path,
                     _F=np.int64(F), _G=np.int64(G), _D=np.int64(64),
                     _UPD=np.int64(update+1),
                     _STEPS=np.int64(total_env_steps + start_steps),
                     **sd)
            torch.save({"opt": opt.state_dict(),
                        "steps": int(total_env_steps + start_steps),
                        "update": int(update + 1)}, out_path + ".opt.pt")

    print(f"DONE wall {time.time()-t0:.0f}s  total env-steps {total_env_steps} "
          f"saved {out_path}")


if __name__ == "__main__":
    main()
