"""P1 — train π_θ (entity-attention) by behavior cloning the strong field.

Runs on the 4090. Input = data/pi_theta_ds.npz from policy_encode.
Heads (Lux Flat-Neurons transfer A):
  gate   BCE on ALL owned slots (pos_weight counters the ~7% launch rate)
  target CE  ONLY on launch-positive slots (pointer over planets)
  frac   MSE ONLY on launch-positive slots
Small net (D=64, 2 attn layers) — must run fast on CPU at inference.
Exports weights as a flat .npz so the Kaggle bundle does pure-numpy
inference (no torch dep / import-timeout risk, cf. GBC-trees pattern).
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

D = 64


class PiTheta(nn.Module):
    def __init__(self, fdim, gdim, d=D, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(fdim, d)
        self.gemb = nn.Linear(gdim, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d),
                "att": nn.MultiheadAttention(d, heads, batch_first=True),
                "n2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                    nn.Linear(2 * d, d)),
            }))
        self.gate = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                                  nn.Linear(d, 1))
        self.frac = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                                  nn.Linear(d, 1))
        self.tq = nn.Linear(d, d)
        self.tk = nn.Linear(d, d)
        self.d = d

    def forward(self, pf, pmask, gf):
        # pf[B,P,F] pmask[B,P] gf[B,G]
        x = self.inp(pf) + self.gemb(gf).unsqueeze(1)
        kpm = pmask < 0.5                       # True = pad → ignored
        for b in self.blocks:
            h = b["n1"](x)
            a, _ = b["att"](h, h, h, key_padding_mask=kpm,
                            need_weights=False)
            x = x + a
            x = x + b["ff"](b["n2"](x))
        gate = self.gate(x).squeeze(-1)          # [B,P]
        frac = torch.sigmoid(self.frac(x).squeeze(-1))
        q = self.tq(x)
        k = self.tk(x)
        tgt = (q @ k.transpose(1, 2)) / (self.d ** 0.5)  # [B,Psrc,Ptgt]
        neg = torch.finfo(tgt.dtype).min
        tgt = tgt.masked_fill(kpm.unsqueeze(1), neg)      # mask pad targets
        eye = torch.eye(pf.shape[1], device=pf.device, dtype=torch.bool)
        tgt = tgt.masked_fill(eye.unsqueeze(0), neg)      # no self-target
        return gate, tgt, frac


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "data/pi_theta_ds.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/pi_theta_w.npz"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(ds)
    pf = torch.tensor(z["pf"]); pm = torch.tensor(z["pmask"])
    gf = torch.tensor(z["gf"]); om = torch.tensor(z["omask"])
    yl = torch.tensor(z["y_launch"]); yt = torch.tensor(z["y_target"])
    yfr = torch.tensor(z["y_frac"])
    N = pf.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    nval = max(1, N // 10)
    vi, ti = perm[:nval], perm[nval:]

    owned = om.bool()
    pos = (yl * om).sum().item()
    negc = owned.sum().item() - pos
    pw = torch.tensor([negc / max(1.0, pos)], device=dev)
    print(f"N={N} train={len(ti)} val={len(vi)} | owned={int(owned.sum())} "
          f"launch={int(pos)} ({pos/owned.sum().item():.1%}) pos_weight={pw.item():.1f}")

    net = PiTheta(pf.shape[-1], gf.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    bs = 256

    def batches(idx, shuf):
        idx = idx[torch.randperm(len(idx))] if shuf else idx
        for s in range(0, len(idx), bs):
            yield idx[s:s + bs]

    def run(idx, train):
        net.train(train)
        tot = gtp = gtn = 0
        tcorr = tcnt = 0
        fse = fcnt = 0.0
        lsum = 0.0
        torch.set_grad_enabled(train)
        for bi in batches(idx, train):
            PF = pf[bi].to(dev); PM = pm[bi].to(dev); GF = gf[bi].to(dev)
            OM = om[bi].to(dev).bool()
            YL = yl[bi].to(dev); YT = yt[bi].to(dev); YF = yfr[bi].to(dev)
            gate, tgt, frac = net(PF, PM, GF)
            lg = Fnn.binary_cross_entropy_with_logits(
                gate[OM], YL[OM], pos_weight=pw)
            launch = (YL > 0.5) & OM                 # launch-positive slots
            if launch.any():
                bsi, psi = launch.nonzero(as_tuple=True)
                lt = Fnn.cross_entropy(tgt[bsi, psi], YT[bsi, psi])
                lf = Fnn.mse_loss(frac[launch], YF[launch])
            else:
                lt = lf = torch.zeros((), device=dev)
            loss = lg + lt + lf
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            lsum += loss.item() * len(bi)
            with torch.no_grad():
                pr = (gate[OM] > 0)
                tr = YL[OM] > 0.5
                gtp += (pr & tr).sum().item()
                gtn += (~pr & ~tr).sum().item()
                tot += OM.sum().item()
                if launch.any():
                    tcorr += (tgt[bsi, psi].argmax(-1)
                              == YT[bsi, psi]).sum().item()
                    tcnt += len(bsi)
                    fse += (frac[launch] - YF[launch]).abs().sum().item()
                    fcnt += launch.sum().item()
        return (lsum / max(1, len(idx)), (gtp + gtn) / max(1, tot),
                gtp, tcorr / max(1, tcnt), fse / max(1, fcnt))

    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    best = 0.0
    for ep in range(epochs):
        run(ti, True)
        vl, gacc, gtp, tacc, fmae = run(vi, False)
        score = tacc + gacc
        flag = ""
        if score > best:
            best = score
            sd = {k: v.detach().cpu().numpy()
                  for k, v in net.state_dict().items()}
            np.savez(out, _F=np.int64(pf.shape[-1]),
                     _G=np.int64(gf.shape[-1]), _D=np.int64(D), **sd)
            flag = " *saved"
        if ep % 4 == 0 or flag:
            print(f"ep{ep:02d} val_loss={vl:.3f} gate_acc={gacc:.3f} "
                  f"gate_TP={gtp} tgt_top1={tacc:.3f} frac_MAE={fmae:.3f}"
                  f"{flag}")
    print(f"DONE best(gate_acc+tgt_top1)={best:.3f} → {out}")


if __name__ == "__main__":
    main()
