"""Naked PPO agent bundle — the policy plays its OWN game (greedy gate, all ships).

Unlike build_pitheta_cand (PPO as candidate on v4-rollout, floor v4), this is the
RAW policy → shows its true ladder strength. Use when arena vs v4 is strong
(v3 295M: 4P-greedy 44%). Pair always-safe with a known floor in the 2-latest slots.

No torch/jax/__file__. Inlines policy_encode + pi_theta_infer (greedy). Writes
submit/main_ppo_naked.py.
"""
import base64
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
W_PATH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "data" / "jax_ppo_v3_295m.npz")
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "main_ppo_naked.py"

ENC = (ROOT / "src" / "policy_encode.py").read_text()
INF = (ROOT / "src" / "pi_theta_infer.py").read_text()
W_B64 = base64.b64encode(W_PATH.read_bytes()).decode()


def _nf(s):
    return s.replace("from __future__ import annotations\n", "")


ENC_BLOCK = _nf(ENC[ENC.index("MAXP = 48"):ENC.index("\ndef build_dataset")])
INF_BLOCK = INF[INF.index("def _ln("):INF.index('if __name__')]

MAIN = '''"""Naked PPO policy (greedy). Self-contained, no deps beyond numpy."""
import base64 as _b64
import io as _io
import math

import numpy as np

_PI_W_B64 = "@@PIW@@"

# ---------------------- encode_state (inlined, parity-tested) ----------------------
@@ENC@@

# ---------------------- weights ----------------------
_z = np.load(_io.BytesIO(_b64.b64decode(_PI_W_B64)))
_W = {k: _z[k].astype(np.float32) for k in _z.files if not k.startswith("_")}
_META = (int(_z["_F"]), int(_z["_G"]), int(_z["_D"]))


def load(*a, **k):
    return _W, _META


# ---------------------- pi_theta numpy forward + decode (greedy) ----------------------
@@INF@@


def agent(obs, config=None):
    try:
        p = obs["player"] if isinstance(obs, dict) else obs.player
        r = decode_moves(obs, int(p))
        return list(r) if r else []
    except Exception:
        return []


__all__ = ["agent"]
'''

MAIN = (MAIN.replace("@@PIW@@", W_B64).replace("@@ENC@@", ENC_BLOCK)
        .replace("@@INF@@", INF_BLOCK))
out = ROOT / "submit" / OUT_NAME
out.write_text(MAIN)
print(f"Wrote {out} ({len(MAIN)} bytes)  weights={W_PATH.name}")
