# Orbit Wars — RTS Kaggle Competition

Solo entry to [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
(2P/4P space RTS, 500 turns, 1s per turn, ~2,650 teams, deadline 2026-06-23).

This repo is **the honest journal** of one team's run — what worked, what
didn't, and the hard-earned lessons. Verified-correct primitives (a perfect
Python simulator, a faster Rust port, bit-identical parity tests) underpin
every claim. Negative results are kept on equal footing with positive ones.

## Headline numbers

| Submission | Ladder score | What it was |
|---|--:|---|
| Pure rule-base v4 | 977.4 | Baseline strong heuristic |
| `ensemble_E` (buggy) | 944.6 → 920 → fading | The "deep adversarial rollout" — that never actually ran |
| **`ensemble_fixed`** | **+57 → 1004 → drift 977** | Fix one `_AttrDict` wrapper |
| **`ensemble_pitheta_cand`** | **1010.4 → 1007.9** | π_θ (offline-BC from top replays) as 3rd candidate |
| `ensemble_ppo_cand` | PENDING | PPO self-play on Rust env (R-phase MVP, 327K env-steps) |

Frontier of competition: **~1725** (Vadasz). Gap remains huge; this repo
documents what's necessary to close it (fast compiled env + real RL scale).

## The arc, briefly

1. **Hunt phase** — 7 independent ML/search approaches failed to beat rule-based
   v4 (~982 ladder). Concluded v4 is a near-optimal heuristic at the public
   level.
2. **Ensemble breakthrough** — v4 + marco as candidates, "deep exact-sim rollout
   vs v4 opponent" picks between them. Submitted ~1018, drifted ~944 as field
   strengthened.
3. **🔴 Pipeline bug discovered** — verified byte-level + parity-tested + empirical:
   the rollout opponent passed a plain dict to v4, which calls
   `obs.angular_velocity` (attribute) → `AttributeError` → `except: return []`.
   The depth-140/200 adversarial rollout had **never actually run**.
   One `_AttrDict` wrapper fix → **+57 elo on the real ladder**. The fix-vs-buggy
   arena was 73%; vs v4 floor was 90%. See `progress.md`.
4. **P1: π_θ behavior cloning** — distill the bovard top-10% replays (rated
   1394-1847) into a small entity-attention policy (~85K params). Offline,
   parity-tested numpy inference. Baseline-as-opponent FAILED (16.7% local).
   ×4 symmetry augmentation + 90 epochs → tgt_top1 0.82, candidate version
   converged to **1010.4** on ladder.
5. **R-phase: Rust + RL self-play** — porting `fast_sim` to Rust:
   - **R1** Full step port, parity 11,432 steps × 30 episodes **0 divergence**,
     54.8× single-thread.
   - **R2** `EnvPool` with rayon multi-thread, 169k SPS aggregate on 24 threads.
   - **R3** `abi3` wheel for any Python ≥3.10.
   - **R4** PPO self-play (entity-attention actor-critic, shaped reward +
     terminal ±1, GAE clip). First MVP: 327K env-steps. After optimization
     (`observe_batch` + `step_from_samples` in Rust): SPS 400 → **4,400 (10×)**.
     300M env-steps (matching 2nd-place Lux scale) now feasible in ~21h.
   - **R5** Always-safe submit of the PPO bundle.

## What's in here

```
src/
  fast_sim.py             # the verified-perfect Python simulator (reference)
  policy_encode.py        # entity feature encoder (used everywhere, parity-tested)
  policy_dataset.py       # extract (state, action) labels from bovard replays
  policy_train.py         # π_θ BC training (PyTorch)
  pi_theta_infer.py       # pure-numpy inference (parity-tested vs PyTorch)
  ppo_train.py            # PPO self-play training loop on ow_sim.EnvPool
  replay_parser.py        # bovard replay → samples
  audit_counterfactual.py # the rigour that caught the no-op rollout bug
  ensemble_agent.py, ensemble_v2.py, v4_instance.py  # earlier ensemble work
  verify_sim.py           # sanity check on the simulator port

ow_sim/                   # Rust port of fast_sim with PyO3 bindings
  Cargo.toml              # pyo3 0.22 + rayon + ndarray + numpy
  src/lib.rs              # State (full step), EnvPool (rayon batch + observe + samples)
  parity_full.py          # Rust vs Python parity harness (50 eps × 500 steps)
  bench_envpool.py        # SPS throughput benchmark
  observe_parity.py       # feature-encoder parity (Δ ≤ 5.96e-08 = f32 epsilon)

submit/
  build.py                # build the ensemble bundle (v4 + marco + fast_sim + AttrDict fix)
  build_pitheta_cand.py   # build π_θ-candidate or PPO-candidate bundle (extends build.py)
  main_fixed.py           # the bug-fixed ensemble (current 977 baseline)
  main_pitheta_cand.py    # π_θ candidate on v4-rollout (LIVE 1010.4)
  main_ppo_cand.py        # PPO candidate (PENDING)
  ensemble_*.tar.gz       # the actual self-contained submission bundles
  daily.sh                # build + 4P validator gate + always-safe submit

agents/                   # the rule-based agents we use as candidates
  v4_rudra.py             # the "v4" lineage 1224 fork (~982 standalone)
  marcodg_v33.py          # marcodg's notebook (~960 standalone)
  romantamrazov_1224.py   # public 1224-claim notebook (~46% vs v4 local)
  ykhnkf_1100.py          # public 1100-claim notebook (~18% vs v4 local)

competition/unzipped/     # the upstream competition spec (README, agents.md, main.py)
references/               # baseline notebooks for context
progress.md               # the full lab notebook — read this for every decision
plan.md                   # current direction + phase gates
glossary.md               # terms / engine mechanics
```

## Hard-earned lessons (one-liners)

- **"If the env is painfully slow, do not even attempt RL"** is real. Our 11k
  SPS Python sim was OK for offline BC; it would never have trained RL.
  Rust port → 169k SPS aggregate → RL became feasible.
- **Parity-test every reimplementation.** Bit-identical to the reference within
  float epsilon, on many seeds × steps. The same discipline that caught the
  no-op-rollout bug caught two Rust subtle bugs.
- **Local arena ≠ ladder.** π_θ candidate was a local "51.7% tie" → +33 on
  ladder. 4P diverse ladder rewards designs that local 2P arenas miss.
- **One delta at a time + always-safe.** Every submission paired with a known-good
  floor (best-of-2 latest). No regression cost; experimental cost = 1 slot/5.
- **AI cost ≠ AI value.** A 4070 / 4090 / Opus-Code-billing all matter less than:
  one verifiable observation, one parity test, one bug found. The bug-fix
  delivered +57 elo. The 4070 of PPO compute did not yet.

## Setup

```bash
# Python deps
uv venv && uv pip sync   # or: pip install -e . -r requirements.txt
# Rust dep (for ow_sim, needs Rust 1.74+; abi3 wheel works on Python ≥3.10)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cd ow_sim && maturin develop --release
```

## Reproduce

```bash
# Verify the Python simulator vs kaggle_environments
python src/verify_sim.py

# Verify Rust port (full step parity, 30 episodes × 500 steps)
python ow_sim/parity_full.py 30 500

# Verify Rust feature encoder vs Python
python ow_sim/observe_parity.py

# Throughput benchmark
python ow_sim/bench_envpool.py

# Train PPO (uses ow_sim.EnvPool, saves checkpoint to data/ppo_w.npz)
N_ENVS=64 N_STEPS=128 UPDATES=1000 python src/ppo_train.py data/ppo_w.npz

# Daily submit (build PPO bundle + 4P gate + always-safe Kaggle submit)
bash submit/daily.sh
```

## License

Code for educational / competition entry use. No license declared yet
(if you want to reuse, open an issue).
