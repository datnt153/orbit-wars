# Orbit Wars — JAX Port Plan

> Quyết định 2026-05-22 (user): port engine + training sang **JAX** để bỏ
> bottleneck F2 (round-trip GPU tuần tự) và mở đường TPU/scale lớn.
> Branch `jax-port`. Master + run PyTorch giữ nguyên làm **baseline an toàn**.

## Nguyên tắc (theo Opus-addendum của 1st place)
- **Parity-first, MỘT phase một lần.** Mỗi phase port xong phải khớp `fast_sim`
  (trọng tài vàng, đã byte-identical vs Kaggle env) trước khi sang phase sau.
- **Không big-bang.** Không ship 7 thay đổi rồi vỡ không biết cái nào.
- **Không vứt cái đang work.** PyTorch+Rust vẫn chạy/submit tới khi JAX vượt qua.
- Quyết định lớn ("đổi hướng/bỏ phase") = user chốt.

## Tại sao JAX (nhắc lại lý do, tránh quên)
F2: env (CPU) ↔ policy (GPU) round-trip mỗi step do Python điều phối → 4k SPS dù
Rust env chạy 132k. JAX biên dịch **cả env+policy+sample thành 1 graph XLA** trên
device (`lax.scan`+`vmap`) → bỏ Python khỏi loop → ~10k SPS (như winner).
**Bắt buộc env phải JAX-native** (env Rust ở CPU không nhét vào graph được).

## State representation (cốt lõi)
List Python → mảng fixed-shape + mask:
- planets `[P=48]`: id, owner, x, y, rad, ships, prod, valid + init_x, init_y (rotation).
- fleets `[F=512]`: id, owner, x, y, angle, from_id, ships, valid. (engine không cap;
  512 dư; fleet sống vài chục turn.)
- scalars: next_fleet_id, av, step, ship_speed, episode_steps.
- **comets: HOÃN** (paths nested) — bắt đầu planets+fleets+rotation+combat, parity
  trên state không-comet; thêm comet (padded paths) ở phase sau. Khớp `fast_sim`
  vốn cũng SKIP comet spawning.
- **float32** cho JAX env (train); chính xác tuyệt đối khi submit vẫn dùng
  fast_sim/numpy. Parity tolerance ~1e-3 over rollout (đủ cho RL).

## Lộ trình (gate parity từng bước)
- [x] **S0** branch + plan + state pytree + converters dict↔jax (round-trip test) + production
- [ ] **S1** phase 1 fleet launch (slot allocation theo mask, next_fleet_id)
- [ ] **S2** phase 3 fleet movement + collision (sun seg-dist, bounds, planet hit)
- [ ] **S3** phase 4 planet rotation + sweep (chỉ khi r+rad < 50)
- [ ] **S4** phase 6 combat resolution (group theo player, top-2, tie → -1)
- [ ] **S5** phase 7 termination + scoring (±1, alive≤1, step≥ep-2)
- [ ] **S6** comets (advance theo padded paths; spawn skip)
- [ ] **S7** full `step()` parity vs fast_sim: N=1000 state ngẫu nhiên + rollout 50 bước, divergence=0 (tol)
- [ ] **S8** `vmap` batch + `lax.scan` rollout → đo SPS (mục tiêu ≥10k, hết F2)
- [ ] **S9** policy entity-transformer trong JAX (Flax/Equinox), parity forward vs PyTorch
- [ ] **S10** PPO loop JAX (jitted): port recipe anchor+league+anneal; log EV/clip_frac/awr
- [ ] **S11** export weights → numpy bundle (parity inference) → arena vs PyTorch-best → submit khi vượt

## Go/no-go
- Nếu tới **S8** SPS không vượt PyTorch đáng kể → dừng, quay lại PyTorch (sunk cost).
- Mỗi phase parity fail → fix CHỖ ĐÓ, không chồng thay đổi.

## Files
- `src/jax_env.py` — env JAX (đang xây).
- `src/test_jax_parity.py` — parity vs fast_sim.
- `src/fast_sim.py` — TRỌNG TÀI (không sửa).
