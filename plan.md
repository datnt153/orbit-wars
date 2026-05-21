# Orbit Wars — Plan

> Kế hoạch hiện tại + next steps. Cập nhật khi đổi hướng.

## Mục tiêu
Vượt LB **978** (rule-base v4 hiện tại). Top public ~1000+.

## Chiến lược submission (BẮT BUỘC)
LB chỉ tính 2 latest, show best of 2, quota 5/day. Quy tắc **always-safe**:
1. Trước khi thử experimental: đảm bảo 2 slots latest = v4 safe (~978). Nếu chưa → submit v4 thêm để đẩy sub yếu ra.
2. Submit experimental C → slot {C, v4}. LB = max(C, 978). Worst case không tụt.
3. Đánh giá C:
   - C ≥ 978 → giữ, C thành baseline mới
   - C < 978 → submit v4 ngay để recovery

## ⮕ HƯỚNG HIỆN TẠI (2026-05-18): Value net + opponent model trong search

**Vấn đề:** top ~1018 đuối xuống <1k. **Root cause (có trong code, KHÔNG phải noise):**
- `ensemble_agent.py:65-82` `_score_for` = leaf eval thô `Σ(ship+8·prod)` diff.
- `ensemble_agent.py:88-103` rollout opponent = v4 thuần (~982).
- **Bằng chứng cứng:** manifest bovard → field top tháng 4-5 rated **1490–1630**. Search tối ưu chống bóng ma yếu hơn thực tế ~500–650 điểm → đuối là tất yếu.

**Vì sao value-learning work ở chỗ BC chết:** BC trước thay policy qua action-abstraction 8-cand (mất precision, 0% vs v4). Value = regression (state→outcome), KHÔNG qua abstraction, cắm vào search đã chạy được. Search vẫn sinh nước bằng v4/marco.

- **Phase 0c (CPU, $0):** counterfactual replay audit — định lượng failure mode trước khi tiêu slot/GPU.
- **Phase 1 (GPU phút):** V_φ(state)→outcome train trên replay top-10% (label = final reward, ground truth incl. comet). Thay `_score_for`. MỘT delta, always-safe, chờ ≥1 ngày.
- **Phase 2:** π_θ distill từ người rated cao → đối thủ rollout nhanh (thay v4-only). Worst-case rổ {v4, π_θ}. MỘT delta tách biệt.
- Data: `data/bovard/` (series bovard/orbit-wars-top10-episodes). Parser: `src/replay_parser.py`.
- Guardrails (từ discussion Light+Opus): một delta một lần; KHÔNG tin arena 16-40g/peak; quyết định "run chết chưa/kiến trúc" là của user.

**STATUS 2026-05-18:** Phase 0 XONG. Phát hiện+fix bug no-op rollout (verified). Gate: fixed vs buggy 73%, vs v4 90%, 4P validator-safe. **ĐÃ SUBMIT `ensemble_fixed.tar.gz`** (08:00 UTC, PENDING; floor=ensemble_E 944.6 = LIVE thật, KHÔNG phải "1018"). CHỜ ≥1 ngày (~2026-05-19) hội tụ mới đánh giá — không phản ứng theo peak.

**STATUS 2026-05-20 — PIVOT R-Phase:** ensemble_fixed hội tụ **~1004** LIVE (xác nhận bug-fix). P1 (π_θ) đã thử cả 2 fork — A (candidate) 51.7% hòa, B (opponent) 25% thua — empirically NULL, đã submit Fork A always-safe (zero downside). User quyết **dừng Nemotron, dồn toàn lực Rust+RL self-play** (đường thật theo 2nd-place Lux: fast compiled env → RL on 4090).

## ⮕ R-Phase: Rust + RL self-play (commit 2026-05-20)

**Why:** Exact-sim+heuristic ceiling ~1004; frontier 1725. BC-clone trần ≈ field. Đường duy nhất tới 1200+ (per 2nd-place Lux modest-compute thắng) = fast compiled env (Rust ~80-100k SPS expected trên 13700K 24t) → PPO/IMPALA self-play trên 4090. ~3-4 tuần, go/no-go ở mỗi pha.

- **R1 (1 tuần)** — Rust-port `fast_sim` với parity harness vs Python (test-driven, 2nd-place template). Maturin/PyO3 cho Python bindings. **Go/no-go:** 100% parity trên ≥1000 seeds × 500 steps + SPS ≥50k single-thread; nếu fail → bail sang fallback (V_φ leaf eval, P2).
- **R2 (3 ngày)** — Vectorized batch step (N env trong 1 Rust call, multithread). **Go/no-go:** ≥80k SPS aggregate trên 24 threads.
- **R3 (1 ngày)** — Build wheel, ship sang server-ai (4090), test inference parity + tốc độ.
- **R4 (~8 ngày)** — PPO/IMPALA self-play training trên 4090. Entity-attention policy (per-planet head, tái dùng kiến trúc π_θ). Sparse +1/-1 reward + dynamic entropy schedule (Flat Neurons playbook). League/frozen opponent pool. Periodic checkpoint arena vs `ensemble_fixed` 1004. **Go/no-go:** sau ~100M steps, arena vs `ensemble_fixed` ≥55% → tiếp; nếu plateau <50% → bail.
- **R5** — Always-safe submit khi vượt floor 1004. `ensemble_fixed` LIVE trong toàn pha = floor không tụt.

## D-Phase: Daily 10M cadence (commit 2026-05-21)

Mục tiêu: cumulative 300M env-steps (~scale 2nd-place Lux) chia thành 10M/ngày × 30 ngày tới deadline 23/06.

- **Train**: persistent nohup PPO, resume mỗi ngày từ ckpt `data/ppo_w.npz`. Save mỗi 20 updates → ckpt luôn fresh. Hiện đang chạy lần đầu (10M, PID launched 2026-05-21 ~15:00, ETA ~11h).
- **Submit sáng**: `bash submit/daily.sh` — build PPO-cand bundle từ ckpt mới nhất, 4P validator gate, submit always-safe.
- **Tối assess**: check `kaggle competitions submissions orbit-wars` sau ≥1 ngày hội tụ.
- **Resume tối**: training tiếp 10M nữa qua đêm (nếu PID đã chết, relaunch với `RESUME=data/ppo_w.npz`).
- **Quota**: 5/ngày — chỉ submit 1 (dành slot cho safe-refresh nếu cần).
- **Floor maintenance**: nếu LB tụt do prior submission rơi khỏi 2-latest, submit lại bản tốt nhất (vd `submit/ensemble_pitheta_cand.tar.gz` = 1010.4) làm safe-refresh.
- **Speed optimization DONE 2026-05-21:** port `encode_state` + `step_from_samples` sang Rust (`EnvPool.observe_batch` + `step_from_samples`). Parity Δ≤6e-8 (f32 epsilon, bit-equivalent). Encode 29-131× nhanh hơn Python. Tổng SPS 400 → 4000 = **10×**. 300M cumulative giờ feasible ~21h (thay vì 14 ngày). 10M = ~40 phút. Slow overnight (PID 3032007 @ 6.06M) killed; fast run (PID 3191484) resumed từ ckpt, target +50M new steps. Bottleneck giờ là PPO update + sample bernoulli/multinomial trên GPU (chứ không còn rollout). Optimization tiếp có thể: vectorize/JIT log_prob_and_entropy, tăng mb, hoặc compile net với torch.compile.

Env: dev+parity local (i7-13700K 24t, Rust 1.72 cài sẵn — có thể rustup update). Training 4090 server-ai. P2 V_φ giữ làm fallback nếu R-phase bail.

## (Cũ — superseded) DAGGER distill v4 → NN

### Phase 1 — Behavior Cloning baseline
- [ ] Data collection: chơi N games với v4 (vs v4 / random / mixed), record:
  - `encode_turn(obs)` features (self/candidate/global) per source planet per turn
  - Label = candidate_index mà v4's move khớp (match angle gần nhất với target_angles); no-launch → idx 0
- [ ] Train `PlanetPolicy` (architecture có sẵn) bằng cross-entropy
- [ ] Arena BC-NN vs v4 → baseline win rate

### Phase 2 — DAGGER iteration
- [ ] Run BC-NN trong env, collect states nó visit
- [ ] Query v4 cho các states đó → correct label
- [ ] Aggregate dataset + retrain
- [ ] Lặp 2-3 vòng, arena mỗi vòng

### Phase 3 — Submit
- [ ] Build NN inference main.py (torch, có `__file__` guard)
- [ ] Test 4P file mode (`env.run(['main.py']*4)`) — không crash, không vượt 1s/turn
- [ ] Always-safe submit + theo dõi score

## Trạng thái (2026-05-18) — MỤC TIÊU 1200 elo

Chốt hướng Value net + opponent model (xem section ⮕ trên). DAGGER bỏ.

## Trạng thái (2026-05-16) — MỤC TIÊU 1200 elo

**BEST THẬT = ensemble_v1 (D140) ~1018** (đã hội tụ, ổn định). Slot1=E(D200) 987, slot2=v1 1018 → LB 1018.

**Phương pháp (BẮT BUỘC, xem [[feedback-kaggle-ladder-eval]]):**
- Score ladder dao động mạnh lúc ít games. KHÔNG đánh giá qua peak hay arena 16-40 games.
- Submit experimental → CHỜ ≥1 NGÀY hội tụ rồi mới so baseline.
- Always-safe: luôn 1 slot = baseline ổn định (v1 ~1018).
- Ưu tiên cải tiến CHẤT, không tune param qua noise.

**Hướng tiếp (triển vọng nhất):** thay heuristic eval rollout (`ship+8·prod`) bằng marcodg `chain_val` potential → eval chính xác hơn. Build cẩn thận, submit, CHỜ ≥1 ngày đánh giá.

**Đã chốt:** pool {v4,marco}+opp v4 tối ưu; depth tuning qua noise vô nghĩa (D200 peak 1044→987 < v1 1018).

## Code đã persist vào repo (KHÔNG còn ở /tmp)
- `src/` — fast_sim.py (verified perfect 768 steps), verify_sim.py, v4_instance.py, ensemble_agent.py (=v1 D140, BEST ~1018), ensemble_v2.py (env-param: OWE_AGENTS/OPPS/DEPTH/MARGIN)
- `agents/` — v4_rudra.py, marcodg_v33.py, romantamrazov_1224.py, ykhnkf_1100.py
- `submit/` — build.py (assemble base64 bundle), main.py (bundle), ensemble_v1_D140.tar.gz, ensemble_E_D200.tar.gz

⚠ **HARDCODED PATHS cần sửa khi reuse:** src/*.py trỏ `/tmp/ow_train/rule-based submission.py`, `/tmp/hunt/marcodg.../main.py`, `/tmp/ow_sim`. Sau clear/reboot /tmp mất → đổi sang `~/kaggle/orbit-wars/agents/v4_rudra.py`, `agents/marcodg_v33.py`, `src/`. Riêng submit/main.py + build.py: build.py đọc từ /tmp → trước rebuild phải point lại vào agents/. Bản .tar.gz trong submit/ đã self-contained (base64) — submit lại trực tiếp được, không cần rebuild.

## Reproduce nhanh (session sau)
1. Verify sim: `cd ~/kaggle/orbit-wars && .venv/bin/python -c "import sys;sys.path.insert(0,'src');import verify_sim"` (sau khi sửa path)
2. Submit lại baseline an toàn: `kaggle competitions submit -c orbit-wars -f submit/ensemble_v1_D140.tar.gz -m "..."` (đã self-contained)
3. Arena: `.venv/bin/python arena.py <A.py> <B.py> --games 100 --players 2 --alternate` (cần ≥100 games mới tin)

## Backlog (nếu DAGGER không vượt 978)
- H1: v4 + GBC value filter (v4 moves → sim đánh giá → drop move yếu)
- H2: Search với candidate từ v4 thay heuristic
- Tải thêm replays Meta-Kaggle (BigQuery, hàng nghìn) → BC từ top players thật thay vì chỉ v4

## Lưu ý kỹ thuật
- Test PHẢI file mode + 4P (Kaggle exec loader không có `__file__`, `__name__`)
- Đọc replay/logs qua API trực tiếp (Kaggle CLI v1.x bug `content-length`):
  `curl -u user:key https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay -d '{"episodeId":N}'`
- venv: `/home/dat/kaggle/orbit-wars/.venv` (torch 2.11 cu128, CUDA OK)
