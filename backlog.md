# Orbit Wars — Backlog & Findings

> Kho thông tin tìm được + cải tiến đã làm + việc cần làm (ưu tiên).
> Bổ sung cho `progress.md` (nhật ký) và `plan.md` (kế hoạch). Ngày = ICT.
> Cập nhật gần nhất: **2026-05-22**.

---

## 📊 Trạng thái hiện tại (2026-05-22)

- **PPO self-play đang chạy** trên hg-rtx4090 (PID 3464038, config C).
  Resume từ `ppo_best` 17.15M, anchor=ppo_best, league promotion.
  **wandb: https://wandb.ai/kingkong153/orbit-wars** (run `ppo_20260522_1054`).
- **Kết quả raw-policy (arena, đa map, swap ghế):**
  - best 59M **vs floor 15.7M = 0.983** (nghiền baseline đã submit).
  - best 59M **vs best 19M (3h trước) = 0.945** (vẫn đang leo nhanh).
  - Generalize: **0.979 map CHƯA thấy ≈ 0.984 map đã train** → KHÔNG overfit map.
- **LIVE Kaggle:** pitheta_cand ~996 (chưa thay). PPO **chưa** test transfer sang ladder.
- **SPS ~4100** (config C). 295M ETA ~20h.

---

## 🔬 Findings — từ thực nghiệm của ta

### F1. Pure self-play KHÔNG anchor → policy trôi về random (đã fix)
- Run 300M đầu (resume 15.7M, pure self-play) **làm policy TỆ ĐI**: 118M thua
  15.7M ở arena **0.350**. Entropy nổ **7.4→27** rồi đứng yên.
- Gốc rễ: self-play zero-sum → advantage ~0 → `-ent_coef·H` thống trị → gate
  về 50/50 (~random). + resume KHÔNG khôi phục Adam state.
- **`wr` self-play (~0.5) VÔ NGHĨA** — luôn 0.5 vì đấu bản sao. Phải đo `awr`
  (vs frozen anchor) hoặc arena vs frozen baseline.

### F2. Bottleneck training KHÔNG phải CPU
- `bench_env.py`: Rust env chạy **65k–132k SPS** ở RAYON=80. Training chỉ 2560.
- Nút thắt = **round-trip GPU tuần tự mỗi step** do Xeon single-thread chậm điều
  khiển ~chục kernel + sync `multinomial`. Net 85K bé → GPU compute không đáng kể.
- Đổ thêm CPU vô ích. N_ENVS lớn khấu hao một phần (saturate ~3600-4100 SPS).

### F3. `n_steps` quá ngắn hại learning
- Config D (n_steps=32) nhanh nhất nhưng live learner lùi **0.161 vs best**.
- n_steps≥64 (validate) ok. Horizon 32 quá ngắn cho game 500-turn, gamma 0.999.
- **Resume phải từ `ppo_best` (BEST), KHÔNG phải `ppo_w` (latest live, có thể đã drift).**

### F4. Entity-transformer generalize map tốt
- Held-out test: 0.979 (map mới) ≈ 0.984 (map cũ) → **128 map là đủ**, không cần
  thêm. Lý do: transformer xử lý theo thực thể, không theo tọa độ tuyệt đối.

### F5. Reward shaping hiện tại (phân tích)
- `(Φ(s')−Φ(s))×0.01` với `Φ = material_mình − avg(material_đối_thủ)`,
  `material = Σ(ships + prod×8)`; + terminal ±1.
- Là **potential-based shaping** (Ng et al. 1999) → không đổi optimal policy, chỉ
  tăng tốc. Hiệu quả (98% trong 3h).
- **Lệch:** dày cộng dồn ~±2.0 > terminal ±1 → policy bị thúc ~2/3 vì material,
  ~1/3 vì thắng. `prod×8` là heuristic chưa tune. Là nguồn value loss cao.

---

## 🏆 Findings — tình báo từ forum đối thủ

### 1st place (Lin Myat Ko + Opus) — "Sharing our RL lessons"
- **Engine nhanh là bắt buộc.** Họ port sang **JAX end-to-end**, ~10k SPS (gồm cả
  forward + PPO update). Feature/arch phức tạp → SPS tụt 2k và **ngừng học**.
- **Architecture: entity transformer.** Action head chuẩn (target argmax khi
  infer / softmax khi train; fleet % theo bins). Học từ Lux AI winning solutions.
- **"+1/−1 là ĐỦ cho 2P"** — họ dùng terminal thuần, KHÔNG shaped. (Budget 600M
  steps nên kham được học chậm.)
- **Feature engineering = nhồi inductive bias càng nhiều càng tốt** (họ không
  viết heuristic agent nhưng phân tích game để thu hẹp search space).
- Budget: Claude $100 + Vast.ai 5090 ~$150. Model tốt nhất (F12) = **600M steps,
  ~3 ngày**, self-play from scratch.
- **explained_variance:** nên đạt 0.8 trong 100 iter (run sớm của họ: 0.9 trong
  20 iter). **EV không vượt 0.5 ⇒ nghi obs representation / architecture.**

### Opus addendum (cảnh báo cách dùng AI cho RL) — RẤT đáng đọc
- **Một architecture delta một lần. LUÔN.** Ship 7 thay đổi/2 ngày → vỡ không
  biết cái nào. "7 wins 1 loss" không work trong RL.
- **Baseline "ngu" mà train được > baseline "thông minh" mà không train.** Hạn
  chế của baseline đang làm regularization ngầm.
- **`clip_frac` là cảnh báo SỚM tin cậy nhất:** creep 0.10→0.30+ TRƯỚC khi
  entropy sụp / KL nổ. Thấy creep → cắt lr hoặc revert capacity, đừng chờ nổ.
- **Đừng tin AI tự nhất quán:** Opus "đổi kết luận trên cùng data nhiều lần/giờ".
  AI tốt cho: port code, parity test, script phân tích, refactor. Dở ở: tách
  "research thú vị" vs "việc dự án cần", biết khi nào dừng, nhớ correction hôm qua.
  **Quyết định architecture + "run chết chưa" thuộc về người cầm budget.**
- **Playbook transformer-PPO:** warmup_cosine, lr decay, entropy schedule,
  per-head ent_coef (IMPALA / AlphaStar / OpenAI Five). Đọc TRƯỚC khi sáng tạo.

### istinetz (#40) — heuristic đánh giá mục tiêu
- **Present-value của planet:** `pv = prod·(γ^arrival − γ^horizon)/(1−γ)`,
  γ=0.99 (chiết khấu thời gian bay + game kết thúc turn 500). "Chỉ thế + quỹ đạo
  → đạt ~1000 LB."
- **Danger map:** hardcode gamma theo 3 planet gần nhất (ally/enemy/neutral) →
  **+50-80 elo**, BẤT NGỜ tốt. Gradient-based (`Σ ships/dist·owner`) thì TỆ.
  `ally/(ally+enemy)` hứa hẹn nhưng chưa tune xong.
- Bài học: feature engineered tay đơn giản đôi khi >> "thông minh".

---

## ✅ Cải tiến đã làm (2026-05-22)

| # | Cải tiến | Kết quả | Commit |
|---|---|---|---|
| 1 | **Robust recipe**: anchor opponent + league promotion + entropy anneal + Adam save/restore + multi-map | sửa drift; validate +0.578 vs 15.7M | 9215602 |
| 2 | **Vectorize rollout** (bỏ 256 GPU scalar-write/step) + fix bug O(n_envs²) `diff_vs_avg_opp` | +6% (Xeon), value loss ổn hơn | b3ebbc3 |
| 3 | **Config C** (512×64) + resume từ ppo_best | +38% SPS, học đúng | a38825d |
| 4 | **wandb** auto-detect qua netrc | live tracking | b3ebbc3 |
| 5 | **Dọn git**: untrack data/*.npz, ignore data/+wandb/ | hết kẹt pull | 6b46939 |
| 6 | **Diagnostics**: log explained_variance + clip_frac + MAP_OFFSET test | observability theo chuẩn winner | aecc14e |
| 7 | **`src/arena_raw.py`**: ow_sim head-to-head nhanh, đa map, swap ghế | công cụ verify ground-truth | b3ebbc3 |
| 8 | **`src/bench_env.py`**: profiler throughput | tìm ra bottleneck | b3ebbc3 |

---

## 📋 BACKLOG (ưu tiên) — quyết định lớn thuộc về user

> Theo Opus-addendum: **architecture-level + "run chết chưa" là của người cầm
> budget**. Tôi prototype/đo khi được chọn, một delta một lần, arena verify.

### P0 — Kiểm chứng transfer (TRƯỚC khi đụng gì khác)
- [ ] 98% self-play **CHƯA chắc = elo ladder**. Build bundle từ `ppo_best` →
      arena vs **v4 + marco + LIVE pitheta_cand** (đối thủ đa dạng, KHÁC dòng dõi).
- [ ] Nếu transfer tốt → submit always-safe (slot {ppo, floor}), chờ ≥1 ngày hội tụ.
- [ ] Nếu transfer kém → policy giòn vs đối thủ lạ → mới sang P1/P2.

### P1 — Chất lượng value (nếu EV thấp / training chững)
- [ ] Đo `explained_variance` ở lần launch tới. <0.5 ⇒ nghi obs/arch (theo winner).
- [ ] Nếu cần: **value clipping** (PPO value clip) — gọn value loss.
- [ ] Cân nhắc hạ `shape_scale` 0.01→0.003 (tín hiệu "thắng" thuần hơn, theo
      winner "+1/−1 đủ cho 2P") hoặc bỏ shaped hẳn nếu budget steps đủ lớn.

### P2 — Feature engineering (inductive bias, winner nhấn mạnh)
- [ ] Thêm feature engineered vào input: **present-value planet** (istinetz),
      **danger map** (3 planet gần nhất). Cần parity test.
- [ ] Sun-mask (Opus-addendum nhắc): mask hành động bay vào mặt trời.

### P3 — Robustness training (nếu instability quay lại)
- [ ] Theo dõi `clip_frac` creep (0.10→0.30+) = cảnh báo sớm → cắt lr/revert.
- [ ] Playbook transformer-PPO: **lr warmup + cosine decay**, entropy schedule,
      có thể per-head ent_coef (IMPALA/AlphaStar/OpenAI Five).

### P4 — Coverage
- [ ] Train cả **4P** (hiện chỉ 2P) + arena 4P vs v4/marco.
- [ ] (Nếu cần scale lớn) cân nhắc port JAX end-to-end như winner (~10k SPS) —
      nhưng chi phí lớn, chỉ khi PyTorch+Rust chạm trần.

---

## ⚠️ Rủi ro / câu hỏi mở
- **Self-play overfit đối thủ (không phải map):** mạnh trong dòng dõi, giòn vs
  v4/team khác? → P0 trả lời.
- **Value loss cao (vl~60):** chưa đo EV → chưa biết là benign (do reward scale +
  anchor động) hay triệu chứng obs/arch. Advantage được normalize nên CHƯA cản
  học (98% chứng minh), nhưng có thể giới hạn trần.
- **2P-only:** game có 4P, policy chưa thấy động lực 4P.
- **AI self-consistency:** tôi đã kill run / đổi config vài lần phiên này — bám
  bằng chứng (arena trước khi tin) nhưng quyết định lớn cần user chốt.

---

## 🛠 Lệnh / artifact tham chiếu
```bash
# Train remote (workflow: code → commit → push → pull → run)
ssh datnt114@hg-rtx4090 'cd ~/orbit-wars && git pull origin master && bash run_remote_train.sh'

# Arena raw-policy (verify ground-truth)
.venv/bin/python src/arena_raw.py A.npz B.npz [per_side] [n_envs]
MAP_OFFSET=2000 ... # test trên map chưa thấy

# Profile throughput
RAYON_NUM_THREADS=80 .venv/bin/python src/bench_env.py

# Build bundle từ checkpoint + submit always-safe
.venv/bin/python submit/build_pitheta_cand.py data/ppo_best.npz main_ppo.py
bash submit/daily.sh
```
- Checkpoints: `data/ppo_best.npz` (BEST, league-promoted) · `data/ppo_w.npz` (live).
- wandb metrics chính: `win_rate_vs_anchor` (tiến bộ thật), `anchor_promoted`,
  `explained_variance`, `clip_frac` (cảnh báo sớm), `value_loss`, `kl`, `entropy`.
