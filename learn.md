# Orbit Wars — Bài học RL (từ top solutions Lux AI S3 + thực nghiệm của ta)

> Mục đích: KHÔNG QUÊN những gì làm RL self-play THẬT SỰ work. Cập nhật khi học thêm.
> Nguồn gốc: Lux AI Season 3 (NeurIPS 2024) — cùng họ game (multi-agent, self-play,
> map sinh ngẫu nhiên). 1st = Flat Neurons (IMPALA), 2nd = Frog Parade (Isaiah
> Pressman, PPO — **đúng stack của ta**: Rust env + PPO + transformer + wandb + PyO3).
> Link: github.com/IsaiahPressman/kaggle-lux-2024 · github.com/tonykozlovsky/lux-ai3-pub

---

## 🔴 BÀI HỌC #1 — vì sao ta KẸT (và cách thoát)

**Triệu chứng của ta:** 4 recipe self-play (PyTorch material, JAX pure-terminal 4P,
JAX territory 4P/2P) đều **chững ~7% vs v4**. awr self-play tới 0.9 (2P) nhưng arena
v4 vẫn 1/14. ⇒ population tự-nhất-quán mạnh nội bộ nhưng DƯỚI v4.

**CẢ 2 đội top đụng ĐÚNG bức tường này** ("when pure self-play plateaued") và vượt
qua bằng CÙNG bộ kỹ thuật → đây là thứ ta THIẾU:

| # | Kỹ thuật | Ta (trước) | Top teams | Trạng thái |
|---|---|---|---|---|
| 1a | **Frozen Opponent POOL (PFSP)** | chỉ anchor=self gần nhất → overfit | pool nhiều ckpt cũ + best | ⬜ làm |
| 1b | **Teacher-KL loss** KL(student‖best) | không | cả 2 đội dùng → chống quên, ổn định | ⬜ làm |
| 2 | **Value xem CẢ 2 phe** (centralized) | value 1 phe | Frog Parade: softmax 2 value → ổn định | ⬜ làm |
| 3 | **Features + history** | 15 feat, 0 history | 80–100+ feat, **10-frame stack** | ⬜ làm |

→ KHÔNG cần train vs rule-based (v4-in-loop): cả 2 đội thắng bằng PURE self-play +
3 thứ trên. Self-play ĐÚNG CÁCH là đủ vượt v4.

---

## 📐 RECIPE chuẩn (tổng hợp 2 đội)

**Thuật toán:** PPO (Frog Parade) hoặc IMPALA+V-trace (Flat Neurons). PPO đủ tốt.
- GAE-Lambda, **gamma cao 0.9999–1.0** (ta đang 0.999 → nâng).
- Clipping + illegal-action masking + entropy + **teacher-KL**.
- Win/loss reward assign mức PLAYER → sum log-prob mọi đơn vị (ta: sum theo planet ✓).

**Reward (cả 2 hội tụ về):** **sparse ±1 win/loss + một ít partial-point**.
- Flat Neurons: ±1 match + ±2.5 tổng điểm (partial). "Partial scoring **chống do-nothing
  stagnation**" ← đúng bệnh 4P-stuck của ta. → territory(prod) shaping của ta đúng
  hướng, GIỮ ±1 terminal + shaping nhỏ.
- **Early stopping** khi 1 phe chắc thắng (Frog Parade) → tín hiệu sạch hơn.

**Value:** "cheat" xem cả 2 phe (chỉ lúc train, test không cần) → ổn định mạnh.

**Entropy (dynamic):** target-entropy cao→0 tuyến tính (vd Flat Neurons 0.9 move /3.9
sap, giảm qua 100M). Tự chỉnh ent_coef để bám target. Reset target nhỏ hơn khi train
tiếp >100M (dip ngắn rồi mạnh hơn). ← KHÔNG dùng ENT_FLOOR cố định (ta từng kẹt gate
0.5 vì floor). Anneal về ~0 cho gate sắc.

**Opponent (chống plateau):** mix = % latest-self + % **pool ckpt cũ** + % **teacher(best)**.
Teacher-KL giữ kỹ năng cũ; pool chống overfit 1 lối self-play.

**Inference:** Frog Parade SAMPLE (stochastic tốt hơn, tránh bị đoán); Flat Neurons
greedy. Orbit Wars: ta đang sample gate ✓. + data-aug khi infer (flip/rotate, trung
bình policy) — Orbit Wars có đối xứng quay quanh sun(50,50).

**Symmetry:** luôn coi mình là "player 0" — flip/rotate obs+action cho seat khác (cả 2
đội). Tăng sample-efficiency. (Ta có _sym trong policy_encode — tái dùng.)

---

## 🧠 FEATURES — đòn bẩy LỚN nhất (winner nhấn mạnh)

- Flat Neurons: **~1000+ feature/tile** (100 continuous + ~900 one-hot). Bin continuous
  → one-hot. Temporal: "bao nhiêu step từ lần cuối thấy". State-tracking nội bộ.
- Frog Parade: 80 global + ~100 spatial, **history 10 frame** của feature temporal.
  Tách 4 loại: global/spatial × temporal/nontemporal.
- **Inferred features** (suy luận, không có sẵn): vị trí đối xứng (map đối xứng),
  predict vị trí tương lai chướng ngại, v.v.

**Áp cho Orbit Wars (entity-based, không grid):** ngoài 15 feat hiện có, thêm:
- **istinetz**: danger-map (3 planet gần nhất ally/enemy/neutral, +50-80 elo),
  present-value `pv = prod·(γ^arrival − γ^horizon)/(1−γ)`.
- **History**: stack vài frame (vị trí/ships planet + fleet đang tới) → mô hình thấy
  động lực (quỹ đạo quay, fleet đang bay).
- Sun-mask (mask hành động bay vào mặt trời). Pressure (fleet đang tới mỗi planet) —
  ta có rồi nhưng có thể tăng độ phân giải.
- one-hot/bin các discrete (owner, is_orb, comet) thay vì chỉ scalar.

---

## ⚙️ ENGINEERING (Frog Parade — y hệt ta nên dễ học)

- Rust env + Rust feature-eng (≈ ow_sim của ta) → 110k steps/s; GPU là bottleneck →
  giống F2 của ta. JAX-port của ta giải đúng chỗ này (env in-graph).
- Model: 8-block CNN + squeeze-excitation, d=256, **10M params** @ 430 steps/s
  (vs ta 85K — NHỎ; có thể scale lên). 300M game-steps, plateau ~200M, 8 ngày.
- Transformer "học rất nhanh khi imitate teacher, ít param mà ngang CNN" → arch
  transformer của ta OK, nên thử **BC-warmstart từ teacher** rồi RL.
- test-time data-aug (reflections + rotation, average policy).
- bfloat16 + torch.compile (Flat Neurons ~1.5×). JAX ta có jit sẵn.
- **2-model submission trick** (Flat Neurons): nhét model yếu (85%, đã top) + mạnh
  (15%, đang test) trong 1 bài → test thầm trên ladder không lộ chiến lược cho IL copy.

---

## 🧪 KỶ LUẬT (Opus-addendum 1st-place Orbit Wars — đã ghi backlog)
- MỘT delta một lần. Parity test (ta làm với engine ✓). clip_frac creep = cảnh báo
  sớm. EV nên >0.8 (ta đạt 0.7-0.8 ✓). Đừng tin AI tự nhất quán; "run chết chưa" =
  người quyết. **Đừng tin awr self-play** — luôn arena vs đối thủ NGOÀI dòng dõi (v4).

---

## ✅ Checklist áp dụng (cập nhật khi làm xong)
- [x] **1a Opponent pool (PFSP)** trong jax_train (sample anchor từ pool, POOL_EVERY/POOL_MAX)
- [x] **1b Teacher-KL loss** (KL student‖teacher, teacher refresh TEACHER_EVERY) — commit c32124f
- [x] **2 Value xem cả 2 phe → N/A cho Orbit Wars.** Trick này cần vì **Lux có FOG OF
      WAR**; Orbit Wars FULL-OBSERVABILITY → obs per-seat ĐÃ chứa đủ planet đối thủ →
      value đã thấy cả 2 phe sẵn. KHÔNG cần centralized critic. (Bài học: trick này
      chỉ cho game partial-observable.)
- [x] **3 Features mở rộng** (F 15→21): nearest-enemy/neutral/mine dist, threat,
      takeable, net_pressure — parity jax↔numpy **1.2e-7**. (History 10-frame: HOÃN —
      cần plumbing history buffer trong env scan; relational features đã bắt phần lớn
      cấu trúc tĩnh. Thêm history nếu cần lift thêm.)
- [x] **gamma 0.999 → 0.9997** (env GAMMA)
- [ ] (sau) 2-model submission trick khi có model mạnh thật
- [ ] (nếu cần) features history 10-frame + danger-map 3-nearest + present-value đầy đủ

> Xem thêm: `backlog.md` (findings + forum intel), `jax_port.md` (JAX infra),
> `progress.md` (nhật ký). Gate THẬT = `arena_vs_v4.py` (≥0.5 mới submit).
