# Orbit Wars — Progress Log

> Nhật ký thực nghiệm. Cập nhật sau mỗi mốc. Ngày theo ICT.

## Bối cảnh cuộc thi
- Kaggle Orbit Wars: RTS không gian 2/4 player, 500 turns, sun ở (50,50) r=10
- Deadline 2026-06-23, reward $50K, ~2650 teams
- Submit: tar.gz có `main.py` (hàm `agent`) ở root + optional weights
- **Rule LB:** chỉ track **2 submission mới nhất**, hiển thị **best of 2**. Quota **5/day**
- Validation episode = 4 bản agent tự đánh nhau; fail → ERROR
- actTimeout = 1s/turn; runTimeout = 1200s/game

## Submissions & Scores

| Ngày | Approach | LB Score | Ghi chú |
|---|---|---|---|
| 2026-05-14 | Rule-base v4 (thisisn0mad fallback) | **978.8** | Validator weights mismatch → fallback v4. Mạnh nhất hiện tại |
| 2026-05-14 | Search+Value GBC (aidensong123) lần 1 | ERROR | Bug `__file__` NameError ở Kaggle exec loader |
| 2026-05-14 | Search+Value GBC — đã fix `__file__` | **904.1** | Yếu hơn v4 dù tác giả claim "LB 1000+" |
| 2026-05-15 | Rule-base v4 backup ×2 | (pending) | Chiếm 2 slots latest bảo vệ LB ~978 |

## Findings cốt lõi

1. **Rule-base v4 (~978-982) là agent MẠNH NHẤT.** Mọi approach thay thế đều thua:
   - Search+Value GBC: 904 < 978
   - MCTS v0 (lookahead 30, topK 12, budget 0.85s): **20% win vs v4** (40 games)
   - PPO v2 (reward shaping + curriculum, 2000 updates): reward -2.36 → collapse
   - **BC distill v4 → NN: 0/40 vs v4 (0%)**, val_acc chỉ 53%
2. **Bottleneck = feature/action representation, KHÔNG phải search depth hay training.** encode_turn (8 candidates + self/global) không capture được logic v4 (reinforcement, sun collision, multi-source coord, comet, ship-count tùy biến). Action abstraction (8 discrete + fixed ships) mất precision.
3. **PPO from scratch luôn collapse vs strong opponent** — đúng như tác giả notebook ghi (5 lineage PPO/SFT thất bại).
4. MCTS v0 chỉ dùng ~4ms/turn (budget 1s) → lãng phí compute, nhưng tăng search vẫn thua vì candidate yếu.
5. **v4-PV (present value scoring thay linear): 13.8% vs v4 gốc** — TỆ HƠN. Writeup "1000 với present value" là từ scratch, không phải lên v4 đã tuned.
6. **KẾT LUẬN: v4 (lineage 1224, ~982) đã ở local optimum rất tốt. Cả thay thế (ML/search) lẫn sửa scoring đều làm tệ đi** (4 approaches độc lập confirm). Hướng khả thi duy nhất: tìm agent public MẠNH HƠN v4 (notebook romantamrazov claim LB max 1224 — v4 ta chỉ là nhánh 982).
6. v4 score noisy trên ladder: cùng agent dao động 955↔982 (μ₀=600 điều chỉnh dần). Cần nhiều episodes mới ổn định → đánh giá experimental phải tính noise này.

## Hạ tầng đã build

- `arena.py` — chơi N games song song giữa 2 file agents, win rate (2P/4P, alternate sides). ~1.7 games/s với 8 workers
- `/tmp/ow_train/` — PPO pipeline (reward shaping ở `src/env.py`, curriculum ở `src/opponents.py`)
- `/tmp/ow_train/build_ppo_submission.py` — build PPO submission từ checkpoint (torch inference)
- `/tmp/mcts-build/` — MCTS v0 (Base B + tuned hyperparams)
- Datasets tải sẵn: `aidensong123/orbit-wars-value` (GBC trees), rule-base v4 source

## Hunt agent public mạnh nhất (2026-05-15)

Pull candidates (claimed score trong title, KHÔNG tin — arena thực tế vs v4 982):
- rahulchauhan016 "2000.4" — MCTS+BeamSearch+Neural, cần execute notebook (inspect.getsource) để extract
- romantamrazov "LB max 1224" — submission.py 113KB ✓ extracted
- yijue1 "1103" — submission.py 84KB ✓
- ykhnkf "1100" — submission.py 106KB ✓
- marcodg "1060.5" — .py 104KB ✓
- zacharymaronek ">1000" — .py 23KB ✓
- takmada "1000+" — = v4 thuần (giống rule-based submission.py), SKIP
- Arena round-robin vs v4 (982), 60 games/cái — KẾT QUẢ:
  - romantamrazov 1224: **11.7%** win vs v4 (thua nặng)
  - ykhnkf 1100: **18.3%** (thua nặng)
  - marcodg 1060.5: 60g→46.7%, **200g→46.0%** (92-108, xác nhận hơi yếu hơn v4, không phải noise)
  - zacharymaronek >1000: 36.7% (thua)
  - yijue1 1103: notebook bị sabotage (nhiều syntax error cố ý) — SKIP
  - rahulchauhan016 2000.4: notebook execute fail (50 cells phức tạp, inspect.getsource) — SKIP, cost cao
  - **KẾT LUẬN HUNT: không agent public nào ≥ v4 982. v4 = near-optimal public agent.**
- **FINDING: v4 ta (982) đánh bại MỌI public notebook claim điểm cao hơn.** Claimed scores là PEAK lịch sử lúc LB ít competition, không phản ánh head-to-head. v4 = Rudra9439 fork đã cải tiến trên lineage → mạnh hơn cả bản gốc 1224/1100.

## Engine mechanics (đọc từ source `.venv/.../envs/orbit_wars/orbit_wars.py`)

Turn order: launch → **production (+prod planet owned)** → fleet move+collision (oob/sun/planet) → **rotation+sweep** → combat.

- **Combat:** gom fleet theo owner; top vs 2nd `survivor=top−2nd`; tie 2 lớn nhất → cả 2 = 0. Survivor vs garrison: cùng owner→cộng; khác→`garrison−=survivor`, `<0`→flip owner ships=abs(dư). Solo chiếm cần `ships ≥ garrison+1`.
- **Production trước movement** → planet vừa chiếm sản xuất ngay turn kế.
- **Sweep:** planet/comet xoay quét trúng fleet → kéo vào combat (không chỉ fleet bay vào).
- **Speed:** `1+5·(ln(ships)/ln1000)^1.5` cap 6; 1 ship=1/turn (nhỏ=chậm).
- **Score:** Σships planet owned + Σships fleet; mọi player score==max(>0) đều +1 (4P nhiều người cùng thắng được).
- **Map:** 4-fold symmetry, 5-10 groups×4, ≥3 static ≥1 orbiting (diagonal 4P). Home 10 ships. radius=1+ln(prod), prod∈1..5. Orbit nếu `orbital_r+radius<50`, angular_vel∈[0.025,0.05].
- **Comet:** spawn step∈{50,150,250,350,450}, ships=min(4×rand(1,99)) rất thấp, prod=1, hết path→mất ships.
- **Sun:** point-segment dist tới (50,50) < 10 → fleet hủy (continuous cả path).
- **Terminate:** step≥498 hoặc alive_players≤1.

→ Cơ hội: build **fast exact simulator** (port `interpreter()`, bỏ kaggle_env overhead) cho arena nhanh + MCTS forward-sim chính xác (Base B chỉ đoán).

### ✅ fast_sim DONE + VERIFIED (2026-05-15)
- `/tmp/ow_sim/fast_sim.py` — port interpreter() game loop. `new_state(obs)`, `clone()`, `step(state, joint_actions, num_agents)`
- Verify `/tmp/ow_sim/verify_sim.py`: **768 steps × 16 games (seed 0-7, 2P+4P) = 0 mismatches, PERFECT MATCH**
- Limitation đã biết: KHÔNG spawn comet (random ship count không predict được từ obs). Chính xác 100% tới spawn step (50/150/250/350/450), sau đó approx (không có comet mới). OK cho MCTS lookahead ngắn.
- Tốc độ: **~11K steps/s, 91µs/step** (Python thuần). Budget 1s/turn → ~360 rollouts depth-30/turn.
- → Đây là lợi thế Base B KHÔNG có (sim đoán → 904). MCTS + sim chính xác là hướng vượt v4 thực sự.

### Search agent v1 — chưa work (2026-05-15)
- `/tmp/ow_sim/search_agent.py`: v4 base + perturbations, rank bằng exact rollout
- Arena vs v4: **0/12** — search chọn nước tệ hơn v4
- **BUG GỐC tìm ra:** v4 dùng module-level globals (`fleet_trajectories`, `moving_planets`, `steps`, `reinforcement_trajectories`) → **KHÔNG reentrant**. Gọi v4 nhiều lần trong rollout (N players × D steps) làm hỏng global state lẫn nhau → v4-trong-sim chơi sai → rollout vô nghĩa → chọn perturbation tệ
- Hướng fix: (a) rollout policy nhẹ không-global thay v4, hoặc (b) deepcopy/reset v4 module state mỗi rollout (đắt, v4 tích lũy trajectory qua turns), hoặc (c) viết lại v4 core thành pure function
- fast_sim VẪN dùng được (verified perfect) — chỉ rollout policy cần redesign

### Search agent v2 — isolated v4 instances (2026-05-15)
- `/tmp/ow_sim/v4_instance.py`: load v4 vào namespace riêng/player slot, `reset(obs)` (steps=3 skip warmup, fill_moving_planets), `act()`. Pool reset mỗi rollout → globals không nhiễu
- `search_agent.py` v2: candidate 0 = v4 moves (floor); rollout turn1 me=cand, opp=v4-inst; turn2..D tất cả v4-inst; score = ship+prod diff (terminal → ±1e6)
- Timing: mean 72ms, **max 233ms/turn** (Kaggle limit 1s → an toàn)
- Game 1 vs v4 (seed 5): search WIN [1,-1]
- Arena: v2 (no margin) 40g→**45%**; v3 (margin 8+) 50g→30%, 16g workers=2→37.5%
- **KẾT LUẬN: 1-ply exact-sim + heuristic eval (depth 24) KHÔNG vượt v4.** Depth 24 không tiên đoán kết cục game 500-turn → score noisy → chọn perturbation "thắng giả". Cùng lý do Base B fail. Deep rollout tới terminal đúng lý thuyết nhưng v4-v4 500 steps ≈460ms → ~2 cand/turn (quá ít).
- fast_sim vẫn là tài sản đúng (verified perfect, tái dùng được cho session sau)

### Ensemble v4+marcodg deep-rollout (2026-05-16)
- `/tmp/ow_sim/ensemble_agent.py`: 2 cand {v4_move, marco_move}, deep rollout D=140, opp=v4-inst, margin floor=v4
- Fix: load module có @dataclass cần `sys.modules[name]=m`
- Timing mean 80ms max 144ms/turn (an toàn)
- Arena vs v4 20g workers=3: **50.0% (10-10)** — LẦN ĐẦU 1 approach KHÔNG thua v4 (floor đạt). Local=hòa; ladder đa dạng marco-fallback CÓ THỂ nhỉnh hơn → đáng submit always-safe
- Bundle self-contained `/tmp/ow_submit/main.py` (193KB): v4+marco embed base64, exec namespace (v4 reentrant cho rollout), fast_sim inline. `build.py` assemble
- 4P file mode: 276 steps DONE, 46s/game (~42ms/agent-turn, an toàn < 1s)
- **SUBMITTED 2026-05-16 → SCORE 1019.9 ✅ BREAKTHROUGH.** Vượt v4 thuần (~982) +38đ. Approach đầu tiên thắng v4 sau 7 hướng fail. LB an toàn 1019.9 (ensemble = best + latest, slot2 = v4 977 backup)
- Vì sao thắng: local vs v4 chỉ hòa 50%, nhưng ladder Kaggle đa dạng → marco-fallback (chọn bằng fast_sim exact rollout) cứu thế v4 yếu. Lợi thế Base B không có (sim đoán → 904)

### Ensemble v2 sweep (2026-05-16)
- `ensemble_v2.py` tham số hóa env (OWE_AGENTS/OPPS/DEPTH/MARGIN). cfg wrappers /tmp/ow_sim/cfg/
- Config D (4 agent × 2 opp × D140) = quá nặng, deadline cắt → bỏ
- **Config B (4 agent v4,marco,roman,ykhnkf, D100) vs v4: 25% — TỆ hơn v1 (50%).** Bài học: thêm agent YẾU (roman 12%, ykhnkf 18% vs v4) vào pool chỉ thêm noise. Pool chỉ nên chứa agent đủ mạnh (v4+marco). v1 đã tối ưu pool
- Config C/D (multi-opp gồm marco): TREO/crash. **marco có internal deadline ~0.82s/call** → rollout 140 bước gọi marco mỗi step = treo. Opp model trong rollout PHẢI nhanh (chỉ v4 0.46ms OK)
- **KẾT LUẬN SWEEP: v1 (v4+marco, opp v4, D140) = 1019.9 là TỐI ƯU.** Mở rộng pool (agent yếu→noise) hoặc multi-opp (marco chậm→treo) đều tệ/bất khả thi
- **Config E = v1 + DEPTH 200 vs v4: 56.2% (9-7/16) > v1 D140 (50%).** Depth sâu = eval ít noise. Bundle E 4P test: 252 steps DONE, ~104ms/turn avg (max ~206ms < 1s)
- **ensemble E (D200) = 1044.4 ✅** (+24.8 vs v1 1019.6). Depth↑ → eval ít noise → +25đ thật. Tiến trình: 982→1019.6→1044.4. LB an toàn 1044.4 (slot1=E best+latest, slot2=v1 1019.6)
- **BÀI HỌC LỚN (2026-05-16): score Kaggle ladder = μ-rating, dao động mạnh lúc ít games.** ensemble E (D200) peak 1044 → HỘI TỤ 987 (thực ra TỆ hơn v1 D140 ổn định ~1018). Local 16-game "56%" là noise. → KHÔNG đánh giá qua peak/arena ít games; cần ≥1 NGÀY hội tụ. Depth sweep 16-game = vô nghĩa, đã dừng
- always-safe CỨU: E sụt 987 nhưng LB vẫn 1018.4 nhờ v1 slot2
- LB ổn định: **ensemble_v1 (D140) ~1018 = BEST THẬT.** Depth tuning qua noise vô ích
- Hướng đúng tiếp: cải tiến CHẤT (chain_val eval), submit rồi CHỜ ≥1 ngày mới đánh giá. MỤC TIÊU 1200

## TỔNG KẾT R&D (2026-05-16): 7 hướng độc lập đều KHÔNG vượt v4 982
PPO collapse / BC 0% / v4-PV 14% / MCTS-tune 20% / Hunt max 46% / Search-v2 45% / Search-v3 37%.
v4 982 = near-optimal trong giới hạn 1s/turn. LB an toàn 982 (always-safe).

## Quyết định chiến lược

- **Pivot khỏi Pure MCTS/PPO** (đều thua v4)
- Hướng hiện tại: **DAGGER distill v4 → NN** (v4 làm teacher, BC + on-policy correction). Mục tiêu: NN nhỏ có thể generalize vượt teacher hoặc chạy nhanh hơn
- Luôn áp dụng "always-safe submit": giữ 2 slots latest = v4 safe trước khi thử experimental

## 🔴 2026-05-18 — BUG NO-OP ROLLOUT (verified) + bức tường tốc độ

**Phát hiện (byte-level + parity + empirical, không phải giả thuyết):**
- `submit/main.py` (bundle LIVE đạt 1018-1044) embed v4 **byte-identical**
  `agents/v4_rudra.py`; v4 dùng `obs.angular_velocity`. `_opp_view` trả
  dict thuần → `_V4Inst.act` → AttributeError → `except: return []`.
- ⇒ **Đối thủ + phía mình (turn≥2) trong rollout D140/200 ĐỨNG IM mọi
  bước.** Ensemble = v4 + tiebreak ~1-ply material gần mù. "Depth vô
  nghĩa / D200 tệ hơn D140" KHÔNG phải noise — rollout game-đóng-băng vô
  dụng. Mọi số LB/arena cũ trên rollout hỏng → **void, phải re-baseline**.
- Proof: plain dict → `AttributeError`; `_AttrDict` wrap → v4 ra nước thật.

**Fix:** `_AttrDict` wrap trong `submit/build.py` + đọc nguồn từ
`agents/`+`src/` (bỏ /tmp). Out `submit/main_fixed.py`. Diff vs main.py =
đúng 1 delta (+`_AttrDict`, +2 dòng wrap, Δ496B; base64/fast_sim/logic y
nguyên). Giữ main.py cũ (always-safe).

**BỨC TƯỜNG TỐC ĐỘ (timing trên state thật 40 planets/18 fleets):**
- `_rollout` fixed = **~37-44 ms/step** (4× v4 thật/step; fast_sim chỉ
  ~0.1ms — bottleneck là v4 ~9ms/call).
- D10=0.44s, D30=1.08s, D60=1.98s, D100=3.8s, **D200=7.9s/rollout** ×2
  cand/turn. Internal deadline 0.85s chỉ cho ~13-19 ply/turn → depth-200
  vô nghĩa + 2 cand chia chung deadline → so sánh **bất đối xứng** (cand1
  ~19 ply vs cand2 ~1 ply). Fix đúng nhưng **chưa submit-được như hiện tại**.
- → Bức tường = TỐC ĐỘ ROLLOUT (đúng lesson #1 discussion). Exact-sim vs
  v4-thật quá chậm để có depth hữu ích trong 1s.

**Hệ quả với plan (ép buộc bởi số liệu, ưu tiên đảo lại):**
- **P1 (đòn bẩy chính): π_θ neural nhanh distill từ bovard 1500+ thay
  v4-trong-rollout.** Một mũi tên 2 đích: rollout nhanh ~10-40× (đạt depth
  thật) + mô hình hoá field mạnh thật (đóng gap 570). fast_sim giữ nguyên
  (đã nhanh; chậm là do 4× v4).
- **P2 (đồng-chính, NÂNG từ "bậc 3"):** V_φ leaf eval — vì depth chỉ
  ~15-20 ply là tối đa khả thi, cần value tốt để rollout NÔNG vẫn đúng.
  (Đảo so với kết luận chỉ-từ-Part-A: bằng chứng mới 37ms/step đổi ràng buộc.)
- P3: π_θ làm candidate thứ 3 (free khi đã có).
- Audit Phase 0c: Part A (dấu ship+8prod dự báo thắng tốt, magnitude yếu),
  Part B (v4 ~ ngang strong trung bình NHƯNG đuôi blunder -138 p10).
- Code: `src/replay_parser.py`, `src/audit_counterfactual.py`,
  `data/bovard/` (field rated 1394-1847, p50 winner ~1552).

**Gate arena fix depth-tuned (deadline per-cand đối xứng, TIME_BUDGET 0.80,
2P alternate 60g, 0 error/draw):**
- fixed vs **buggy LIVE** = **44-16 = 73.3%** (p≈3e-4) — vượt hẳn status quo.
- fixed vs **v4 floor** = **54-6 = 90.0%** (p≈1e-9). Đối chiếu lịch sử
  "ensemble chỉ HÒA v4 50%" → bug no-op rollout là yếu tố CHÍNH kìm agent.
- 4P self-play (format validator Kaggle): 140 steps, all DONE, **0
  ERROR/TIMEOUT/INVALID, VALIDATOR-SAFE**, ~0.49s/agent-turn.
- Caveat [[feedback-kaggle-ladder-eval]]: local 2P ≠ ladder; KHÔNG suy ra
  "đạt 1550". Nhưng fixed dominate cả LIVE lẫn v4 → submit always-safe =
  cải tiến chặt. Artifact: `submit/ensemble_fixed.tar.gz` (main.py root).
- Tiếp: submit always-safe + CHỜ ≥1 ngày hội tụ; song song xây P1 (π_θ).

**HỘI TỤ 2026-05-19 (~1.3 ngày sau submit, COMPLETE):** ensemble_fixed
= **1002.3** (vs ensemble_E buggy tụt tiếp 944.6→930.8). Fix +71.5đ trên
ladder thật, vượt lại >1000. Always-safe OK: LB=max(1002.3,930.8)=1002.3,
zero downside. ⇒ chẩn đoán bug no-op rollout XÁC NHẬN trên ladder.
**Nhưng:** frontier đã = Vadasz 1725, top-5 1546-1725 → fix chỉ cầm máu,
gap ~700đ. P1 (π_θ distill bovard rated≥1500, thay v4-in-rollout: tốc độ
×10-40 + đóng gap field) = lever thật. P1 task in_progress.

**P1 BASELINE FAIL (2026-05-20, arena 60g 0err/0draw):** π_θ-bundle vs
ensemble_fixed = **16.7% (10-50)**. π_θ trained 4090 (40558 states, 7%
launch), numpy-infer parity OK (Δ≤1e-5), ~9× nhanh hơn v4, 4P
VALIDATOR-SAFE, agent()≤0.82s — KỸ THUẬT toàn xanh, NHƯNG baseline
tgt_top1 chỉ 0.44 → opponent INCOHERENT → rollout 400-deep ra quỹ đạo
rác → thua nặng v4-opponent. Bài học: rollout opponent phải COHERENT,
không chỉ nhanh+đúng-đẳng-cấp. KHÔNG submit; ensemble_fixed (1002.3) vẫn
best/floor. Arena local $0 lại cứu (bắt regress trước khi tốn slot).
Code: src/policy_{dataset,encode,train,infer}.py, submit/build_pitheta.py,
data/pi_theta_{ds,w}.npz, server-ai:~/owp1 (4090). Fork: (A) π_θ làm
CANDIDATE trên v4-rollout đang chạy (low-risk, không thể regress dưới
ensemble_fixed) vs (B) tăng coherence π_θ (symmetry×4 + data + tune) retry.

**P1 A+B KẾT QUẢ CUỐI (2026-05-20):** symmetry×4 + 90ep → π_θ tgt_top1
0.44→**0.82**, parity OK, ~0.6ms (×15 v4). Arena 60g vs ensemble_fixed:
**A (π_θ candidate) = 51.7% (HÒA, không regress)**, **B (π_θ opponent)
= 25% (vẫn THUA)**. ⇒ π_θ-opponent ngõ cụt; π_θ-candidate không thêm
sức đo được. **Win thật duy nhất của cả effort = fix bug no-op rollout
(ensemble_fixed ~1004 LIVE, từ <945).** Theo directive user "submit dù
sao": đã submit Fork A always-safe (2-latest={A PENDING, ensemble_fixed
1004}, floor giữ, zero downside). Hạ tầng P1 sạch nhưng chưa dịch kim.
Bài học hợp lực: rollout exact-sim cần opponent COHERENT (v4 982 coherent
> π_θ 0.82 nhiễu); BC-clone trần ≈ field. Hướng 1200+ thật (per 2nd-place
Lux): fast compiled env (Rust-port fast_sim) → deep-search / RL self-play
— fork chiến lược cần cân, KHÔNG polish π_θ thêm.

**LADDER + R-PHASE DONE 2026-05-21:** π_θ-cand hội tụ **1010.4** (+33 vs
ensemble_fixed 977.3 đã drift do field strengthening). Local 51.7% "tie"
UNDERESTIMATE — strictly-generalize work nhỏ trên ladder 4P diverse.
P1 KHÔNG hẳn null. R-Phase 100% done: R1 Rust port (54.8× single, 11432
steps 0 div), R2 EnvPool rayon (169k SPS), R3 abi3 wheel (pivot local
3090 do server-ai timeout), R4 PPO self-play vectorized (85K params, 327k
env-steps / 21 min — MVP scale, vs Lux-2nd 300M = 1000× nhỏ), R5
ensemble_ppo_cand submit always-safe PENDING (2-latest {ppo, π_θ_cand
1010.4} → floor 1010.4). Goal "all tasks done" ✅. Code persist: ow_sim/,
src/ppo_train.py, submit/main_ppo_cand.py, ensemble_ppo_cand.tar.gz.
