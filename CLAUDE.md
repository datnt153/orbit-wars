# Orbit Wars — project guide (đọc trước khi làm)

Kaggle Orbit Wars (RTS không gian 2P+4P, sun ở (50,50) r=10, 500 turn). Deadline
2026-06-23, $50K. Mục tiêu 1200 elo. LB thật hiện = ensemble rule-based ~1013.

## 📚 Đọc các doc này (đừng quên!)
- **`learn.md`** — BÀI HỌC RL từ top solutions Lux S3 (1st/2nd) + vì sao self-play
  của ta plateau + 3 fix (opponent pool+teacher-KL, value-2-phe, features+history).
- **`backlog.md`** — findings F1-F5 + forum intel (1st-place Orbit Wars, istinetz,
  aidensong) + backlog ưu tiên.
- **`jax_port.md`** — JAX infra (env/encode/policy/PPO parity, ~28k SPS train).
- **`progress.md`** / **`plan.md`** — nhật ký + kế hoạch.

## 🔑 Sự thật cốt lõi (đừng lặp sai lầm)
- **Self-play awr ≠ sức ladder.** Gate THẬT = `arena_vs_v4.py ckpt N {2|4}` — chỉ
  submit khi **≥0.5**. Đã có 4 recipe self-play đều ~7% vs v4 (awr self-play tới 0.9).
- **Submit BẮT BUỘC always-safe** (giữ floor; candidate-wrapped KHÔNG phải floor thật
  khi PPO fire — `ensemble_ppo_jax` 39M tụt 905 < v4).
- **Train 2P** (winners + Lux đều 1v1=2P; 4P terminal quá thưa). Maps 2P:
  `make("orbit_wars",{agents:2})` → `data/maps_2p/`.
- Đừng tin run ngắn / metric log; arena vs đối thủ NGOÀI dòng dõi trước khi tin.

## 🛠 Hạ tầng
- GPU server `datnt114@hg-rtx4090`: worktree `~/orbit-wars-jax` (venv jax[cuda12]+
  optax+wandb riêng) CÔ LẬP, train trên **GPU1** + `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Train: `src/jax_train.py` (env vars: NUM_AGENTS N_ENVS N_STEPS MB UPDATES SHAPE_SCALE
  ENT ANCHOR_CKPT MAPS_DIR OUT WANDB). Arena: `src/arena_vs_v4.py`. wandb: kingkong153.
- tmux/nohup KHÔNG sống sót reboot → lưu ckpt thường xuyên, resume từ ckpt.

## ⚙️ Quy tắc
- UV cho mọi thứ Python (`uv run`, `uv pip`). Git commit KHÔNG co-author.
- Test bundle: file-mode + **4P** (validator), tránh `__file__`, đảm bảo không no-op.
