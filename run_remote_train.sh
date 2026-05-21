#!/usr/bin/env bash
# Launch 300M PPO training on hg-rtx4090 (80 CPU threads, 2× RTX 4090).
# Workflow: code pulled from GitHub; this script orchestrates the run.
#
# Usage on hg-rtx4090:
#   cd ~/orbit-wars
#   git pull origin master
#   bash run_remote_train.sh
set -euo pipefail
cd "$(dirname "$0")"

[[ "$(hostname)" == *4090* ]] || echo "[warn] not on the rtx4090 host: $(hostname)"
[[ -f data/ppo_w.npz ]] || echo "[warn] no ppo_w.npz to resume — will train from scratch"

# Python env: uv venv at .venv. Build once if missing.
if [[ ! -d .venv ]]; then
  uv venv -p 3.12
  source .venv/bin/activate
  uv pip install torch numpy maturin wandb
else
  source .venv/bin/activate
fi

# ow_sim wheel: build once if not installed.
if ! python -c "import ow_sim" 2>/dev/null; then
  pushd ow_sim
  VIRTUAL_ENV=$(realpath ../.venv) maturin develop --release
  popd
fi
python -c "import ow_sim; print('ow_sim ok')"

# Wandb (one-time on this host: wandb login <key>).
if wandb status 2>/dev/null | grep -q "Logged in"; then
  export WANDB=1
  export WANDB_PROJECT="orbit-wars"
  export WANDB_NAME="ppo_300M_$(date +%Y%m%d_%H%M)"
else
  echo "[info] wandb not logged in — disabling. Run 'wandb login' to enable."
  export WANDB=0
fi

export CUDA_VISIBLE_DEVICES="0"            # use GPU 0 only (2 GPUs available)
export RAYON_NUM_THREADS="80"              # all CPU threads for Rust env
export N_ENVS="128"
export N_STEPS="128"
export UPDATES="${UPDATES:-18000}"         # ~300M env-steps target
export MB="1024"
export ENT="0.005"
export SHAPE="0.01"
export LR="3e-4"
export RESUME="data/ppo_w.npz"

LOG=/tmp/ppo_300M_$(date +%s).log
echo "launching, log: $LOG"
nohup python src/ppo_train.py data/ppo_w.npz > "$LOG" 2>&1 &
NEW_PID=$!
disown 2>/dev/null || true
echo "PID $NEW_PID — log $LOG"
sleep 5
head -5 "$LOG"
echo "--- tail -f $LOG  or  wandb dashboard ---"
