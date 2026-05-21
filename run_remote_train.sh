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
# Torch: cu121 build (CUDA 12.x compatible — newer cu130 fails on driver
# 575.x / CUDA 12.9 like on hg-rtx4090).
if [[ ! -d .venv ]]; then
  uv venv -p 3.12
  source .venv/bin/activate
  uv pip install --index-url https://download.pytorch.org/whl/cu121 torch
  uv pip install numpy maturin wandb
else
  source .venv/bin/activate
  # ensure torch can see CUDA (auto-reinstall cu121 if it can't)
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[setup] torch missing CUDA → reinstall cu121"
    uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 torch
  fi
fi

# ow_sim wheel: build once if the COMPILED extension (with State) not present.
# (Naive `import ow_sim` succeeds spuriously because ow_sim/ subdir at repo
#  root loads as an empty namespace package — must check State attribute.)
if ! python -c "import ow_sim; assert hasattr(ow_sim, 'State')" 2>/dev/null; then
  echo "[setup] building ow_sim wheel via maturin develop --release"
  pushd ow_sim
  VIRTUAL_ENV=$(realpath ../.venv) maturin develop --release
  popd
fi
python -c "import ow_sim; assert hasattr(ow_sim, 'State'); print('ow_sim ok (State present)')"

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
PYEXE="$PWD/.venv/bin/python"  # absolute but DON'T resolve symlink (uv venv → site-packages discovery)
echo "launching, log: $LOG  python: $PYEXE"
nohup "$PYEXE" src/ppo_train.py data/ppo_w.npz > "$LOG" 2>&1 &
NEW_PID=$!
disown 2>/dev/null || true
echo "PID $NEW_PID — log $LOG"
sleep 8
head -8 "$LOG"
echo "--- tail -f $LOG  or  wandb dashboard ---"
