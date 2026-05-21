#!/usr/bin/env bash
# Daily PPO submit helper. Usage:
#   submit/daily.sh                                  # build + 4P gate + submit
#   submit/daily.sh skip-gate                        # build + submit (no gate, risky)
# Pairs the new PPO bundle always-safe with whatever is currently in
# slot-2 (the prior latest). Quota 5/day.
set -e
cd "$(dirname "$0")/.."
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_EXE 2>/dev/null || true

CKPT="data/ppo_w.npz"
[ -f "$CKPT" ] || { echo "no checkpoint $CKPT"; exit 1; }
TOTAL_STEPS=$(.venv/bin/python -c "import numpy as np; print(int(np.load('$CKPT')['_STEPS']))" 2>/dev/null || echo "?")
echo "[daily] training steps so far: $TOTAL_STEPS"

echo "[daily] build bundle from $CKPT"
.venv/bin/python submit/build_pitheta_cand.py "$CKPT" main_ppo_daily.py 2>&1 | tail -1

cp submit/main_ppo_daily.py /tmp/main.py
tar -czf submit/ensemble_ppo_daily.tar.gz -C /tmp main.py
rm /tmp/main.py
echo "[daily] tar: $(ls -la submit/ensemble_ppo_daily.tar.gz | awk '{print $5}') bytes"

if [ "$1" != "skip-gate" ]; then
  echo "[daily] 4P validator gate (one self-play game, ~5-15 min)"
  .venv/bin/python - <<PY
import time
from kaggle_environments import make
env=make("orbit_wars", configuration={"agents":4}, debug=False)
t0=time.time(); env.run(["submit/main_ppo_daily.py"]*4); wall=time.time()-t0
steps=env.steps
bad={s.get("status") for st in steps for s in st} - {"ACTIVE","DONE","INACTIVE"}
fin=steps[-1]
ok=(not bad) and all(s["status"]=="DONE" for s in fin)
print(f"[daily] 4P: {len(steps)}s wall={wall:.0f}s SAFE={ok} bad={bad or 'NONE'}")
if not ok:
    import sys; sys.exit(2)
PY
fi

MSG="PPO daily ${TOTAL_STEPS} env-steps. Build_pitheta_cand on v4-opponent rollout + 1-ply prefilter. Strictly generalizes ensemble_fixed (v4 floor). Always-safe."
echo "[daily] submit always-safe"
.venv/bin/kaggle competitions submit -c orbit-wars \
  -f submit/ensemble_ppo_daily.tar.gz -m "$MSG" 2>&1 | tail -2

echo "[daily] verify"
.venv/bin/kaggle competitions submissions orbit-wars 2>&1 | grep -v -i warning | head -5
