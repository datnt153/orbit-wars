"""R1 — full-step parity Python fast_sim vs Rust ow_sim.

Loads real bovard initial states, plays the same random joint actions in
both sims for K steps, compares state field-by-field. R1 go/no-go:
- 100% parity over many seeds × steps (zero divergence beyond float eps).
- Rust single-thread SPS ≥50k (vs Python's ~11k).
"""
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fast_sim
import ow_sim


def gen_actions(state_dict, num_agents, rng):
    """Random joint action: with prob 0.3, each owned planet launches a
    random ship fraction toward a random target planet."""
    pos = {int(p[0]): (p[2], p[3]) for p in state_dict["planets"]}
    acts = []
    for ag in range(num_agents):
        moves = []
        owned = [p for p in state_dict["planets"] if int(p[1]) == ag]
        targets = [p for p in state_dict["planets"]]
        if not targets:
            acts.append(moves); continue
        for op in owned:
            if rng.random() > 0.3 or op[5] < 2:
                continue
            tgt = rng.choice(targets)
            if int(tgt[0]) == int(op[0]):
                continue
            frac = rng.uniform(0.3, 1.0)
            n = max(1, int(op[5] * frac))
            angle = math.atan2(tgt[3] - op[3], tgt[2] - op[2])
            moves.append([int(op[0]), float(angle), int(n)])
        acts.append(moves)
    return acts


def cmp_states(py_s, rs_d):
    """Field-by-field comparison. Returns (ok, max_float_diff, summary)."""
    fails = []
    def fdiff(a, b):
        if isinstance(a, float) or isinstance(b, float):
            return abs(float(a) - float(b))
        return 0.0 if a == b else float("inf")

    # planets
    pp = py_s["planets"]; rp = rs_d["planets"]
    if len(pp) != len(rp):
        fails.append(f"planets len {len(pp)} vs {len(rp)}")
    else:
        for i, (a, b) in enumerate(zip(pp, rp)):
            for k in range(7):
                d = fdiff(a[k], b[k])
                if d > 1e-9:
                    fails.append(f"planets[{i}][{k}] {a[k]} vs {b[k]} Δ={d:.2e}")
    # fleets
    pf = py_s["fleets"]; rf = rs_d["fleets"]
    if len(pf) != len(rf):
        fails.append(f"fleets len {len(pf)} vs {len(rf)}")
    else:
        for i, (a, b) in enumerate(zip(pf, rf)):
            for k in range(7):
                d = fdiff(a[k], b[k])
                if d > 1e-9:
                    fails.append(f"fleets[{i}][{k}] {a[k]} vs {b[k]} Δ={d:.2e}")
    # scalars
    for k in ("next_fleet_id", "step", "_done"):
        if py_s.get(k) != rs_d.get(k):
            fails.append(f"{k}: {py_s.get(k)} vs {rs_d.get(k)}")
    return (not fails), fails


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    eps_dir = ROOT / "data" / "bovard" / "2026-05-04" / "episodes" / "episodes"
    eps = sorted(eps_dir.glob("*.json"))[:n_eps]
    print(f"Parity harness: {len(eps)} episodes × {n_steps} steps")

    total_fail = 0
    total_step = 0
    py_time = 0.0
    rs_time = 0.0
    for ei, ep_path in enumerate(eps):
        replay = json.loads(ep_path.read_text())
        obs = replay["steps"][0][0]["observation"]
        num_agents = len(replay["steps"][0])
        ship_speed = float(replay.get("configuration", {}).get("shipSpeed", 6.0))
        episode_steps = int(replay.get("configuration", {}).get("episodeSteps", 500))

        # Init both sims from the SAME obs.
        py_state = fast_sim.new_state(obs, ship_speed=ship_speed,
                                      episode_steps=episode_steps)
        rs_state = ow_sim.State(obs, ship_speed, episode_steps)

        rng = random.Random(ei * 7 + 1)
        for s in range(n_steps):
            acts = gen_actions(py_state, num_agents, rng)
            t = time.perf_counter()
            fast_sim.step(py_state, acts, num_agents)
            py_time += time.perf_counter() - t
            t = time.perf_counter()
            rs_state.step(acts, num_agents)
            rs_time += time.perf_counter() - t
            total_step += 1
            rs_d = rs_state.to_dict()
            ok, fails = cmp_states(py_state, rs_d)
            if not ok:
                total_fail += 1
                if total_fail <= 3:
                    print(f"  ep{ei} step{s} DIVERGE:")
                    for f in fails[:5]:
                        print(f"    {f}")
                break  # next episode
            if py_state.get("_done"):
                break

    print(f"\n=== R1 PARITY: {total_step} steps, {total_fail} ep diverged "
          f"({'PASS' if total_fail==0 else 'FAIL'}) ===")
    print(f"  Python fast_sim: {1000*py_time/max(1,total_step):.3f} ms/step  "
          f"({total_step/max(1e-9,py_time):.0f} SPS)")
    print(f"  Rust ow_sim    : {1000*rs_time/max(1,total_step):.3f} ms/step  "
          f"({total_step/max(1e-9,rs_time):.0f} SPS)")
    print(f"  speedup        : {py_time/max(1e-9,rs_time):.1f}×")


if __name__ == "__main__":
    main()
