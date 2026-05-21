"""R1 MVP parity + speed test: Rust ow_sim vs Python src/fast_sim.

Toolchain proof. Once green we know maturin/PyO3 pipeline works → port
the full `step()` with confidence (test-driven, 2nd-place Lux template).
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fast_sim import _distance as py_dist, _point_to_segment_distance as py_pts
from ow_sim import py_distance as rs_dist, py_point_to_segment_distance as rs_pts

random.seed(0)
max_d = max_p = 0.0
N = 50_000
samples = [
    tuple(random.uniform(-100.0, 100.0) for _ in range(6))
    for _ in range(N)
]
for a in samples:
    max_d = max(max_d, abs(py_dist(a[0], a[1], a[2], a[3])
                            - rs_dist(a[0], a[1], a[2], a[3])))
    max_p = max(max_p, abs(py_pts(*a) - rs_pts(*a)))
print(f"PARITY over {N} random samples:")
print(f"  distance         max|Δ| = {max_d:.2e}")
print(f"  point_to_segment max|Δ| = {max_p:.2e}")
assert max_d < 1e-9 and max_p < 1e-9, "PARITY FAIL — Rust port diverges"
print("PARITY OK ✓")

# --- speed: tight loop over the hot path (point_to_segment is the main
# cost in fast_sim's per-step fleet/sweep collision checks). ---
M = 1_000_000
sub = samples[:1000]
t = time.perf_counter()
for _ in range(M // 1000):
    for a in sub:
        py_pts(*a)
py_t = time.perf_counter() - t
t = time.perf_counter()
for _ in range(M // 1000):
    for a in sub:
        rs_pts(*a)
rs_t = time.perf_counter() - t
print(f"SPEED {M:,} calls of point_to_segment_distance:")
print(f"  Python : {py_t:6.3f}s  ({M/py_t/1e6:.2f} M/s)")
print(f"  Rust   : {rs_t:6.3f}s  ({M/rs_t/1e6:.2f} M/s)")
print(f"  speedup: {py_t/rs_t:.1f}×  (per-call overhead dominates at this granularity)")
