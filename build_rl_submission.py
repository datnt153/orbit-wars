from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a tar.gz submission for the PPO strategy agent.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("ppo_strategy_policy.npz"),
        help="Path to the exported PPO weights (.npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist_rl_submission"),
        help="Directory where the submission payload will be assembled.",
    )
    parser.add_argument(
        "--tarball",
        type=Path,
        default=Path("dist_rl_submission.tar.gz"),
        help="Tarball to create after staging the files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    out = args.output_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy2(root / "main.py", out / "baseline_bot.py")
    shutil.copy2(root / "ppo_strategy.py", out / "ppo_strategy.py")
    shutil.copy2(root / "rl_submission_main.py", out / "main.py")

    if args.weights.exists():
        shutil.copy2(args.weights.resolve(), out / "ppo_strategy_policy.npz")
    else:
        print(f"warning: weights not found at {args.weights}, building heuristic-only wrapper")

    with tarfile.open(args.tarball.resolve(), "w:gz") as tar:
        for path in sorted(out.iterdir()):
            tar.add(path, arcname=path.name)

    print(f"staged={out}")
    print(f"tarball={args.tarball.resolve()}")


if __name__ == "__main__":
    main()
