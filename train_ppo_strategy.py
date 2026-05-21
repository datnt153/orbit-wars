from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent, starter_agent
from torch import nn
from torch.distributions import Categorical

import main as baseline_bot
import ppo_strategy


@dataclass
class Transition:
    obs: np.ndarray
    raw_obs: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool


class RunningNorm:
    def __init__(self, dim: int, clip: float = 5.0):
        self.dim = dim
        self.clip = clip
        self.count = 1e-4
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.ones(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        return self.m2 / max(1.0, self.count - 1.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-6))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)


class StrategyActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


def export_numpy_policy(model: StrategyActorCritic, norm: RunningNorm, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "w1": model.fc1.weight.detach().cpu().numpy().T.astype(np.float32),
        "b1": model.fc1.bias.detach().cpu().numpy().astype(np.float32),
        "w2": model.fc2.weight.detach().cpu().numpy().T.astype(np.float32),
        "b2": model.fc2.bias.detach().cpu().numpy().astype(np.float32),
        "policy_w": model.policy_head.weight.detach().cpu().numpy().T.astype(np.float32),
        "policy_b": model.policy_head.bias.detach().cpu().numpy().astype(np.float32),
        "value_w": model.value_head.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "value_b": model.value_head.bias.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "obs_mean": norm.mean.astype(np.float32),
        "obs_std": norm.std.astype(np.float32),
    }
    np.savez(path, **payload)


def fixed_strategy_agent(strategy_id: int):
    def agent(obs, config=None):
        world = baseline_bot.build_world(obs)
        if not world.my_planets:
            return []
        modes = baseline_bot.build_modes(world)
        policy = baseline_bot.build_policy_state(world)
        return ppo_strategy.plan_with_strategy(
            world,
            strategy_id=strategy_id,
            config=config,
            policy=policy,
            modes=modes,
        )

    return agent


def snapshot_strategy_agent(policy_path: Path):
    policy_path = policy_path.resolve()
    model = ppo_strategy.NumpyStrategyPolicy.load(policy_path)

    def agent(obs, config=None):
        world = baseline_bot.build_world(obs)
        if not world.my_planets:
            return []
        modes = baseline_bot.build_modes(world)
        policy = baseline_bot.build_policy_state(world)
        strategy_id, _, _ = ppo_strategy.choose_strategy(
            world,
            policy=policy,
            modes=modes,
            model=model,
        )
        return ppo_strategy.plan_with_strategy(
            world,
            strategy_id=strategy_id,
            config=config,
            policy=policy,
            modes=modes,
        )

    return agent


def build_fixed_opponent_pool() -> list[tuple[str, callable]]:
    return [
        ("random", random_agent),
        ("starter", starter_agent),
        ("baseline", baseline_bot.agent),
        ("balanced", fixed_strategy_agent(0)),
        ("comet", fixed_strategy_agent(2)),
        ("hostile", fixed_strategy_agent(3)),
        ("fortress", fixed_strategy_agent(4)),
    ]


def snapshot_paths(snapshot_dir: Path) -> list[Path]:
    if not snapshot_dir.exists():
        return []
    return sorted(snapshot_dir.glob("snapshot_ep*.npz"))


def trim_snapshots(snapshot_dir: Path, max_snapshots: int) -> list[Path]:
    paths = snapshot_paths(snapshot_dir)
    if max_snapshots <= 0:
        for path in paths:
            path.unlink(missing_ok=True)
        return []
    while len(paths) > max_snapshots:
        oldest = paths.pop(0)
        oldest.unlink(missing_ok=True)
    return paths


def save_snapshot(model: StrategyActorCritic, norm: RunningNorm, snapshot_dir: Path, episode: int) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"snapshot_ep{episode:04d}.npz"
    export_numpy_policy(model, norm, path)
    return path


def sample_opponent(
    fixed_pool: list[tuple[str, callable]],
    snapshot_pool: list[tuple[str, callable]],
    self_play_prob: float,
) -> tuple[str, callable]:
    if snapshot_pool and random.random() < self_play_prob:
        return random.choice(snapshot_pool)
    return random.choice(fixed_pool)


def build_fixed_opponent_specs() -> list[dict[str, object]]:
    return [
        {"type": "builtin", "name": "random"},
        {"type": "builtin", "name": "starter"},
        {"type": "builtin", "name": "baseline"},
        {"type": "fixed", "name": "balanced", "strategy_id": 0},
        {"type": "fixed", "name": "comet", "strategy_id": 2},
        {"type": "fixed", "name": "hostile", "strategy_id": 3},
        {"type": "fixed", "name": "fortress", "strategy_id": 4},
    ]


def snapshot_specs(snapshot_dir: Path, max_snapshots: int) -> list[dict[str, object]]:
    return [
        {"type": "snapshot", "name": f"snapshot:{path.stem}", "path": str(path.resolve())}
        for path in trim_snapshots(snapshot_dir, max_snapshots)
    ]


def sample_opponent_spec(
    fixed_specs: list[dict[str, object]],
    snapshot_specs_list: list[dict[str, object]],
    self_play_prob: float,
) -> dict[str, object]:
    if snapshot_specs_list and random.random() < self_play_prob:
        return dict(random.choice(snapshot_specs_list))
    return dict(random.choice(fixed_specs))


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return np.full_like(probs, 1.0 / len(probs))
    return probs / probs_sum


def opponent_action_from_spec(spec: dict[str, object], obs, config=None):
    kind = spec["type"]
    if kind == "builtin":
        name = spec["name"]
        if name == "random":
            return random_agent(obs)
        if name == "starter":
            return starter_agent(obs)
        if name == "baseline":
            return baseline_bot.agent(obs, config)
        raise ValueError(f"unknown builtin opponent: {name}")

    world = baseline_bot.build_world(obs)
    if not world.my_planets:
        return []
    modes = baseline_bot.build_modes(world)
    policy = baseline_bot.build_policy_state(world)

    if kind == "fixed":
        return ppo_strategy.plan_with_strategy(
            world,
            strategy_id=int(spec["strategy_id"]),
            config=config,
            policy=policy,
            modes=modes,
        )

    if kind == "snapshot":
        model = ppo_strategy.NumpyStrategyPolicy.load(spec["path"])
        strategy_id, _, _ = ppo_strategy.choose_strategy(
            world,
            policy=policy,
            modes=modes,
            model=model,
        )
        return ppo_strategy.plan_with_strategy(
            world,
            strategy_id=strategy_id,
            config=config,
            policy=policy,
            modes=modes,
        )

    raise ValueError(f"unknown opponent type: {kind}")


def run_episode_worker(task: dict[str, object]) -> tuple[list[Transition], dict[str, float]]:
    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)

    args = task["args"]
    actor_path = Path(task["actor_path"])
    opponent_spec = dict(task["opponent_spec"])
    actor_model = ppo_strategy.NumpyStrategyPolicy.load(actor_path)

    baseline_bot.TOTAL_STEPS = int(args["episode_steps"])
    env = make(
        "orbit_wars",
        configuration={"episodeSteps": int(args["episode_steps"])},
        debug=False,
    )
    state = env.reset(2)
    transitions: list[Transition] = []
    episode_reward = 0.0
    action_hist = np.zeros(len(ppo_strategy.STRATEGY_PRESETS), dtype=np.int64)

    while not env.done:
        obs0 = state[0].observation
        obs1 = state[1].observation
        world0 = baseline_bot.build_world(obs0)

        if world0.my_planets:
            policy0 = baseline_bot.build_policy_state(world0)
            modes0 = baseline_bot.build_modes(world0)
            raw_feats = ppo_strategy.extract_features(world0, policy=policy0, modes=modes0)
            logits, value = actor_model.forward(raw_feats)
            probs = logits_to_probs(logits)
            action = int(np.random.choice(len(probs), p=probs))
            logp = float(np.log(max(probs[action], 1e-8)))
            action_hist[action] += 1
            moves0 = ppo_strategy.plan_with_strategy(
                world0,
                strategy_id=action,
                config=env.configuration,
                policy=policy0,
                modes=modes0,
            )
            score_before = ppo_strategy.progress_score(world0)
            transition = Transition(
                obs=actor_model._normalize(raw_feats),
                raw_obs=raw_feats.astype(np.float32, copy=True),
                action=action,
                logp=logp,
                value=value,
                reward=0.0,
                done=False,
            )
        else:
            moves0 = []
            transition = None
            score_before = 0.0

        moves1 = opponent_action_from_spec(opponent_spec, obs1, env.configuration)
        state = env.step([moves0, moves1])

        if transition is not None:
            next_world0 = baseline_bot.build_world(state[0].observation)
            score_after = ppo_strategy.progress_score(next_world0) if next_world0.my_planets else -1.0
            reward = float(args["shaping_scale"]) * (score_after - score_before)
            if env.done:
                reward += float(args["outcome_scale"]) * float(state[0].reward)
            transition.reward = reward
            transition.done = env.done
            transitions.append(transition)
            episode_reward += reward

    info = {
        "opponent": str(opponent_spec["name"]),
        "episode_reward": episode_reward,
        "final_env_reward": float(state[0].reward),
        "steps": float(len(transitions)),
    }
    for idx, preset in enumerate(ppo_strategy.STRATEGY_PRESETS):
        info[f"action_{preset.name}"] = float(action_hist[idx])
    return transitions, info


def choose_action(
    model: StrategyActorCritic,
    norm: RunningNorm,
    obs: np.ndarray,
    device: torch.device,
) -> tuple[int, float, float]:
    obs_t = torch.from_numpy(norm.normalize(obs)).to(device).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
    return int(action.item()), float(logp.item()), float(value.item())


def compute_returns_and_advantages(
    transitions: list[Transition],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.array([item.reward for item in transitions], dtype=np.float32)
    values = np.array([item.value for item in transitions], dtype=np.float32)
    dones = np.array([item.done for item in transitions], dtype=np.float32)

    advantages = np.zeros_like(rewards)
    last_adv = 0.0
    next_value = 0.0
    for idx in range(len(transitions) - 1, -1, -1):
        mask = 1.0 - dones[idx]
        delta = rewards[idx] + gamma * next_value * mask - values[idx]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[idx] = last_adv
        next_value = values[idx]
    returns = advantages + values
    return returns, advantages


def update_policy(
    model: StrategyActorCritic,
    optimizer: torch.optim.Optimizer,
    norm: RunningNorm,
    transitions: list[Transition],
    args,
    device: torch.device,
) -> dict[str, float]:
    returns, advantages = compute_returns_and_advantages(
        transitions,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

    obs = np.stack([item.obs for item in transitions]).astype(np.float32)
    actions = np.array([item.action for item in transitions], dtype=np.int64)
    old_logp = np.array([item.logp for item in transitions], dtype=np.float32)

    obs_t = torch.from_numpy(obs).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    old_logp_t = torch.from_numpy(old_logp).to(device)
    returns_t = torch.from_numpy(returns.astype(np.float32)).to(device)
    adv_t = torch.from_numpy(advantages.astype(np.float32)).to(device)

    batch_size = obs_t.shape[0]
    minibatch = min(args.minibatch_size, batch_size)
    stats = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
    updates = 0
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for _ in range(args.update_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch):
            mb_idx = indices[start : start + minibatch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, values = model(obs_t[mb_idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions_t[mb_idx])
                ratio = torch.exp(logp - old_logp_t[mb_idx])
                unclipped = ratio * adv_t[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * adv_t[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (returns_t[mb_idx] - values).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            stats["loss"] += float(loss.item())
            stats["policy"] += float(policy_loss.item())
            stats["value"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            updates += 1

    if updates:
        for key in stats:
            stats[key] /= updates
    return stats


def run_episode(
    model: StrategyActorCritic,
    norm: RunningNorm,
    opponent_name: str,
    opponent_agent,
    args,
    device: torch.device,
) -> tuple[list[Transition], dict[str, float]]:
    baseline_bot.TOTAL_STEPS = args.episode_steps
    env = make(
        "orbit_wars",
        configuration={"episodeSteps": args.episode_steps},
        debug=False,
    )
    state = env.reset(2)
    transitions: list[Transition] = []
    episode_reward = 0.0
    action_hist = np.zeros(len(ppo_strategy.STRATEGY_PRESETS), dtype=np.int64)

    while not env.done:
        obs0 = state[0].observation
        obs1 = state[1].observation
        world0 = baseline_bot.build_world(obs0)

        if world0.my_planets:
            policy0 = baseline_bot.build_policy_state(world0)
            modes0 = baseline_bot.build_modes(world0)
            feats = ppo_strategy.extract_features(world0, policy=policy0, modes=modes0)
            norm.update(feats)
            action, logp, value = choose_action(model, norm, feats, device=device)
            action_hist[action] += 1
            moves0 = ppo_strategy.plan_with_strategy(
                world0,
                strategy_id=action,
                config=env.configuration,
                policy=policy0,
                modes=modes0,
            )
            score_before = ppo_strategy.progress_score(world0)
            transition = Transition(
                obs=feats,
                action=action,
                logp=logp,
                value=value,
                reward=0.0,
                done=False,
            )
        else:
            moves0 = []
            transition = None
            score_before = 0.0

        moves1 = opponent_agent(obs1, env.configuration)
        state = env.step([moves0, moves1])

        if transition is not None:
            next_world0 = baseline_bot.build_world(state[0].observation)
            score_after = ppo_strategy.progress_score(next_world0) if next_world0.my_planets else -1.0
            reward = args.shaping_scale * (score_after - score_before)
            if env.done:
                reward += args.outcome_scale * float(state[0].reward)
            transition.reward = reward
            transition.done = env.done
            transitions.append(transition)
            episode_reward += reward

    info = {
        "opponent": opponent_name,
        "episode_reward": episode_reward,
        "final_env_reward": float(state[0].reward),
        "steps": float(len(transitions)),
    }
    for idx, preset in enumerate(ppo_strategy.STRATEGY_PRESETS):
        info[f"action_{preset.name}"] = float(action_hist[idx])
    return transitions, info


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO strategy selector for Orbit Wars.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--self-play-prob", type=float, default=0.35)
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("ppo_strategy_snapshots"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shaping-scale", type=float, default=0.8)
    parser.add_argument("--outcome-scale", type=float, default=1.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ppo_strategy_policy.npz"),
    )
    return parser.parse_args()


def collect_rollout_batch(
    model: StrategyActorCritic,
    norm: RunningNorm,
    fixed_specs: list[dict[str, object]],
    snapshot_specs_list: list[dict[str, object]],
    args,
    total_episodes: int,
) -> tuple[list[Transition], list[dict[str, float]], int]:
    remaining = args.episodes - total_episodes
    batch_size = min(args.rollout_episodes, remaining)
    if batch_size <= 0:
        return [], [], total_episodes

    rollout_path = args.snapshot_dir / "_rollout_actor.npz"
    export_numpy_policy(model, norm, rollout_path)
    task_args = {
        "episode_steps": args.episode_steps,
        "shaping_scale": args.shaping_scale,
        "outcome_scale": args.outcome_scale,
    }
    tasks = []
    for offset in range(batch_size):
        opponent_spec = sample_opponent_spec(
            fixed_specs,
            snapshot_specs_list,
            self_play_prob=args.self_play_prob,
        )
        tasks.append(
            {
                "seed": args.seed + total_episodes + offset + 1,
                "args": task_args,
                "actor_path": str(rollout_path.resolve()),
                "opponent_spec": opponent_spec,
            }
        )

    if args.num_workers <= 1 or batch_size == 1:
        results = [run_episode_worker(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.num_workers, batch_size)) as pool:
            results = list(pool.map(run_episode_worker, tasks))

    transitions: list[Transition] = []
    infos: list[dict[str, float]] = []
    for batch_transitions, info in results:
        transitions.extend(batch_transitions)
        infos.append(info)
    total_episodes += batch_size
    return transitions, infos, total_episodes


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ppo_strategy.set_baseline_module(baseline_bot)
    obs_dim = ppo_strategy.extract_features(
        baseline_bot.build_world(
            {
                "player": 0,
                "step": 0,
                "planets": [],
                "fleets": [],
                "angular_velocity": 0.0,
                "initial_planets": [],
                "comets": [],
                "comet_planet_ids": [],
            }
        ),
        policy={"reaction_time_map": {}, "reserve": {}, "attack_budget": {}},
        modes={
            "domination": 0.0,
            "is_behind": False,
            "is_ahead": False,
            "is_finishing": False,
        },
    ).shape[0]
    action_dim = len(ppo_strategy.STRATEGY_PRESETS)
    if args.device == "auto":
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = StrategyActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    norm = RunningNorm(obs_dim)
    fixed_specs = build_fixed_opponent_specs()
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_specs_list = snapshot_specs(args.snapshot_dir, args.max_snapshots)

    all_transitions: list[Transition] = []
    total_episodes = 0
    while total_episodes < args.episodes:
        batch_start_episode = total_episodes + 1
        transitions, infos, total_episodes = collect_rollout_batch(
            model=model,
            norm=norm,
            fixed_specs=fixed_specs,
            snapshot_specs_list=snapshot_specs_list,
            args=args,
            total_episodes=total_episodes,
        )
        for idx, info in enumerate(infos):
            print(
                f"episode={batch_start_episode + idx} opponent={info['opponent']} "
                f"reward={info['episode_reward']:.3f} env_reward={info['final_env_reward']:.1f} "
                f"steps={int(info['steps'])}"
            )
        for item in transitions:
            norm.update(item.raw_obs)
        all_transitions.extend(transitions)
        if all_transitions:
            stats = update_policy(
                model=model,
                optimizer=optimizer,
                norm=norm,
                transitions=all_transitions,
                args=args,
                device=device,
            )
            print(
                "update "
                f"loss={stats['loss']:.4f} policy={stats['policy']:.4f} "
                f"value={stats['value']:.4f} entropy={stats['entropy']:.4f}"
            )
            all_transitions.clear()
            export_numpy_policy(model, norm, args.output)
            if args.snapshot_interval > 0 and total_episodes % args.snapshot_interval == 0:
                snap_path = save_snapshot(model, norm, args.snapshot_dir, total_episodes)
                snapshot_specs_list = snapshot_specs(args.snapshot_dir, args.max_snapshots)
                print(f"snapshot={snap_path} snapshot_pool={len(snapshot_specs_list)}")

    if all_transitions:
        stats = update_policy(
            model=model,
            optimizer=optimizer,
            norm=norm,
            transitions=all_transitions,
            args=args,
            device=device,
        )
        print(
            "final_update "
            f"loss={stats['loss']:.4f} policy={stats['policy']:.4f} "
            f"value={stats['value']:.4f} entropy={stats['entropy']:.4f}"
        )

    export_numpy_policy(model, norm, args.output)
    print(f"saved_policy={args.output}")


if __name__ == "__main__":
    main()
