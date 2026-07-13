from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from rlgym.rocket_league import common_values
from rlgym_ppo.ppo import DiscreteFF

from env_setup import load_config, make_gym_env


BLUE = common_values.BLUE_TEAM
ORANGE = common_values.ORANGE_TEAM


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def find_latest_checkpoint(root: str | Path = "outputs/ppo", run_prefix: str | None = None) -> Path:
    root = Path(root)
    candidates = []
    for book_path in root.rglob("BOOK_KEEPING_VARS.json"):
        run_dir = book_path.parent.parent
        if run_prefix and not (run_dir.name == run_prefix or run_dir.name.startswith(f"{run_prefix}-")):
            continue

        try:
            with book_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            candidates.append((book_path.stat().st_mtime, int(info["cumulative_timesteps"]), book_path.parent))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue

    if not candidates:
        suffix = f" matching run prefix {run_prefix!r}" if run_prefix else ""
        raise FileNotFoundError(f"No PPO checkpoints found under {root}{suffix}")

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def latest_checkpoint_from_config(config: Dict[str, Any]) -> Path:
    save_folder = Path(config["ppo"].get("checkpoints_save_folder") or "outputs/ppo")
    if save_folder.exists():
        return find_latest_checkpoint(save_folder)

    if save_folder.parent.exists():
        return find_latest_checkpoint(save_folder.parent, run_prefix=save_folder.name)

    return find_latest_checkpoint()


def resolve_checkpoint(path: str | Path, config: Dict[str, Any]) -> Path:
    if str(path).lower() == "latest":
        return latest_checkpoint_from_config(config)

    checkpoint = Path(path)
    if (checkpoint / "BOOK_KEEPING_VARS.json").exists():
        return checkpoint

    if checkpoint.exists():
        numeric_children = [child for child in checkpoint.iterdir() if child.is_dir() and child.name.isdigit()]
        if numeric_children:
            return max(numeric_children, key=lambda child: int(child.name))

    raise FileNotFoundError(f"Could not resolve checkpoint folder: {checkpoint}")


def load_bookkeeping(checkpoint: Path) -> Dict[str, Any]:
    with (checkpoint / "BOOK_KEEPING_VARS.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_obs_stats(bookkeeping: Dict[str, Any]):
    stats = bookkeeping.get("obs_running_stats")
    if not stats:
        return None, None

    count = int(stats["count"])
    mean = np.asarray(stats["mean"], dtype=np.float32)
    running_var = np.asarray(stats["var"], dtype=np.float32)
    if count < 2:
        return mean, np.ones_like(mean, dtype=np.float32)

    variance = running_var / (count - 1)
    variance = np.where(variance == 0, 1.0, variance)
    return mean, np.sqrt(variance).astype(np.float32)


def normalize_obs(obs: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if mean is None or std is None:
        return obs
    return np.clip((obs - mean) / std, a_min=-5.0, a_max=5.0)


def load_policy(config: Dict[str, Any], env, checkpoint: Path, device: torch.device) -> DiscreteFF:
    policy = DiscreteFF(
        input_shape=int(np.prod(env.observation_space.shape)),
        n_actions=int(env.action_space.n),
        layer_sizes=list(config["ppo"]["hidden_sizes"]),
        device=device,
    )
    try:
        state_dict = torch.load(checkpoint / "PPO_POLICY.pt", map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint / "PPO_POLICY.pt", map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def policy_actions(
    policy: DiscreteFF,
    obs: np.ndarray,
    deterministic: bool,
) -> np.ndarray:
    with torch.no_grad():
        probs = policy.get_output(obs).view(obs.shape[0], policy.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1.0)
        if deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return actions.cpu().numpy().astype(np.int64)


def uses_policy(mode: str, team: int) -> bool:
    if mode == "policy_vs_policy":
        return True
    if mode == "policy_vs_random":
        return team == BLUE
    if mode == "random_vs_policy":
        return team == ORANGE
    if mode == "random_vs_random":
        return False
    raise ValueError(f"Unknown eval mode: {mode}")


def team_name(team: int) -> str:
    return "blue" if team == BLUE else "orange"


def evaluate(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    eval_cfg = config.get("eval", {})
    checkpoint = resolve_checkpoint(args.checkpoint or eval_cfg.get("checkpoint_folder", "latest"), config)
    bookkeeping = load_bookkeeping(checkpoint)
    device = resolve_device(args.device or str(config["ppo"].get("device", "auto")))
    env_config = copy.deepcopy(config)
    env_config.setdefault("state_curriculum", {})
    env_config["state_curriculum"]["enabled"] = bool(eval_cfg.get("use_curriculum", False))
    env = make_gym_env(env_config)
    policy = load_policy(config, env, checkpoint, device)
    obs_mean, obs_std = load_obs_stats(bookkeeping) if config["ppo"].get("standardize_obs", True) else (None, None)

    mode = args.mode or eval_cfg.get("mode", "policy_vs_random")
    episodes = int(args.episodes or eval_cfg.get("episodes", 20))
    max_steps = int(args.max_steps or eval_cfg.get("max_steps", 4500))
    deterministic = bool(eval_cfg.get("deterministic", True)) if args.deterministic is None else args.deterministic
    render = bool(eval_cfg.get("render", False)) if args.render is None else args.render
    render_delay = float(args.render_delay if args.render_delay is not None else eval_cfg.get("render_delay", 0.0))
    rng = np.random.default_rng(int(args.seed or eval_cfg.get("seed", 123)))

    episode_rows = []
    total_policy_latency = 0.0
    policy_calls = 0

    for episode in range(episodes):
        obs = env.reset()
        blue_reward = 0.0
        orange_reward = 0.0
        blue_touches = 0
        orange_touches = 0
        blue_goals = 0
        orange_goals = 0
        done = False
        truncated = False
        steps = 0

        while not done and not truncated and steps < max_steps:
            state = env.rlgym_env.state
            teams = [state.cars[env.agent_map[idx]].team_num for idx in range(len(obs))]
            use_policy = np.asarray([uses_policy(mode, team) for team in teams], dtype=bool)

            actions = rng.integers(0, env.action_space.n, size=len(obs), dtype=np.int64)
            if np.any(use_policy):
                normalized = normalize_obs(obs, obs_mean, obs_std)
                t0 = time.perf_counter()
                model_actions = policy_actions(policy, normalized, deterministic)
                total_policy_latency += time.perf_counter() - t0
                policy_calls += 1
                actions[use_policy] = model_actions[use_policy]

            obs, rewards, done, truncated, info = env.step(actions.reshape(-1, 1))
            state = info["state"]

            for idx, reward in enumerate(rewards):
                agent = env.agent_map[idx]
                team = state.cars[agent].team_num
                if team == BLUE:
                    blue_reward += float(reward)
                    blue_touches += int(state.cars[agent].ball_touches > 0)
                else:
                    orange_reward += float(reward)
                    orange_touches += int(state.cars[agent].ball_touches > 0)

            if state.goal_scored:
                if state.scoring_team == BLUE:
                    blue_goals += 1
                elif state.scoring_team == ORANGE:
                    orange_goals += 1

            steps += 1
            if render:
                env.render()
                if render_delay > 0:
                    time.sleep(render_delay)

        episode_rows.append(
            {
                "episode": episode,
                "steps": steps,
                "blue_reward": blue_reward,
                "orange_reward": orange_reward,
                "reward_diff_blue": blue_reward - orange_reward,
                "blue_goals": blue_goals,
                "orange_goals": orange_goals,
                "goal_diff_blue": blue_goals - orange_goals,
                "blue_touch_steps": blue_touches,
                "orange_touch_steps": orange_touches,
                "ended_by_goal": bool(done),
                "truncated": bool(truncated or steps >= max_steps),
            }
        )

    env.close()

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in episode_rows])) if episode_rows else 0.0

    def total(key: str) -> int:
        return int(np.sum([row[key] for row in episode_rows])) if episode_rows else 0

    summary = {
        "mode": mode,
        "checkpoint": str(checkpoint),
        "checkpoint_timesteps": int(bookkeeping.get("cumulative_timesteps", 0)),
        "episodes": episodes,
        "deterministic": deterministic,
        "avg_steps": mean("steps"),
        "avg_blue_reward": mean("blue_reward"),
        "avg_orange_reward": mean("orange_reward"),
        "avg_reward_diff_blue": mean("reward_diff_blue"),
        "blue_goals": total("blue_goals"),
        "orange_goals": total("orange_goals"),
        "goal_diff_blue": total("blue_goals") - total("orange_goals"),
        "avg_blue_touch_steps": mean("blue_touch_steps"),
        "avg_orange_touch_steps": mean("orange_touch_steps"),
        "goal_ended_episodes": total("ended_by_goal"),
        "truncated_episodes": total("truncated"),
        "avg_policy_latency_ms": (total_policy_latency / max(policy_calls, 1)) * 1000.0,
        "episodes_detail": episode_rows,
    }

    output_dir = Path("outputs/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = checkpoint.parent.name
    safe_run_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_name)
    policy_mode = "deterministic" if deterministic else "stochastic"
    output_path = output_dir / f"{mode}_{policy_mode}_{safe_run_name}_{summary['checkpoint_timesteps']}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["output_path"] = str(output_path)
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"mode: {summary['mode']}")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"checkpoint_timesteps: {summary['checkpoint_timesteps']}")
    print(f"episodes: {summary['episodes']}")
    print(f"deterministic: {summary['deterministic']}")
    print(f"avg_blue_reward: {summary['avg_blue_reward']:.3f}")
    print(f"avg_orange_reward: {summary['avg_orange_reward']:.3f}")
    print(f"avg_reward_diff_blue: {summary['avg_reward_diff_blue']:.3f}")
    print(f"blue_goals: {summary['blue_goals']}")
    print(f"orange_goals: {summary['orange_goals']}")
    print(f"goal_diff_blue: {summary['goal_diff_blue']}")
    print(f"avg_blue_touch_steps: {summary['avg_blue_touch_steps']:.3f}")
    print(f"avg_orange_touch_steps: {summary['avg_orange_touch_steps']:.3f}")
    print(f"avg_policy_latency_ms: {summary['avg_policy_latency_ms']:.3f}")
    print(f"output_path: {summary['output_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=["policy_vs_random", "random_vs_policy", "policy_vs_policy", "random_vs_random"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=None)
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=None)
    parser.add_argument("--render-delay", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    print_summary(evaluate(cfg, cli_args))
