from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from env_setup import action_count, load_config, make_gym_env
from policy import make_policy, sample_actions
from world_model import load_world_model


def _normalize(obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (obs - mean) / std


@torch.no_grad()
def plan_action(model, obs, obs_mean, obs_std, horizon: int, candidates: int, discount: float) -> np.ndarray:
    act_dim = model.act_dim
    planned = []
    for agent_obs in obs:
        obs_t = torch.as_tensor(agent_obs[None, :], dtype=torch.float32)
        obs_t = _normalize(obs_t, obs_mean, obs_std).repeat(candidates, 1)
        sequences = torch.randint(0, act_dim, size=(candidates, horizon))
        score = torch.zeros(candidates)
        pred_obs = obs_t
        alive = torch.ones(candidates)

        for t in range(horizon):
            pred = model(pred_obs, sequences[:, t])
            score += alive * (discount ** t) * pred["reward"]
            alive *= 1.0 - (torch.sigmoid(pred["done_logit"]) > 0.5).float()
            pred_obs = pred["next_obs"]

        best = int(torch.argmax(score).item())
        planned.append(int(sequences[best, 0].item()))
    return np.asarray(planned, dtype=np.int64).reshape(-1, 1)


def evaluate(config_path: str, mode: str) -> dict[str, float]:
    config = load_config(config_path)
    plan_cfg = config["planning"]
    env = make_gym_env(config)
    act_dim = action_count()
    policy = make_policy(
        plan_cfg["policy"],
        obs_dim=env.observation_space.shape[0],
        checkpoint_path=config["rollouts"].get("policy_checkpoint"),
    )

    if mode == "planner":
        model, obs_mean, obs_std, _metrics = load_world_model(config["world_model"]["checkpoint_path"])
    else:
        model = obs_mean = obs_std = None

    episode_returns = []
    latencies = []
    goals_for = 0
    goals_against = 0

    for _episode in range(int(plan_cfg["execute_episodes"])):
        obs = env.reset()
        total_reward = 0.0
        for _step in range(int(plan_cfg["max_steps"])):
            start = time.perf_counter()
            if mode == "planner":
                action = plan_action(
                    model,
                    obs,
                    obs_mean,
                    obs_std,
                    int(plan_cfg["horizon"]),
                    int(plan_cfg["candidate_sequences"]),
                    float(plan_cfg["discount"]),
                )
            else:
                action = sample_actions(policy, obs, act_dim)
            latencies.append(time.perf_counter() - start)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(np.mean(reward))
            state = info.get("state")
            if getattr(state, "goal_scored", False):
                if getattr(state, "scoring_team", None) == 0:
                    goals_for += 1
                else:
                    goals_against += 1
            if done or truncated:
                break
        episode_returns.append(total_reward)

    env.close()
    return {
        "avg_reward": float(np.mean(episode_returns)),
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "avg_latency_ms": float(np.mean(latencies) * 1000.0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["baseline", "planner"], default="planner")
    args = parser.parse_args()
    metrics = evaluate(args.config, args.mode)
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
