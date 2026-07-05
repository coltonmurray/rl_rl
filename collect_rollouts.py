from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env_setup import action_count, load_config, make_gym_env
from policy import make_policy, sample_actions
from wandb_utils import init_wandb, log_artifact, log_metrics


def collect(config_path: str) -> Path:
    config = load_config(config_path)
    cfg = config["rollouts"]
    run = init_wandb(config, job_type="collect_rollouts")
    env = make_gym_env(config)
    act_dim = action_count()
    policy = make_policy(
        cfg["policy"],
        obs_dim=env.observation_space.shape[0],
        checkpoint_path=cfg.get("policy_checkpoint"),
    )

    rows = {
        "obs": [],
        "action": [],
        "reward": [],
        "next_obs": [],
        "done": [],
        "episode": [],
        "timestep": [],
    }

    for episode in range(int(cfg["episodes"])):
        obs = env.reset()
        episode_reward = 0.0
        for timestep in range(int(cfg["max_steps"])):
            action = sample_actions(policy, obs, act_dim)
            action_flat = action.reshape(-1)
            next_obs, reward, done, truncated, _info = env.step(action)
            done_flag = bool(done or truncated)

            for agent_idx in range(len(obs)):
                rows["obs"].append(obs[agent_idx].copy())
                rows["action"].append(int(action_flat[agent_idx]))
                rows["reward"].append(float(reward[agent_idx]))
                rows["next_obs"].append(next_obs[agent_idx].copy())
                rows["done"].append(done_flag)
                rows["episode"].append(episode)
                rows["timestep"].append(timestep)
                episode_reward += float(reward[agent_idx])

            obs = next_obs
            if done_flag:
                break
        log_metrics(
            run,
            {
                "episode_reward": episode_reward,
                "episode_steps": timestep + 1,
            },
            prefix="rollouts",
            step=episode,
        )

    env.close()

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=np.asarray(rows["obs"], dtype=np.float32),
        action=np.asarray(rows["action"], dtype=np.int64),
        reward=np.asarray(rows["reward"], dtype=np.float32),
        next_obs=np.asarray(rows["next_obs"], dtype=np.float32),
        done=np.asarray(rows["done"], dtype=np.float32),
        episode=np.asarray(rows["episode"], dtype=np.int64),
        timestep=np.asarray(rows["timestep"], dtype=np.int64),
    )
    log_metrics(
        run,
        {
            "transitions": len(rows["obs"]),
            "episodes": int(cfg["episodes"]),
            "mean_reward": float(np.mean(rows["reward"])) if rows["reward"] else 0.0,
        },
        prefix="rollouts",
    )
    if config.get("wandb", {}).get("log_artifacts", True):
        log_artifact(run, out_path, artifact_type="rollout_dataset", name="rollouts")
    if run is not None:
        run.finish()
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    path = collect(args.config)
    print(f"saved rollouts to {path}")
