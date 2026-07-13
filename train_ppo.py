from __future__ import annotations

import argparse
from functools import partial

from env_setup import load_config, make_gym_env


def build_ppo_env(config):
    return make_gym_env(config)


def train(config_path: str) -> None:
    from rlgym_ppo import Learner

    config = load_config(config_path)
    cfg = config["ppo"]
    n_proc = int(cfg["n_proc"])
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    env_factory = partial(build_ppo_env, config)

    learner = Learner(
        env_factory,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=int(cfg["batch_size"]),
        policy_layer_sizes=list(cfg["hidden_sizes"]),
        critic_layer_sizes=list(cfg["hidden_sizes"]),
        ts_per_iteration=int(cfg["batch_size"]),
        exp_buffer_size=int(cfg["batch_size"]) * 3,
        ppo_minibatch_size=int(cfg["minibatch_size"]),
        ppo_ent_coef=float(cfg["ent_coef"]),
        policy_lr=float(cfg["policy_lr"]),
        critic_lr=float(cfg["critic_lr"]),
        ppo_epochs=int(cfg["epochs"]),
        standardize_returns=True,
        standardize_obs=bool(cfg.get("standardize_obs", True)),
        save_every_ts=int(cfg.get("save_every_ts", 250000)),
        timestep_limit=int(cfg["timesteps"]),
        checkpoints_save_folder=cfg.get("checkpoints_save_folder"),
        checkpoint_load_folder=cfg.get("checkpoint_load_folder"),
        add_unix_timestamp=bool(cfg.get("add_unix_timestamp", True)),
        device=str(cfg.get("device", "auto")),
        log_to_wandb=bool(config.get("wandb", {}).get("enabled", False)),
        wandb_project_name=config.get("wandb", {}).get("project"),
        wandb_group_name=config.get("wandb", {}).get("group"),
        wandb_run_name=cfg.get("run_name", "ppo-model-free"),
    )
    learner.learn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)
