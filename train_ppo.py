from __future__ import annotations

import argparse

from env_setup import load_config, make_gym_env


def train(config_path: str) -> None:
    from rlgym_ppo import Learner

    config = load_config(config_path)
    cfg = config["ppo"]
    n_proc = int(cfg["n_proc"])
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    def env_factory():
        return make_gym_env(config)

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
        standardize_obs=False,
        save_every_ts=250000,
        timestep_limit=int(cfg["timesteps"]),
        log_to_wandb=False,
    )
    learner.learn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)
