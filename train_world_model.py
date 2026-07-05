from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from env_setup import action_count, load_config
from world_model import WorldModel, save_world_model


def _normalize(obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (obs - mean) / std


def _denormalize(obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return obs * std + mean


def rollout_error(model, data, horizon, obs_mean, obs_std) -> float:
    by_episode = defaultdict(list)
    for i, episode in enumerate(data["episode"]):
        by_episode[int(episode)].append(i)

    errors = []
    with torch.no_grad():
        for indices in by_episode.values():
            indices = sorted(indices, key=lambda i: data["timestep"][i])
            if len(indices) <= horizon:
                continue
            for start in range(0, len(indices) - horizon, horizon):
                idx0 = indices[start]
                pred = _normalize(
                    torch.as_tensor(data["obs"][idx0: idx0 + 1], dtype=torch.float32),
                    obs_mean,
                    obs_std,
                )
                for offset in range(horizon):
                    idx = indices[start + offset]
                    action = torch.as_tensor([data["action"][idx]], dtype=torch.long)
                    pred = model(pred, action)["next_obs"]
                target_idx = indices[start + horizon - 1]
                target = torch.as_tensor(
                    data["next_obs"][target_idx: target_idx + 1],
                    dtype=torch.float32,
                )
                pred_raw = _denormalize(pred, obs_mean, obs_std)
                errors.append(F.mse_loss(pred_raw, target).item())
    return float(np.mean(errors)) if errors else float("nan")


def train(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    cfg = config["world_model"]
    data = np.load(config["rollouts"]["output_path"])

    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    next_obs = torch.as_tensor(data["next_obs"], dtype=torch.float32)
    action = torch.as_tensor(data["action"], dtype=torch.long)
    reward = torch.as_tensor(data["reward"], dtype=torch.float32)
    done = torch.as_tensor(data["done"], dtype=torch.float32)

    obs_mean = obs.mean(dim=0, keepdim=True)
    obs_std = obs.std(dim=0, keepdim=True).clamp_min(1e-6)
    obs_n = _normalize(obs, obs_mean, obs_std)
    next_obs_n = _normalize(next_obs, obs_mean, obs_std)

    n = len(obs)
    indices = torch.randperm(n)
    val_n = max(1, int(n * float(cfg["validation_fraction"])))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    train_ds = TensorDataset(obs_n[train_idx], action[train_idx], next_obs_n[train_idx], reward[train_idx], done[train_idx])
    val_ds = TensorDataset(obs_n[val_idx], action[val_idx], next_obs_n[val_idx], reward[val_idx], done[val_idx])
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)

    model = WorldModel(
        obs_dim=obs.shape[1],
        act_dim=action_count(),
        hidden_size=int(cfg["hidden_size"]),
        hidden_layers=int(cfg["hidden_layers"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    for epoch in range(int(cfg["epochs"])):
        model.train()
        losses = []
        for batch_obs, batch_action, batch_next_obs, batch_reward, batch_done in train_loader:
            pred = model(batch_obs, batch_action)
            obs_loss = F.smooth_l1_loss(pred["next_obs"], batch_next_obs)
            reward_loss = F.smooth_l1_loss(pred["reward"], batch_reward)
            done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], batch_done)
            loss = obs_loss + reward_loss + done_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}: train_loss={np.mean(losses):.5f}")

    model.eval()
    with torch.no_grad():
        val_obs, val_action, val_next_obs, val_reward, val_done = val_ds.tensors
        pred = model(val_obs, val_action)
        naive = F.mse_loss(val_obs, val_next_obs).item()
        metrics = {
            "one_step_mse": F.mse_loss(pred["next_obs"], val_next_obs).item(),
            "naive_one_step_mse": naive,
            "reward_mae": F.l1_loss(pred["reward"], val_reward).item(),
            "done_accuracy": (
                ((torch.sigmoid(pred["done_logit"]) > 0.5).float() == val_done).float().mean().item()
            ),
        }

    metrics["rollout_5_mse"] = rollout_error(model, data, 5, obs_mean, obs_std)
    metrics["rollout_10_mse"] = rollout_error(model, data, 10, obs_mean, obs_std)

    save_world_model(cfg["checkpoint_path"], model, obs_mean, obs_std, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    metrics = train(args.config)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
