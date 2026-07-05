from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from env_setup import OBS_NAMES, action_count, load_config
from wandb_utils import init_wandb, log_artifact, log_metrics
from world_model import WorldModel, save_world_model


FEATURE_GROUPS = {
    "car_pos": ["car_pos_x", "car_pos_y", "car_pos_z"],
    "car_vel": ["car_vel_x", "car_vel_y", "car_vel_z"],
    "car_rot": ["car_rot_pitch", "car_rot_yaw", "car_rot_roll"],
    "car_ang_vel": ["car_ang_vel_x", "car_ang_vel_y", "car_ang_vel_z"],
    "ball_pos": ["ball_pos_x", "ball_pos_y", "ball_pos_z"],
    "ball_vel": ["ball_vel_x", "ball_vel_y", "ball_vel_z"],
    "rel_ball_pos": ["rel_ball_x", "rel_ball_y", "rel_ball_z"],
    "rel_ball_vel": ["rel_ball_vel_x", "rel_ball_vel_y", "rel_ball_vel_z"],
    "opponent": [
        "opp_pos_x",
        "opp_pos_y",
        "opp_pos_z",
        "opp_vel_x",
        "opp_vel_y",
        "opp_vel_z",
        "opp_boost",
        "opp_on_ground",
    ],
}


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _normalize(obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (obs - mean) / std


def _denormalize(obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return obs * std + mean


def _trajectory_groups(data) -> list[list[int]]:
    if "agent" in data:
        by_key = defaultdict(list)
        for i, (episode, agent) in enumerate(zip(data["episode"], data["agent"])):
            by_key[(int(episode), int(agent))].append(i)
        return [sorted(indices, key=lambda i: data["timestep"][i]) for indices in by_key.values()]

    by_episode = defaultdict(list)
    for i, episode in enumerate(data["episode"]):
        by_episode[int(episode)].append(i)

    groups = []
    for indices in by_episode.values():
        by_timestep = defaultdict(list)
        for idx in indices:
            by_timestep[int(data["timestep"][idx])].append(idx)
        slots = defaultdict(list)
        for timestep in sorted(by_timestep):
            for slot, idx in enumerate(sorted(by_timestep[timestep])):
                slots[slot].append(idx)
        groups.extend(slots.values())
    return groups


def rollout_error(model, data, horizon, obs_mean, obs_std, device, max_samples: int, seed: int) -> float:
    starts = []
    for indices in _trajectory_groups(data):
        indices = sorted(indices, key=lambda i: data["timestep"][i])
        if len(indices) <= horizon:
            continue
        starts.extend((indices, start) for start in range(len(indices) - horizon))

    if not starts:
        return float("nan")

    if max_samples > 0 and len(starts) > max_samples:
        rng = np.random.default_rng(seed + horizon)
        starts = [starts[i] for i in rng.choice(len(starts), size=max_samples, replace=False)]

    obs_idx = [indices[start] for indices, start in starts]
    pred = _normalize(
        torch.as_tensor(data["obs"][obs_idx], dtype=torch.float32, device=device),
        obs_mean,
        obs_std,
    )

    with torch.no_grad():
        for offset in range(horizon):
            action_idx = [indices[start + offset] for indices, start in starts]
            action = torch.as_tensor(data["action"][action_idx], dtype=torch.long, device=device)
            pred = model(pred, action)["next_obs"]

        target_idx = [indices[start + horizon - 1] for indices, start in starts]
        target = torch.as_tensor(data["next_obs"][target_idx], dtype=torch.float32, device=device)
        pred_raw = _denormalize(pred, obs_mean, obs_std)
        return F.mse_loss(pred_raw, target).item()


def feature_metrics(pred_raw, next_raw, obs_raw) -> tuple[dict[str, float], list[tuple[str, float, float]]]:
    model_feature_mse = ((pred_raw - next_raw) ** 2).mean(dim=0).detach().cpu().numpy()
    naive_feature_mse = ((obs_raw - next_raw) ** 2).mean(dim=0).detach().cpu().numpy()

    metrics = {}
    for group_name, names in FEATURE_GROUPS.items():
        indices = [OBS_NAMES.index(name) for name in names]
        metrics[f"{group_name}_mse"] = float(model_feature_mse[indices].mean())
        metrics[f"naive_{group_name}_mse"] = float(naive_feature_mse[indices].mean())

    ranked = sorted(
        (
            (name, float(model_feature_mse[i]), float(naive_feature_mse[i]))
            for i, name in enumerate(OBS_NAMES)
        ),
        key=lambda row: row[1],
        reverse=True,
    )
    return metrics, ranked


def train(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    cfg = config["world_model"]
    run = init_wandb(config, job_type="train_world_model")
    data = np.load(config["rollouts"]["output_path"])
    device = get_device(str(cfg.get("device", "auto")))
    print(f"using device: {device}")

    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(data["next_obs"], dtype=torch.float32, device=device)
    action = torch.as_tensor(data["action"], dtype=torch.long, device=device)
    reward = torch.as_tensor(data["reward"], dtype=torch.float32, device=device)
    done = torch.as_tensor(data["done"], dtype=torch.float32, device=device)

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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    for epoch in range(int(cfg["epochs"])):
        model.train()
        losses = []
        obs_losses = []
        reward_losses = []
        done_losses = []
        for batch_obs, batch_action, batch_next_obs, batch_reward, batch_done in train_loader:
            pred = model(batch_obs, batch_action)
            target_delta = batch_next_obs - batch_obs
            obs_loss = F.smooth_l1_loss(pred["delta"], target_delta)
            reward_loss = F.smooth_l1_loss(pred["reward"], batch_reward)
            done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], batch_done)
            loss = obs_loss + reward_loss + done_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            obs_losses.append(obs_loss.item())
            reward_losses.append(reward_loss.item())
            done_losses.append(done_loss.item())
        epoch_metrics = {
            "train_loss": float(np.mean(losses)),
            "train_delta_loss": float(np.mean(obs_losses)),
            "train_reward_loss": float(np.mean(reward_losses)),
            "train_done_loss": float(np.mean(done_losses)),
        }
        print(f"epoch {epoch + 1}: train_loss={epoch_metrics['train_loss']:.5f}")
        log_metrics(run, epoch_metrics, prefix="world_model", step=epoch + 1)

    model.eval()
    with torch.no_grad():
        val_obs, val_action, val_next_obs, val_reward, val_done = val_ds.tensors
        pred = model(val_obs, val_action)
        val_obs_raw = _denormalize(val_obs, obs_mean, obs_std)
        val_next_raw = _denormalize(val_next_obs, obs_mean, obs_std)
        pred_raw = _denormalize(pred["next_obs"], obs_mean, obs_std)
        group_metrics, ranked_features = feature_metrics(pred_raw, val_next_raw, val_obs_raw)
        naive = F.mse_loss(val_obs, val_next_obs).item()
        metrics = {
            "one_step_mse": F.mse_loss(pred["next_obs"], val_next_obs).item(),
            "naive_one_step_mse": naive,
            "one_step_raw_mse": F.mse_loss(pred_raw, val_next_raw).item(),
            "naive_one_step_raw_mse": F.mse_loss(val_obs_raw, val_next_raw).item(),
            "reward_mae": F.l1_loss(pred["reward"], val_reward).item(),
            "done_accuracy": (
                ((torch.sigmoid(pred["done_logit"]) > 0.5).float() == val_done).float().mean().item()
            ),
        }
        metrics.update(group_metrics)

    max_samples = int(cfg.get("eval_rollout_samples", 1000))
    seed = int(cfg.get("eval_rollout_seed", 123))
    metrics["rollout_5_mse"] = rollout_error(model, data, 5, obs_mean, obs_std, device, max_samples, seed)
    metrics["rollout_10_mse"] = rollout_error(model, data, 10, obs_mean, obs_std, device, max_samples, seed)

    print("largest per-feature one-step raw MSE:")
    for name, model_mse, naive_mse in ranked_features[:8]:
        print(f"  {name}: model={model_mse:.9g}, naive={naive_mse:.9g}")

    save_world_model(cfg["checkpoint_path"], model, obs_mean, obs_std, metrics)
    log_metrics(run, metrics, prefix="world_model")
    if config.get("wandb", {}).get("log_artifacts", True):
        log_artifact(run, cfg["checkpoint_path"], artifact_type="world_model", name="world_model")
    if run is not None:
        run.finish()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    metrics = train(args.config)
    for key, value in metrics.items():
        print(f"{key}: {value:.9g}")
