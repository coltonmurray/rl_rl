from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        layers = []
        in_dim = obs_dim + act_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.next_delta = nn.Linear(in_dim, obs_dim)
        self.reward = nn.Linear(in_dim, 1)
        self.done_logit = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_oh = F.one_hot(action.long(), num_classes=self.act_dim).float()
        x = torch.cat([obs.float(), action_oh], dim=-1)
        h = self.trunk(x)
        return {
            "next_obs": obs + self.next_delta(h),
            "reward": self.reward(h).squeeze(-1),
            "done_logit": self.done_logit(h).squeeze(-1),
        }


def save_world_model(
    path: str | Path,
    model: WorldModel,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    metrics: Dict[str, float],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": model.obs_dim,
            "act_dim": model.act_dim,
            "hidden_size": model.hidden_size,
            "hidden_layers": model.hidden_layers,
            "obs_mean": obs_mean.cpu(),
            "obs_std": obs_std.cpu(),
            "metrics": metrics,
        },
        path,
    )


def load_world_model(path: str | Path) -> tuple[WorldModel, torch.Tensor, torch.Tensor, Dict[str, float]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model = WorldModel(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
        hidden_size=checkpoint.get("hidden_size", 256),
        hidden_layers=checkpoint.get("hidden_layers", 2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["obs_mean"], checkpoint["obs_std"], checkpoint.get("metrics", {})
