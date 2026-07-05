from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from env_setup import OBS_NAMES, action_count


def mlp(sizes: Iterable[int], activation=nn.Tanh, final_activation=None) -> nn.Sequential:
    layers = []
    sizes = list(sizes)
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())
    return nn.Sequential(*layers)


class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes, act_dim])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits = self(obs_t)
        if deterministic:
            return torch.argmax(logits, dim=-1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()


class HeuristicChasePolicy:
    """Tiny chase-ball baseline using the structured observation."""

    def __init__(self):
        self._actions = self._make_action_lookup()
        self._yaw_idx = OBS_NAMES.index("car_rot_yaw")
        self._rel_x_idx = OBS_NAMES.index("rel_ball_x")
        self._rel_y_idx = OBS_NAMES.index("rel_ball_y")

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if obs.ndim == 1:
            return np.asarray(self._act_one(obs), dtype=np.int64)
        return np.asarray([self._act_one(row) for row in obs], dtype=np.int64)

    def _act_one(self, obs: np.ndarray) -> int:
        rel_x = obs[self._rel_x_idx]
        rel_y = obs[self._rel_y_idx]
        yaw = obs[self._yaw_idx] * np.pi
        target_angle = np.arctan2(rel_y, rel_x)
        error = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))
        steer = int(np.sign(error)) if abs(error) > 0.15 else 0
        boost = int(abs(error) < 0.3)
        return self._actions[(1, steer, boost, 0)]

    def _make_action_lookup(self) -> dict[tuple[int, int, int, int], int]:
        from rlgym.rocket_league.action_parsers import LookupTableAction

        table = LookupTableAction.make_lookup_table()
        lookup = {}
        for idx, action in enumerate(table):
            throttle, steer, _pitch, _yaw, _roll, _jump, boost, handbrake = action
            key = (int(throttle), int(steer), int(boost), int(handbrake))
            lookup.setdefault(key, idx)
        return lookup


def load_policy(path: str | Path, obs_dim: int, hidden_sizes=(256, 256)) -> DiscretePolicy:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    policy = DiscretePolicy(
        obs_dim=obs_dim,
        act_dim=checkpoint.get("act_dim", action_count()),
        hidden_sizes=checkpoint.get("hidden_sizes", hidden_sizes),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def make_policy(name: str, obs_dim: int, checkpoint_path: str | None = None):
    if name == "random":
        return None
    if name == "heuristic":
        return HeuristicChasePolicy()
    if name == "torch":
        if checkpoint_path is None:
            raise ValueError("policy_checkpoint is required for policy='torch'")
        return load_policy(checkpoint_path, obs_dim)
    raise ValueError(f"Unknown policy: {name}")


def sample_actions(policy, obs: np.ndarray, act_dim: int) -> np.ndarray:
    if policy is None:
        actions = np.random.randint(0, act_dim, size=(len(obs),), dtype=np.int64)
    else:
        actions = policy.act(obs)
    return np.asarray(actions, dtype=np.int64).reshape(-1, 1)
