from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym_ppo.ppo import DiscreteFF

from env_setup import load_config
from eval_ppo import load_bookkeeping, load_obs_stats, resolve_checkpoint, resolve_device


def export_policy(config_path: str, checkpoint_arg: str | None, output_path: str) -> Path:
    config = load_config(config_path)
    checkpoint = resolve_checkpoint(checkpoint_arg or config.get("rlbot", {}).get("checkpoint_folder", "latest"), config)
    bookkeeping = load_bookkeeping(checkpoint)
    device = resolve_device("cpu")

    policy = DiscreteFF(
        input_shape=92,
        n_actions=len(LookupTableAction.make_lookup_table()),
        layer_sizes=list(config["ppo"]["hidden_sizes"]),
        device=device,
    )
    try:
        state_dict = torch.load(checkpoint / "PPO_POLICY.pt", map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint / "PPO_POLICY.pt", map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    arrays = {
        "checkpoint": np.asarray(str(checkpoint)),
        "checkpoint_timesteps": np.asarray(int(bookkeeping.get("cumulative_timesteps", 0)), dtype=np.int64),
        "tick_skip": np.asarray(int(config["env"]["action_repeat"]), dtype=np.int64),
        "deterministic": np.asarray(bool(config.get("rlbot", {}).get("deterministic", True))),
        "lookup_table": LookupTableAction.make_lookup_table().astype(np.float32),
    }

    obs_mean, obs_std = load_obs_stats(bookkeeping) if config["ppo"].get("standardize_obs", True) else (None, None)
    arrays["obs_mean"] = np.asarray(obs_mean if obs_mean is not None else np.zeros(92), dtype=np.float32)
    arrays["obs_std"] = np.asarray(obs_std if obs_std is not None else np.ones(92), dtype=np.float32)

    layer_index = 0
    for module in policy.model:
        if isinstance(module, torch.nn.Linear):
            arrays[f"w{layer_index}"] = module.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"b{layer_index}"] = module.bias.detach().cpu().numpy().astype(np.float32)
            layer_index += 1
    arrays["layer_count"] = np.asarray(layer_index, dtype=np.int64)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="outputs/rlbot/policy.npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = export_policy(args.config, args.checkpoint, args.output)
    print(f"exported_rlbot_policy: {path}")
