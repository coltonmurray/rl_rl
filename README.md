# Rocket League PPO Baseline

Small RLGym/RocketSim training project for a model-free Rocket League bot.

The current direction is a serious PPO baseline. This repo intentionally avoids
Necto's full distributed `rocket-learn` stack for now, but keeps the pieces that
fit a local, readable training loop:

- RLGym v2 with RocketSim.
- 90-action `LookupTableAction`.
- `action_repeat: 8`.
- `DefaultObs` with normalized Rocket League state.
- A compact Necto-inspired dense reward.
- `rlgym_ppo` for local PPO training.
- Native W&B logging through `rlgym_ppo`.

## Files

- `train_ppo.py`: PPO entrypoint.
- `eval_ppo.py`: checkpoint evaluator for policy-vs-random, policy-vs-policy, and random baselines.
- `env_setup.py`: RLGym/RocketSim environment factory.
- `rewards.py`: `NectoLiteReward`, the dense reward used for training.
- `config.yaml`: training, environment, reward, and W&B knobs.
- `outputs/`: ignored directory for checkpoints and generated run files.

## What Was Removed

The previous model-based prototype files were removed because they are no
longer part of this refactor:

- Rollout collection and world-model training scripts.
- The learned dynamics model and random-shooting planner.
- Planner sweeps and planner-vs-baseline evaluation.
- RLBot GUI adapter/config files for the planner.
- Custom W&B helper code, since `rlgym_ppo` now owns PPO logging.

Keeping those files around made the project look like it still had two active
architectures. Right now the active architecture is model-free PPO.

## Train

From this folder:

```powershell
python train_ppo.py --config config.yaml
```

The main knobs are in `config.yaml`:

```yaml
env:
  action_repeat: 8
  spawn_opponents: true
  team_size: 1

ppo:
  run_name: ppo-necto-lite
  n_proc: 4
  timesteps: 2000000
  batch_size: 50000
  minibatch_size: 10000
  hidden_sizes: [512, 512, 256]
  checkpoint_load_folder:
  checkpoints_save_folder: outputs/ppo/necto-lite
```

`checkpoint_load_folder:` is empty by default so a run starts fresh. Set it to a
checkpoint directory when you want to resume.

Training can mix kickoff starts with simple chase/recovery starts:

```yaml
state_curriculum:
  enabled: true
  kickoff_probability: 0.6
```

This keeps kickoff practice while adding off-kickoff states where the car must
turn, recover, and drive back to the ball.

While training, `rlgym_ppo` supports console controls:

- `q`: quit cleanly.
- `c`: save a checkpoint.
- `p`: pause or resume.

## Evaluate

Evaluate the latest checkpoint under `outputs/ppo`:

```powershell
python eval_ppo.py --config config.yaml
```

Useful modes:

```powershell
python eval_ppo.py --config config.yaml --mode policy_vs_random --episodes 20
python eval_ppo.py --config config.yaml --mode policy_vs_policy --episodes 20
python eval_ppo.py --config config.yaml --mode random_vs_random --episodes 20
```

The default is deterministic `policy_vs_random`: blue uses the checkpointed
policy and orange samples random actions. The script prints reward, goals,
touch-step counts, and policy latency, then writes the full episode details to
`outputs/eval/`.

Evaluation uses kickoff starts by default even when training curriculum is
enabled. Set `eval.use_curriculum: true` to evaluate the mixed reset
distribution directly.

## RLBot GUI

Export the latest PPO actor to a NumPy-only runtime file:

```powershell
python export_rlbot_policy.py --config config.yaml
```

The RLBot adapter loads `outputs/rlbot/policy.npz` and approximates the training
`DefaultObs` from the live `GameTickPacket`. It intentionally avoids importing
training-time packages because RLBotGUIX launches bots with its own Python.

In the RLBot GUI, add this bot config:

```text
C:\Users\Colto\dev\rl_dev\rl_rl\rlbot_bot.cfg
```

Or load the match config:

```text
C:\Users\Colto\dev\rl_dev\rl_rl\rlbot.cfg
```

The adapter is intended for 1v1. Re-run the export command whenever you want the
GUI to use a newer checkpoint.

If RLBotGUIX reports missing `baseline_bot.cfg` or `planner_bot.cfg`, remove
the stale old entries from the GUI. Compatibility alias files with those names
are also present so old cached match setups do not fail immediately.

## Reward

`NectoLiteReward` is intentionally smaller than Necto's original reward stack.
It rewards or penalizes:

- scoring and conceding,
- moving the ball toward the opponent net,
- staying close and aligned with the ball,
- touches, touch acceleration, and aerial touch height,
- boost gain/spend,
- getting demoed,
- every step with `step_penalty`,
- non-goal truncations with `no_touch_timeout_penalty`,
- optional team-spirit and opponent-punish blending.

Tune the reward in `necto_lite_reward` inside `config.yaml`.

The current `touch-v2` settings are intentionally contact-first: passive
distance/alignment rewards are tiny, touches are much larger, and timing out
without a goal is penalized.

## W&B

W&B is enabled with:

```yaml
wandb:
  enabled: true
  project: rl_rl
  group:
```

Log in once if needed:

```powershell
python -m wandb login
```

Set `wandb.enabled: false` for local runs without W&B.

## Current Goal

The first milestone is not fancy architecture. It is a stable model-free
baseline that can train for many iterations, save checkpoints, and produce
learning curves we can compare before deciding whether to add self-play,
league training, larger policies, or a distributed setup.
