# Rocket League Model-Based RL Prototype

Small RLGym/RocketSim prototype for testing whether a learned one-step world model can help short-horizon action selection.

## What was kept

- The RLGym v2/RocketSim setup from `gymmy/base_train.py`: `RLGym`, `RocketSimEngine`, `RepeatAction(LookupTableAction())`, kickoff resets, goal/no-touch/timeout endings, and simple goal/touch rewards.
- The RLBot example's useful behavior idea: a tiny chase-ball heuristic baseline.

## What was removed or replaced

- RLBot rendering, quick chat, boost pad tracking, ball prediction, and sequence/flip logic were not copied. They are useful for a live RLBot example, but they do not help the RLGym rollout/world-model loop.
- The `gymmy` training script's large PPO defaults and W&B logging were reduced. This repo starts with local files and small knobs in `config.yaml`.
- `DefaultObs` was replaced with `StructuredObs` so the world-model inputs are fixed, readable, and tied to named state features.

## Files

- `env_setup.py`: RLGym/RocketSim environment plus structured observation builder.
- `policy.py`: random, heuristic, and loadable Torch policy helpers.
- `train_ppo.py`: small wrapper around `rlgym_ppo.Learner`.
- `collect_rollouts.py`: saves `(obs, action, reward, next_obs, done, episode, timestep)` to `outputs/rollouts.npz`.
- `world_model.py`: MLP dynamics model predicting next observation, reward, and done.
- `train_world_model.py`: trains the MLP and reports one-step, 5-step, 10-step, reward, and naive-baseline metrics.
- `eval_model_based.py`: random-shooting planner that executes the first action from the best predicted sequence.
- `config.yaml`: edit horizon, candidate count, action repeat, reward weights, model size, LR, and batch size here.
- `wandb_utils.py`: optional Weights & Biases setup, metric logging, and artifact logging.

## Basic flow

```powershell
python collect_rollouts.py --config config.yaml
python train_world_model.py --config config.yaml
python eval_model_based.py --config config.yaml --mode baseline
python eval_model_based.py --config config.yaml --mode planner
```

Optional PPO baseline:

```powershell
python train_ppo.py --config config.yaml
```

## Weights & Biases

W&B is optional. To turn it on, log in once and set `wandb.enabled: true` in `config.yaml`.

```powershell
python -m wandb login
```

Logged data:

- `collect_rollouts.py`: per-episode reward/length, transition count, rollout dataset artifact.
- `train_world_model.py`: epoch train loss, final model metrics, world-model checkpoint artifact.
- `eval_model_based.py`: average reward, goals for/against, inference latency.
- `train_ppo.py`: uses `rlgym_ppo`'s native W&B logging when enabled.

For local testing without uploading, set `wandb.enabled: true` and `wandb.mode: offline`.
W&B run, cache, and staging files go under ignored `outputs/wandb*` directories by default.

## Current success checks

- Cleaned bot still runs: the live RLBot-specific bot was replaced by a RLGym/RocketSim prototype, so the runnable check is importing the scripts and stepping the RLGym environment.
- Rollout collection: `collect_rollouts.py` writes a compressed NumPy dataset.
- World model: compare `one_step_mse` against `naive_one_step_mse` in `train_world_model.py`. The script also prints raw-space MSE, group MSEs, and the worst individual observation features so it is easier to see what the model is missing.
- Model-based action scoring: `eval_model_based.py` samples candidate action sequences, rolls them forward through the MLP, scores predicted reward, and executes the first action.

## World Model Diagnostics

The model predicts normalized observation deltas, then adds them to the current observation. This makes the "predict what changes" target explicit.

Useful config knobs:

```yaml
world_model:
  device: auto
  eval_rollout_samples: 1000
  eval_rollout_seed: 123
```

- `device: auto` uses CUDA if PyTorch can see it, otherwise CPU.
- `eval_rollout_samples` limits the slower 5-step and 10-step rollout-error checks to a random sample.
- Per-feature diagnostics are printed in raw normalized-observation space, so compare model vs naive values feature by feature.
