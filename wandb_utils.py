from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def wandb_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("wandb", {}).get("enabled", False))


def init_wandb(config: Dict[str, Any], job_type: str, name: str | None = None):
    if not wandb_enabled(config):
        return None

    wandb_cfg = config.get("wandb", {})
    wandb_dir = Path(wandb_cfg.get("dir", "outputs/wandb"))
    cache_dir = Path(wandb_cfg.get("cache_dir", "outputs/wandb_cache"))
    data_dir = Path(wandb_cfg.get("data_dir", "outputs/wandb_data"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("WANDB_DATA_DIR", str(data_dir))

    import wandb

    return wandb.init(
        project=wandb_cfg.get("project", "rl_rl"),
        entity=wandb_cfg.get("entity") or None,
        group=wandb_cfg.get("group") or None,
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags") or [],
        job_type=job_type,
        name=name,
        config=config,
    )


def log_metrics(run, metrics: Dict[str, float], prefix: str | None = None, step: int | None = None) -> None:
    if run is None:
        return
    if prefix:
        metrics = {f"{prefix}/{key}": value for key, value in metrics.items()}
    run.log(metrics, step=step)


def log_artifact(run, path: str | Path, artifact_type: str, name: str | None = None) -> None:
    if run is None:
        return
    path = Path(path)
    if not path.exists():
        return

    import wandb

    try:
        artifact = wandb.Artifact(name or path.stem, type=artifact_type)
        artifact.add_file(str(path))
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"wandb artifact logging skipped for {path}: {exc}")
