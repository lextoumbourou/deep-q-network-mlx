"""Utility functions for the DQN agent, including model saving and loading."""

import json
from pathlib import Path

import mlx.nn as nn

from .model import DQN


def save_model(model: nn.Module, path: Path, env_name: str, num_actions: int):
    """Save model weights and environment name to a file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(path_obj))

    # Save metadata as JSON
    with Path(path_obj.with_suffix(".json")).open("w") as f:
        json.dump({"env_name": env_name, "num_actions": num_actions}, f)


def load_model(path: Path):
    """Load model weights and environment name from a file."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Model file not found: {path_obj}")

    metadata_path = path_obj.with_suffix(".json")
    with Path(metadata_path).open() as f:
        data = json.load(f)

    model = DQN(data["num_actions"])
    model.load_weights(str(path_obj))

    print(f"Model loaded from {path_obj}, Environment: {data['env_name']}")
    return model, data["env_name"], data["num_actions"]
