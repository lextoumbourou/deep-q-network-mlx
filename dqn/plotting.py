"""Module for plotting training metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_metrics(avg_rewards: list, avg_max_qs: list, save_dir: Path, env_name: str):
    """Save training metrics to a JSON file."""
    metrics = {
        "avg_rewards": avg_rewards,
        "avg_max_qs": avg_max_qs,
        "env_name": env_name,
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "training_metrics.json"

    with Path(metrics_file).open("w") as f:
        json.dump(metrics, f)


def load_metrics(metrics_file: Path) -> tuple[list, list, str]:
    """Load training metrics from a JSON file."""
    with Path(metrics_file).open() as f:
        metrics = json.load(f)
    return metrics["avg_rewards"], metrics["avg_max_qs"], metrics["env_name"]


def plot_rewards(avg_rewards: list, save_dir: Path, env_name: str):
    """Plot average rewards per episode."""
    if not avg_rewards:
        print("No reward data to plot.")
        return

    epochs = np.arange(1, len(avg_rewards) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_rewards, label="Average Reward per Episode")
    plt.xlabel("Training Epochs")
    plt.ylabel("Average Reward per Episode")
    plt.title(f"Average Reward on {env_name}")
    plt.legend()
    plt.grid(True)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_filepath = save_dir / "rewards_plot.png"

    try:
        plt.savefig(save_filepath)
        print(f"Rewards plot saved to {save_filepath}")
    except Exception as e:
        print(f"Error saving rewards plot: {e}")
    plt.close()


def plot_q_values(avg_max_qs: list, save_dir: Path, env_name: str):
    """Plot average max Q-values per epoch."""
    if not avg_max_qs:
        print("No Q-value data to plot.")
        return

    epochs = np.arange(1, len(avg_max_qs) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_max_qs, label="Average Max Q-value")
    plt.xlabel("Training Epochs")
    plt.ylabel("Average Action Value (Q)")
    plt.title(f"Average Q on {env_name}")
    plt.legend()
    plt.grid(True)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_filepath = save_dir / "q_values_plot.png"

    try:
        plt.savefig(save_filepath)
        print(f"Q-values plot saved to {save_filepath}")
    except Exception as e:
        print(f"Error saving Q-values plot: {e}")
    plt.close()


def plot_training_results(
    avg_rewards: list, avg_max_qs: list, save_dir: Path, env_name: str
):
    """Plot both rewards and Q-values as separate plots."""
    plot_rewards(avg_rewards, save_dir, env_name)
    plot_q_values(avg_max_qs, save_dir, env_name)
