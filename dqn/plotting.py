"""Module for plotting training metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(
    avg_rewards: list, avg_max_qs: list, save_dir: Path, env_name: str
):
    """
    Plots the average reward per episode and average max Q-value per epoch.

    Saves the plots to the specified directory.
    """
    if not avg_rewards and not avg_max_qs:
        print("No data to plot.")
        return

    epochs = np.arange(1, len(avg_rewards) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_rewards, label="Average Reward per Episode")
    plt.xlabel("Training Epochs")
    plt.ylabel("Average Reward per Episode")
    plt.title(f"Average Reward on {env_name}")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_max_qs, label="Average Max Q-value")
    plt.xlabel("Training Epochs")
    plt.ylabel("Average Action Value (Q)")
    plt.title(f"Average Q on {env_name}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plot_filename = "training_performance_plots.png"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_filepath = save_dir / plot_filename

    try:
        plt.savefig(save_filepath)
        print(f"Training plots saved to {save_filepath}")
    except Exception as e:
        print(f"Error saving plots: {e}")
    plt.close()
