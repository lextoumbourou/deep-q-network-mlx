"""Module for evaluating a trained DQN agent."""

from pathlib import Path

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from gymnasium.wrappers import RecordVideo
from pydantic import BaseModel

from dqn.actions import select_action
from dqn.atari_env import create_env
from dqn.model import DQN
from dqn.utils import load_model


class EvaluationMetrics(BaseModel):
    """Metrics for evaluating a trained model."""

    episode_rewards: list[float]
    avg_episode_reward: float
    total_reward: float
    episodes_completed: int
    avg_max_q: float | None


def evaluate(
    model: nn.Module,
    env: gym.Env,
    eval_steps: int,
    epsilon: float = 0.05,
    eval_states: mx.array | None = None,
):
    """
    Evaluate a trained model for a specified number of steps.

    Args:
        model: The trained model
        env: The environment to evaluate in
        eval_steps: Number of steps to evaluate for
        epsilon: Epsilon value for the ε-greedy policy
        eval_states: States to evaluate the model on

    """
    total_reward = 0
    episode_rewards = []
    current_episode_reward = 0
    episodes_completed = 0

    state, _ = env.reset()

    for _ in range(eval_steps):
        action = select_action(state, model, epsilon, env.action_space.n)

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        current_episode_reward += reward
        total_reward += reward

        if done:
            episodes_completed += 1
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            state, _ = env.reset()
        else:
            state = next_state

    avg_episode_reward = np.mean(episode_rewards)

    avg_max_q = None
    if eval_states is not None:
        q_values = model(eval_states)
        max_q = mx.max(q_values, axis=1)
        avg_max_q = mx.mean(max_q).item()  # type: ignore
        mx.eval(avg_max_q)

    return EvaluationMetrics(
        episode_rewards=episode_rewards,
        avg_episode_reward=avg_episode_reward,
        total_reward=total_reward,
        episodes_completed=episodes_completed,
        avg_max_q=avg_max_q,
    )


def record_episode_video(
    model_path: str,
    env_name: str,
    output_video_filepath: Path,
    video_length: int = 60000,
    epsilon: float = 0.05,
):
    """Load a trained model, run one episode, and save a video."""
    # Create environment to infer num_actions
    # This temporary env doesn't need special render mode
    temp_env = create_env(env_name, render_mode=None)
    num_actions = temp_env.action_space.n
    temp_env.close()

    model = DQN(num_actions)
    model, env_name, num_actions = load_model(Path(model_path))
    mx.eval(model.parameters())

    env_for_recording = create_env(env_name, render_mode="rgb_array")

    video_capture_folder = output_video_filepath.parent
    video_name_prefix_for_recorder = output_video_filepath.stem

    video_capture_folder.mkdir(parents=True, exist_ok=True)

    env = RecordVideo(
        env_for_recording,
        video_folder=str(video_capture_folder),
        name_prefix=video_name_prefix_for_recorder,
        # Record the first episode
        episode_trigger=lambda episode_id: episode_id == 0,
        disable_logger=True,
        video_length=video_length,
    )

    print(f"Starting video recording for one episode of {env_name}...")
    state, _ = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action = select_action(state, model, epsilon, num_actions)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += float(reward)
        state = next_state

    env.close()

    temp_video_path = (
        video_capture_folder / f"{video_name_prefix_for_recorder}-episode-0.mp4"
    )

    if temp_video_path.exists():
        temp_video_path.rename(output_video_filepath)
        print(f"Video saved successfully to: {output_video_filepath}")
        print(f"Episode Reward: {episode_reward}")
    else:
        print(f"Error: Expected video file {temp_video_path} not found.")
