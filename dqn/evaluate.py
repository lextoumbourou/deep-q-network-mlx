"""Module for evaluating a trained DQN agent."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from gymnasium.wrappers import RecordVideo

from dqn.atari_env import create_env
from dqn.model import DQN
from dqn.utils import load_model
from dqn.actions import select_action


def evaluate(
    model_path: str,
    env_name: str,
    eval_steps: int,
    render: bool = True,
    epsilon: float = 0.05,
):
    """Evaluate a trained model for a specified number of steps.

    Args:
        model_path: Path to the saved model weights
        env_name: Name of the Atari environment
        eval_steps: Number of steps to evaluate for
        render: Whether to render the environment
        epsilon: Epsilon value for the ε-greedy policy

    """
    render_mode = "human" if render else None

    # Create env to get num_actions
    temp_env = create_env(env_name, render_mode=None)
    num_actions = temp_env.action_space.n
    temp_env.close()

    model = DQN(num_actions)
    model, env_name, num_actions = load_model(Path(model_path))
    mx.eval(model.parameters())

    # Create the actual environment for evaluation
    env = create_env(env_name, render_mode=render_mode)

    total_reward = 0
    episode_rewards = []
    current_episode_reward = 0
    episodes_completed = 0

    state, _ = env.reset()

    for _ in range(eval_steps):
        action = select_action(state, model, epsilon, num_actions)

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        current_episode_reward += reward
        total_reward += reward

        if done:
            episodes_completed += 1
            episode_rewards.append(current_episode_reward)
            print(f"Episode {episodes_completed}, Reward: {current_episode_reward}")
            current_episode_reward = 0
            state, _ = env.reset()
        else:
            state = next_state

    env.close()

    if episode_rewards:
        avg_episode_reward = np.mean(episode_rewards)
        print(f"\nEvaluation Results over {eval_steps} steps:")
        print(f"Episodes Completed: {episodes_completed}")
        print(f"Average Episode Reward: {avg_episode_reward:.2f}")
        print(f"Total Reward: {total_reward:.2f}")

    return episode_rewards


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
