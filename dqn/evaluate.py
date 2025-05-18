"""Module for evaluating a trained DQN agent."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from gymnasium.wrappers import RecordVideo

from dqn.atari_env import create_env
from dqn.model import DQN
from dqn.utils import load_model


def evaluate(model_path: str, env_name: str, num_episodes: int, render: bool = True):
    """Evaluate a trained model."""
    render_mode = "human" if render else None

    # Create env to get num_actions. This is a bit inefficient if model is already
    # loaded but needed if we don't store num_actions with the model.
    # A temporary env is created to infer num_actions
    temp_env = create_env(env_name, render_mode=None)
    num_actions = temp_env.action_space.n
    temp_env.close()

    model = DQN(num_actions)
    model, env_name, num_actions = load_model(Path(model_path))
    mx.eval(model.parameters())

    # Create the actual environment for evaluation
    env = create_env(env_name, render_mode=render_mode)

    rewards = []
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            # The state from create_env is already MLX array and preprocessed
            state_batch = mx.expand_dims(state, axis=0)
            q_values = model(state_batch)
            action = mx.argmax(q_values, axis=1)[0].item()

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Update state
            state = next_state

        rewards.append(episode_reward)
        print(f"Episode {i + 1}, Reward: {episode_reward}")

    env.close()
    avg_reward = np.mean(rewards)  # Requires numpy as np
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return rewards


def record_episode_video(
    model_path: str,
    env_name: str,
    output_video_filepath: Path,
    video_length: int = 60000,
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
    episode_reward = 0.

    while not done:
        state_batch = mx.expand_dims(state, axis=0)
        q_values = model(state_batch)
        action = mx.argmax(q_values, axis=1)[0].item()

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
