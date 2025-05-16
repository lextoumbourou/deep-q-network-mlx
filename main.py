#!/usr/bin/env python
# Deep Q-Network Implementation with MLX
# Based on the architecture described in "Playing Atari with Deep Reinforcement Learning"

import os
import argparse
import random
from collections import namedtuple, deque
from typing import List, Deque
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx import optimizers as optim
import ale_py

import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformObservation,
)
from gymnasium.envs.registration import EnvSpec

gym.register_envs(ale_py)

# Define Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, capacity: int):
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def add(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        batch = random.sample(self.buffer, batch_size)
        states = mx.stack([exp.state for exp in batch])
        actions = mx.array([exp.action for exp in batch], dtype=mx.int32)
        rewards = mx.array([exp.reward for exp in batch], dtype=mx.float32)
        next_states = mx.stack([exp.next_state for exp in batch])
        dones = mx.array([float(exp.done) for exp in batch], dtype=mx.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network as described in the DQN paper."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)  # 9x9x32 = 2592
        self.fc2 = nn.Linear(256, num_actions)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))  # (batch, 84, 84, 4) → (batch, 20, 20, 16)
        x = nn.relu(self.conv2(x))  # → (batch, 9, 9, 32)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # → (batch, 2592)
        x = nn.relu(self.fc1(x))  # → (batch, 256)
        x = self.fc2(x)  # → (batch, num_actions)
        return x


def train_agent(
    env_name: str = "ALE/Pacman-v5",
    num_episodes: int = 10000,
    max_steps_per_episode: int = 10000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay_frames: int = 1_000_000,
    batch_size: int = 32,
    replay_buffer_size: int = 10_000,
    learning_rate: float = 0.00025,
    target_update_freq: int = 10000,
    random_frames: int = 50000,
    train_freq: int = 4,
    render: bool = False,
    save_path: Path = None,
):
    """Train a DQN agent on an Atari environment."""

    # Create environment with Atari preprocessing
    render_mode = "human" if render else None
    env = gym.make(env_name, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, stack_size=4)

    # Transform in MLX friendly format.
    env = TransformObservation(
        env=env,
        func=lambda obs: mx.array(obs.transpose(1, 2, 0) / 255.0, dtype=mx.float32),
        observation_space=env.observation_space,
    )

    num_actions = env.action_space.n

    model = DQN(num_actions)

    target_model = DQN(num_actions)
    target_model.update(model.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)

    replay_buffer = ReplayBuffer(replay_buffer_size)

    epsilon = epsilon_start
    epsilon_interval = epsilon_start - epsilon_min

    frame_count = 0
    episode_rewards = []
    running_rewards = []

    # Collect a fixed set of states using a random policy before training
    num_eval_states = 1000
    eval_states = []
    state, _ = env.reset()
    for _ in range(num_eval_states):
        action = np.random.randint(0, num_actions)
        next_state, _, terminated, truncated, _ = env.step(action)
        eval_states.append(state)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
    eval_states = mx.stack(eval_states)

    avg_max_qs = []

    def compute_loss(states, actions, targets):
        q_values = model(states)

        # Select the Q-values for the actions taken
        masks = mx.eye(num_actions)[actions]
        q_action = mx.sum(q_values * masks, axis=1)

        return nn.losses.huber_loss(q_action, targets, reduction="mean")

    def loss_and_grad(model_params, states, actions, targets):
        model.update(model_params)
        loss = compute_loss(states, actions, targets)
        return loss

    grad_fn = mx.grad(loss_and_grad)

    # Training loop
    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state, _ = env.reset()

        episode_reward = 0

        for step in range(max_steps_per_episode):
            frame_count += 1

            # Select action using epsilon-greedy strategy
            if frame_count < random_frames or np.random.rand() < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                # Use model to get action
                state_batch = mx.expand_dims(state, axis=0)
                q_values = model(state_batch)
                mx.eval(q_values)
                action = mx.argmax(q_values, axis=1)[0].item()

            # Decay epsilon
            if frame_count > random_frames:
                epsilon -= epsilon_interval / epsilon_decay_frames
                epsilon = max(epsilon, epsilon_min)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(Experience(state, action, reward, next_state, done))

            # Update state and accumulate reward
            state = next_state
            episode_reward += reward

            if (
                frame_count > random_frames
                and frame_count % train_freq == 0
                and len(replay_buffer) > batch_size
            ):
                # Sample batch from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    batch_size
                )

                # Compute target Q values
                next_q_values = target_model(next_states)

                max_next_q = mx.max(next_q_values, axis=1)
                targets = rewards + gamma * max_next_q * (1 - dones)

                mx.eval(targets)

                # Get current parameters
                params = model.trainable_parameters()

                # Compute gradients
                grads = grad_fn(params, states, actions, targets)

                mx.eval(grads)

                # Update parameters
                optimizer.update(model, grads)

                # Trying to fix an Resource limit (499000) exceeded error.
                # See: https://github.com/ml-explore/mlx-examples/issues/1262
                mx.eval(params)

            # Update target network
            if frame_count % target_update_freq == 0:
                target_model.update(model.trainable_parameters())
                buffer_size = len(replay_buffer.buffer)
                print(
                        f"Episode {episode + 1}/{num_episodes}, Frame {frame_count}, Epsilon {epsilon:.4f}, Buffer size: {buffer_size}"
                )

            if done:
                break

        # Track episode rewards
        episode_rewards.append(episode_reward)
        avg_reward = (
            np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100
            else np.mean(episode_rewards)
        )
        running_rewards.append(avg_reward)

        # Compute average max Q for eval_states
        q_values = model(eval_states)
        max_q = mx.max(q_values, axis=1)
        avg_max_q = mx.mean(max_q).item()
        mx.eval(avg_max_q)
        avg_max_qs.append(avg_max_q)

        print(
            f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}, Avg Max Q: {avg_max_q:.2f}"
        )

        mx.clear_cache()

        # Save model
        if save_path and (episode + 1) % 100 == 0:
            model.save_weights(f"{save_path}/dqn_episode_{episode + 1}.npz")
            print("========================================================")
            print(
                f"Metal active memory: {mx.metal.get_active_memory() / 1024**3:.2f} GB"
            )
            print(f"Metal cache memory: {mx.metal.get_cache_memory() / 1024**3:.2f} GB")
            print(f"Metal peak memory: {mx.metal.get_peak_memory() / 1024**3:.2f} GB")
            print("========================================================")
            print()

        # Check if environment is solved
        if avg_reward >= 40.0:
            print(f"Environment solved after {episode + 1} episodes!")
            if save_path:
                model.save_weights(f"{save_path}/dqn_solved.npz")
            break

    # Close environment
    env.close()
    return model, episode_rewards, running_rewards, avg_max_qs


def save_model(model, path):
    """Save model weights to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))
    print(f"Model saved to {path}")


def load_model(model, path):
    """Load model weights from a file."""
    path = Path(path)
    model.load_weights(str(path))
    print(f"Model loaded from {path}")
    return model


def evaluate(model, env_name, num_episodes=10, render=True):
    """Evaluate a trained model."""
    render_mode = "human" if render else None
    env = gym.make(env_name, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, stack_size=4)

    rewards = []
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            state_batch = mx.expand_dims(state, axis=0)
            q_values = model(state_batch)
            action = mx.argmax(q_values, axis=1)[0].item()

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {i + 1}, Reward: {episode_reward}")

    env.close()
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return rewards


def main():
    """Main function to parse arguments and start training or evaluation."""
    parser = argparse.ArgumentParser(
        description="Deep Q-Network implementation with MLX"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )
    parser.add_argument(
        "--env", type=str, default="ALE/Breakout-v5", help="Atari environment name"
    )
    parser.add_argument(
        "--episodes", type=int, default=10000, help="Number of episodes for training"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument(
        "--load", type=str, default=None, help="Path to load model weights"
    )
    parser.add_argument(
        "--save", type=str, default="./models", help="Path to save model weights"
    )

    args = parser.parse_args()

    if args.mode == "train":
        model, rewards, avg_rewards, avg_max_qs = train_agent(
            env_name=args.env,
            num_episodes=args.episodes,
            render=args.render,
            save_path=Path(args.save),
        )

    elif args.mode == "eval":
        if not args.load:
            print("Must provide model path for evaluation using --load")
            return

        model = DQN(num_actions=5)
        model = load_model(model, args.load)
        evaluate(model, args.env, num_episodes=10, render=args.render)


if __name__ == "__main__":
    main()
