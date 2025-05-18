"""Module for training the DQN agent."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from dqn.atari_env import create_env
from dqn.model import DQN
from dqn.plotting import plot_training_results, save_metrics
from dqn.replay_buffer import Experience, ReplayBuffer
from dqn.utils import save_model


def loss_fn(model, states, actions, targets):
    """Compute the Q-learning loss."""
    q_values = model(states)

    # Select the Q-values for the actions taken
    masks = mx.eye(model.num_actions)[actions]
    q_action = mx.sum(q_values * masks, axis=1)

    return nn.losses.huber_loss(q_action, targets, reduction="mean")


def train_agent(
    env_name: str,
    save_path: Path,
    train_steps: int,
    steps_per_epoch: int,
    frameskip: int,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay_frames: int = 1_000_000,
    batch_size: int = 32,
    replay_buffer_size: int = 100_000,
    learning_rate: float = 5e-4,
    target_update_freq: int = 10000,
):
    """Train a DQN agent on an Atari environment."""
    env = create_env(env_name, frameskip=frameskip)

    num_actions: int = int(env.action_space.n)  # type: ignore

    model = DQN(num_actions)
    mx.eval(model.parameters())

    target_model = DQN(num_actions)
    target_model.update(model.parameters())
    mx.eval(target_model.parameters())

    optimizer = optim.RMSprop(learning_rate=learning_rate)

    replay_buffer = ReplayBuffer(replay_buffer_size)

    epsilon = epsilon_start
    epsilon_interval = epsilon_start - epsilon_min

    frame_count = 0
    episode_rewards = []
    current_episode = 0
    updates_performed = 0

    # Collect a fixed set of states using a random policy before training
    num_eval_states = 1000
    eval_states_l = []
    state, _ = env.reset()
    for _ in range(num_eval_states):
        _action = np.random.randint(0, num_actions)
        next_state, _, terminated, truncated, _ = env.step(_action)
        eval_states_l.append(state)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
    eval_states = mx.stack(eval_states_l)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print(f"Starting training for {train_steps:,} timesteps...")

    state, _ = env.reset()
    episode_reward = 0.0
    epoch = 0
    avg_max_q = 0.0
    avg_reward = 0.0

    epoch_avg_rewards = []
    epoch_avg_max_qs = []

    pbar = tqdm(total=train_steps, desc="Training")
    last_frame_count = 0

    while frame_count < train_steps:
        frame_count += 1

        action: int
        if np.random.rand() < epsilon:
            action = int(np.random.randint(0, num_actions))
        else:
            state_batch = mx.expand_dims(state, axis=0)
            q_values = model(state_batch)
            action = int(mx.argmax(q_values, axis=1)[0].item())  # type: ignore

        epsilon -= epsilon_interval / epsilon_decay_frames
        epsilon = max(epsilon, epsilon_min)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        mx.eval(state, action, reward, next_state)
        replay_buffer.add(Experience(state, action, reward, next_state, done))

        state = next_state
        episode_reward += float(reward)

        if len(replay_buffer) > batch_size:
            states, actions, rewards_batch, next_states_batch, dones_batch = (
                replay_buffer.sample(batch_size)
            )

            next_q_values = target_model(next_states_batch)
            max_next_q = mx.max(next_q_values, axis=1)
            targets = rewards_batch + gamma * max_next_q * (1 - dones_batch)

            mx.eval(states, actions, targets)
            loss, grads = loss_and_grad_fn(model, states, actions, targets)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), loss, optimizer.state)

            updates_performed += 1

        if frame_count % target_update_freq == 0:
            target_model.update(model.parameters())
            mx.eval(target_model.parameters())

        if done:
            episode_rewards.append(episode_reward)

            avg_reward = float(
                np.mean(episode_rewards[-100:])
                if len(episode_rewards) > 100
                else np.mean(episode_rewards)
            )

            q_values = model(eval_states)
            max_q = mx.max(q_values, axis=1)
            avg_max_q = mx.mean(max_q).item()  # type: ignore
            mx.eval(avg_max_q)

            current_episode += 1

            state, _ = env.reset()
            episode_reward = 0

        if frame_count % steps_per_epoch == 0:
            epoch_avg_rewards.append(avg_reward)
            epoch_avg_max_qs.append(avg_max_q)
            if save_path:
                save_model(
                    model,
                    save_path / env_name / f"epoch_{epoch + 1}.safetensors",
                    env_name=env_name,
                    num_actions=num_actions,
                )
                save_metrics(
                    epoch_avg_rewards,
                    epoch_avg_max_qs,
                    save_path / env_name,
                    env_name,
                )

            epoch += 1

        # Update progress bar every 1000 frames
        if frame_count % 1000 == 0:
            pbar.set_postfix(
                {
                    "Avg Reward": f"{avg_reward:.2f}",
                    "Epsilon": f"{epsilon:.3f}",
                    "Avg Max Q": f"{avg_max_q:.2f}",
                    "Epoch": f"{epoch:,}",
                    "Episode": f"{current_episode:,}",
                }
            )
            pbar.update(frame_count - last_frame_count)
            last_frame_count = frame_count

    pbar.close()
    env.close()

    if save_path:
        plot_training_results(
            epoch_avg_rewards,
            epoch_avg_max_qs,
            save_path / env_name,
            env_name,
        )
