import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx import optimizers as optim
from pathlib import Path
from tqdm import tqdm

from .model import DQN
from .replay_buffer import ReplayBuffer, Experience
from .env import create_env
from .utils import save_model


def loss_fn(model, states, actions, targets):
    q_values = model(states)

    # Select the Q-values for the actions taken
    masks = mx.eye(model.num_actions)[actions]
    q_action = mx.sum(q_values * masks, axis=1)

    return nn.losses.huber_loss(q_action, targets, reduction="mean")


def train_agent(
    env_name: str,
    save_path: Path,
    total_steps: int,
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

    num_actions = int(env.action_space.n)

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

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print(f"Starting training for {total_steps:,} timesteps...")

    state, _ = env.reset()
    episode_reward = 0
    epoch = 0
    avg_max_q = 0.0
    avg_reward = 0.0

    pbar = tqdm(total=total_steps, desc="Training")
    last_frame_count = 0

    while frame_count < total_steps:
        frame_count += 1

        if np.random.rand() < epsilon:
            action = np.random.randint(0, num_actions)
        else:
            state_batch = mx.expand_dims(state, axis=0)
            q_values = model(state_batch)
            action = mx.argmax(q_values, axis=1)[0].item()

        epsilon -= epsilon_interval / epsilon_decay_frames
        epsilon = max(epsilon, epsilon_min)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        mx.eval(state, action, reward, next_state)
        replay_buffer.add(Experience(state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward

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

            avg_reward = (
                np.mean(episode_rewards[-100:])
                if len(episode_rewards) > 100
                else np.mean(episode_rewards)
            )

            q_values = model(eval_states)
            max_q = mx.max(q_values, axis=1)
            avg_max_q = mx.mean(max_q).item()
            mx.eval(avg_max_q)

            current_episode += 1

            state, _ = env.reset()
            episode_reward = 0

        if frame_count % steps_per_epoch == 0:
            print(f"Epoch {epoch + 1}/{total_steps // steps_per_epoch} completed")
            if save_path:
                save_model(
                    model,
                    save_path / env_name / f"epoch_{epoch + 1}.safetensors",
                    env_name=env_name,
                    num_actions=num_actions,
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
