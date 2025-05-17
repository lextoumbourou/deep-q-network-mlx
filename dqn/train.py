import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx import optimizers as optim
from pathlib import Path
from gymnasium.wrappers import RecordVideo

from .model import DQN
from .replay_buffer import ReplayBuffer, Experience
from .env import create_env
from .utils import save_model, load_model


def loss_fn(model, states, actions, targets):
    q_values = model(states)

    # Select the Q-values for the actions taken
    masks = mx.eye(model.num_actions)[actions]
    q_action = mx.sum(q_values * masks, axis=1)

    return nn.losses.huber_loss(q_action, targets, reduction="mean")


def train_agent(
    env_name: str = "ALE/Breakout-v5",
    num_episodes: int = 10000,
    max_steps_per_episode: int = 10000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay_frames: int = 1_000_000,
    batch_size: int = 32,
    replay_buffer_size: int = 1_000_000,
    learning_rate: float = 0.00025,
    target_update_freq: int = 10000,
    random_frames: int = 50000,
    train_freq: int = 4,
    render: bool = False,
    save_path: Path = None,
):
    """Train a DQN agent on an Atari environment."""

    env = create_env(env_name)

    num_actions = env.action_space.n

    model = DQN(num_actions)
    mx.eval(model.parameters())

    target_model = DQN(num_actions)
    target_model.update(model.parameters())
    mx.eval(target_model.parameters())

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

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps_per_episode):
            frame_count += 1

            if frame_count < random_frames or np.random.rand() < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                state_batch = mx.expand_dims(state, axis=0)
                q_values = model(state_batch)
                action = mx.argmax(q_values, axis=1)[0].item()

            if frame_count > random_frames:
                epsilon -= epsilon_interval / epsilon_decay_frames
                epsilon = max(epsilon, epsilon_min)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            mx.eval(state, action, reward, next_state)
            replay_buffer.add(Experience(state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            if (
                frame_count > random_frames
                and frame_count % train_freq == 0
                and len(replay_buffer) > batch_size
            ):
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

            if frame_count % target_update_freq == 0:
                target_model.update(model.parameters())
                mx.eval(target_model.parameters())
                buffer_size = len(replay_buffer.buffer)
                print(
                    f"Episode {episode + 1}/{num_episodes}, Frame {frame_count}, Epsilon {epsilon:.4f}, Buffer size: {buffer_size}"
                )

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward = (
            np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100
            else np.mean(episode_rewards)
        )
        running_rewards.append(avg_reward)

        q_values = model(eval_states)
        max_q = mx.max(q_values, axis=1)
        avg_max_q = mx.mean(max_q).item()
        mx.eval(avg_max_q)
        avg_max_qs.append(avg_max_q)

        print(
            f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}, Avg Max Q: {avg_max_q:.2f}"
        )

        mx.clear_cache()

        if save_path and (episode + 1) % 100 == 0:
            # Use save_model from .utils
            save_model(model, f"{save_path}/dqn_episode_{episode + 1}.npz")
            print("========================================================")
            print(
                f"Metal active memory: {mx.metal.get_active_memory() / 1024**3:.2f} GB"
            )
            print(f"Metal cache memory: {mx.metal.get_cache_memory() / 1024**3:.2f} GB")
            print(f"Metal peak memory: {mx.metal.get_peak_memory() / 1024**3:.2f} GB")
            print("========================================================")
            print()

        if avg_reward >= 40.0:
            print(f"Environment solved after {episode + 1} episodes!")
            if save_path:
                save_model(model, f"{save_path}/dqn_solved.npz")
            break

    env.close()
    return model, episode_rewards, running_rewards, avg_max_qs
