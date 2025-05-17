import gymnasium as gym
import ale_py  # Required for gym.register_envs
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformObservation,
)
import mlx.core as mx

gym.register_envs(ale_py)


def create_env(
    env_name: str, render_mode: str = None, frameskip: int = 1, stack_size: int = 4
):
    """
    Creates and preprocesses an Atari environment and transforms to MLX friendly format.
    """
    env = gym.make(env_name, frameskip=frameskip, render_mode=render_mode)
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, stack_size=stack_size)

    # Transform in MLX friendly format.
    env = TransformObservation(
        env=env,
        func=lambda obs: mx.array(obs.transpose(1, 2, 0) / 255.0, dtype=mx.float32),
        observation_space=env.observation_space,
    )
    return env
