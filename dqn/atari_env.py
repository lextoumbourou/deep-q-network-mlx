"""Module for creating and preprocessing Atari environments."""

import ale_py
import gymnasium as gym
import mlx.core as mx
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformObservation,
)

gym.register_envs(ale_py)


def create_env(
    env_name: str,
    render_mode: str | None = None,
    stack_size: int = 4,
    frame_skip: int = 4,
) -> gym.Env:
    """Creates and preprocesses an Atari environment."""
    output = gym.make(env_name, frameskip=1, render_mode=render_mode)
    output = AtariPreprocessing(output, frame_skip=frame_skip)
    output = FrameStackObservation(output, stack_size=stack_size)

    # Transform in MLX friendly format.
    output = TransformObservation(
        env=output,
        func=lambda obs: mx.array(obs.transpose(1, 2, 0), dtype=mx.int8),
        observation_space=output.observation_space,
    )
    return output
