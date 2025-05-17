import random
from collections import namedtuple, deque
from typing import Deque, Tuple

import mlx.core as mx

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

    def sample(
        self, batch_size: int
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        batch = random.sample(self.buffer, batch_size)
        states = mx.stack([exp.state for exp in batch])
        actions = mx.array([exp.action for exp in batch], dtype=mx.int32)
        rewards = mx.array([exp.reward for exp in batch], dtype=mx.float32)
        next_states = mx.stack([exp.next_state for exp in batch])
        dones = mx.array([float(exp.done) for exp in batch], dtype=mx.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
