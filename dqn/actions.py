"""Module for selecting actions using an ε-greedy policy."""

import mlx.core as mx
import numpy as np


def select_action(state, model, epsilon: float, num_actions: int) -> int:
    """
    Return an action using an ε-greedy policy.

    With probability ε a random action is returned; otherwise the greedy
    action according to the current Q-network is chosen.
    """
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, num_actions))

    state_batch = mx.expand_dims(state, axis=0)
    q_values = model(state_batch)
    return int(mx.argmax(q_values, axis=1)[0].item())
