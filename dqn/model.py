"""Defines the DQN model architecture."""

import mlx.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network as described in the DQN paper.

    The input to the neural network consists of an 84x84x4 image.
    The first hidden layer convolves 16 8x8 filters with stride 4 and applies ReLU.
    The second hidden layer convolves 32 4x4 filters with stride 2, also with ReLU.
    The final hidden layer is fully connected with 256 ReLU units.
    The output layer is a fully connected linear layer with one output per action.
    The number of valid actions typically varied between 4 and 18.
    """

    def __init__(self, num_actions: int):
        """Initialize the DQN model layers."""
        super().__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)  # 9x9x32 = 2592
        self.fc2 = nn.Linear(256, num_actions)

    def __call__(self, x):
        """Forward pass through the network."""
        x = nn.relu(self.conv1(x))  # (batch, 84, 84, 4) → (batch, 20, 20, 16)
        x = nn.relu(self.conv2(x))  # → (batch, 9, 9, 32)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # → (batch, 2592)
        x = nn.relu(self.fc1(x))  # → (batch, 256)
        x = self.fc2(x)  # → (batch, num_actions)
        return x
