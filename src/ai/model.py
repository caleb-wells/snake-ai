"""
Deep Q-Network (DQN) model implementation for reinforcement learning.
This module defines a neural network architecture used in DQN algorithms
for Q-value approximation.
"""

import torch.nn.functional as f
from torch import Tensor, nn


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture.

    A fully connected neural network that maps state observations to Q-values
    for each possible action in the environment.

    Attributes:
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Second fully connected layer
        fc3 (nn.Linear): Output layer
    """

    def __init__(
        self, input_size: int = 11, hidden_size: int = 128, output_size: int = 3
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor representing the state observation

        Returns:
            Tensor: Q-values for each possible action
        """
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc3(x)
