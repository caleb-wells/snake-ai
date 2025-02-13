"""
This module implements a replay memory buffer for Deep Q-Learning, storing
and sampling experiences (state, action, reward, next_state, done) for
training.
"""

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch


class ReplayMemory:
    """
    A circular buffer that stores and samples transitions for training a DQL agent.

    The memory stores transitions as tuples of (state, action, reward, next_state, done),
    allowing for experience replay during training to break correlations in sequential data.
    """

    def __init__(self, capacity: int = 100000):
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Adds a transition to the replay memory.

        Args:
            state (np.ndarray): The current state observation.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (np.ndarray): The resulting state after taking the action.
            done (bool): Whether the episode ended after this transition.
            *args: Additional arguments (not used, but allows for extension).
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples a batch of transitions from memory.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[torch.Tensor, ...]: Batch of transitions as PyTorch tensors
            (states, actions, rewards, next_states, dones).
        """
        transitions = random.sample(list(self.memory), batch_size)
        batch = list(zip(*transitions))
        return (
            torch.FloatTensor(np.array(batch[0])),  # states
            torch.LongTensor(batch[1]),  # actions
            torch.FloatTensor(batch[2]),  # rewards
            torch.FloatTensor(np.array(batch[3])),  # next_states
            torch.FloatTensor(batch[4]),  # dones
        )

    def __len__(self) -> int:
        """Returns the current size of the memory."""
        return len(self.memory)
