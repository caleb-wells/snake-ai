import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch


class ReplayMemory:
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
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.FloatTensor, ...]:
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
        return len(self.memory)
