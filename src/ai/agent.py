"""Deep Q-Network (DQN) Agent implementation for reinforcement learning tasks.

This module provides a DQN agent implementation with experience replay and target network.
It supports different device configurations (MPS, CUDA, CPU) and includes save/load functionality.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn, optim

from .memory import ReplayMemory
from .model import DQN


@dataclass
class NetworkConfig:
    """Neural network configuration parameters."""

    state_size: int = 11
    action_size: int = 3
    hidden_size: int = 128
    learning_rate: float = 0.001


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    gamma: float = 0.99
    batch_size: int = 64
    memory_size: int = 100000


@dataclass
class ExplorationConfig:
    """Exploration strategy configuration."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995


class Models:
    """Container for neural networks and optimization components."""

    def __init__(self, config: NetworkConfig, device: str):
        self.policy_net = DQN(
            config.state_size, config.hidden_size, config.action_size
        ).to(device)

        self.target_net = DQN(
            config.state_size, config.hidden_size, config.action_size
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dictionaries for saving."""
        return {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        """Load state dictionaries from checkpoint."""
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class TrainingState:
    """Container for training state and memory."""

    def __init__(self, memory_size: int, epsilon_start: float):
        self.memory = ReplayMemory(memory_size)
        self.steps = 0
        self.epsilon = epsilon_start

    def update_epsilon(self, epsilon_end: float, epsilon_decay: float) -> None:
        """Update epsilon value using decay schedule."""
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

    def increment_steps(self) -> int:
        """Increment and return the step counter."""
        self.steps += 1
        return self.steps


class DQNAgent:
    """Deep Q-Network Agent implementation."""

    def __init__(
        self,
        net_config: Optional[NetworkConfig] = None,
        train_config: Optional[TrainingConfig] = None,
        explore_config: Optional[ExplorationConfig] = None,
    ):
        """Initialize the DQN Agent with given configurations."""
        self._net_config = net_config or NetworkConfig()
        self._train_config = train_config or TrainingConfig()
        self._explore_config = explore_config or ExplorationConfig()

        # Setup device
        self._device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self._device}")

        # Initialize components
        self._models = Models(self._net_config, self._device)
        self._state = TrainingState(
            self._train_config.memory_size, self._explore_config.epsilon_start
        )

    @property
    def memory(self) -> ReplayMemory:
        """Access the replay memory buffer."""
        return self._state.memory

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._state.epsilon

    @property
    def steps(self) -> int:
        """Total number of training steps taken."""
        return self._state.steps

    def act(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        if np.random.random() < self._state.epsilon:
            return int(np.random.randint(self._net_config.action_size))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
        q_values = self._models.policy_net(state_tensor)
        return int(q_values.max(1)[1].item())

    def train_step(self) -> Optional[float]:
        """Perform a single training step on a batch of experiences."""
        if len(self._state.memory) < self._train_config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self._state.memory.sample(
            self._train_config.batch_size
        )
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device)

        current_q = self._models.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self._models.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self._train_config.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self._models.optimizer.zero_grad()
        loss.backward()
        self._models.optimizer.step()

        if self._state.steps % 100 == 0:
            self._models.target_net.load_state_dict(
                self._models.policy_net.state_dict()
            )

        self._state.update_epsilon(
            self._explore_config.epsilon_end, self._explore_config.epsilon_decay
        )
        self._state.increment_steps()

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save the agent's state to a file."""
        try:
            print(f"\nSaving model to {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            save_dict = {
                **self._models.get_state_dict(),
                "epsilon": self._state.epsilon,
                "steps": self._state.steps,
            }

            torch.save(save_dict, path)
            print("Model saved successfully!")

        except (IOError, RuntimeError) as e:
            print(f"Error saving model: {str(e)}")

    def load(self, path: str) -> None:
        """Load the agent's state from a file."""
        try:
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self._device)
            print("Checkpoint loaded successfully")

            print("Loading network states...")
            self._models.load_state_dict(checkpoint)

            print("Loading training parameters...")
            self._state.epsilon = checkpoint.get("epsilon", self._state.epsilon)
            self._state.steps = checkpoint.get("steps", 0)

            print("Model loaded successfully!")
            print(f"Current epsilon: {self._state.epsilon}")
            print(f"Total steps: {self._state.steps}")

        except (IOError, RuntimeError) as e:
            print(f"Error loading model: {str(e)}")
            print("Starting with fresh model")
