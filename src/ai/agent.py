import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .memory import ReplayMemory
from .model import DQN


class DQNAgent:
    def __init__(
        self,
        state_size: int = 11,
        action_size: int = 3,
        hidden_size: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 100000,
    ):
        # Check if MPS (Metal Performance Shaders) is available
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # Initialize networks and move them to MPS device
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-5
        )
        self.memory = ReplayMemory(memory_size)

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.steps = 0

    def act(self, state: np.ndarray) -> int:
        rng = np.random.default_rng(42)
        if rng.random() < self.epsilon:
            return int(rng.integers(self.action_size))

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return int(q_values.max(1)[1].item())

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch and move to device
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q(s_t, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute max Q(s_{t+1}, a) for all next states
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

        return float(loss.item())

    def save(self, path: str) -> None:
        try:
            print(f"\nSaving model to {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save only the state dictionaries and essential data
            save_dict = {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            }

            torch.save(save_dict, path)
            print("Model saved successfully!")

        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load(self, path: str) -> None:
        try:
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            print("Checkpoint loaded successfully")

            print("Loading network states...")
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])

            print("Loading optimizer state...")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("Loading training parameters...")
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.steps = checkpoint.get("steps", 0)

            print("Model loaded successfully!")
            print(f"Current epsilon: {self.epsilon}")
            print(f"Total steps: {self.steps}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Starting with fresh model")
