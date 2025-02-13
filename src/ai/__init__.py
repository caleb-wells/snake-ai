"""
AI Module for Snake Game

This module contains the reinforcement learning components for training an AI agent
to play Snake. It includes:
- DQN (Deep Q-Network) implementation
- Replay Memory for experience storage
- Neural Network model architecture

The AI uses deep Q-learning to learn optimal Snake game strategies through
experience
"""

# Version: 1.0
from .agent import DQNAgent
from .memory import ReplayMemory
from .model import DQN

__all__ = ["DQNAgent", "ReplayMemory", "DQN"]
