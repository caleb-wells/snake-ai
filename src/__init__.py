"""
Snake AI Project

A deep reinforcement learning implementation that teaches an AI agent to play Snake.
This project combines PyTorch-based deep Q-learning with a Pygame implementation
of the classic Snake game to demonstrate reinforcement learning in action.

Main Components:
- Game: Snake game environment and mechanics
- AI: Deep Q-Network implementation and training
- Utils: Helper functions and project utilities

The project uses MPS/CUDA acceleration when available and provides visual feedback
of the training process.
"""

# Version: 1.0
from src.ai.agent import DQNAgent
from src.game.environment import SnakeEnvironment

__all__ = ["SnakeEnvironment", "DQNAgent"]
