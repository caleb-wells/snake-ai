"""
Game Module for Snake AI

This module implements the Snake game environment and mechanics.
It provides:
- Snake game logic and rules
- Game environment compatible with reinforcement learning
- Visual rendering using Pygame
- Constants and configurations for game settings
"""

# Version: 1.0
from .environment import SnakeEnvironment
from .snake import Snake

__all__ = ["SnakeEnvironment", "Snake"]
