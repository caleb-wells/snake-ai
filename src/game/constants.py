"""
Constants module for the Snake game.

This module contains all the constant values used throughout the game including:
- Window and grid dimensions
- Color definitions
- Game settings (FPS, initial snake length)
- Reward values for reinforcement learning
"""

WINDOW_SIZE = 480
GRID_SIZE = 20
GRID_WIDTH = WINDOW_SIZE // GRID_SIZE
GRID_HEIGHT = WINDOW_SIZE // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Game settings
FPS = 30
INITIAL_SNAKE_LENGTH = 3
GRID_COUNT = WINDOW_SIZE // GRID_SIZE

# Colors (Dark theme)
BACKGROUND = (18, 18, 18)  # Dark gray, almost black
GRID_COLOR = (40, 40, 40)  # Slightly lighter gray for grid
SNAKE_COLOR = (50, 168, 82)  # Pleasant green
FOOD_COLOR = (217, 72, 72)  # Soft red
TEXT_COLOR = (200, 200, 200)  # Light gray for text

REWARD_FOOD = 10.0  # Reward for eating food
REWARD_DEATH = -10.0  # Penalty for dying
REWARD_MOVE = -0.01  # Small penalty for each move to encourage efficiency
