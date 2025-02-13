"""
This module implements the Snake class used in the game.
It handles the snake's body, movement, and collision detection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .constants import (
    GRID_HEIGHT,
    GRID_WIDTH,
)


@dataclass
class Snake:
    """
    Represents the snake in the game, including its body segments and current moving direction.
    """

    body: List[Tuple[int, int]]
    direction: Tuple[int, int]

    @property
    def head(self) -> Tuple[int, int]:
        """Return the current head position of the snake."""
        return self.body[0]

    def move(self, direction: Optional[Tuple[int, int]] = None) -> None:
        """
        Move the snake in the given direction.
        If no new direction is provided, continue moving in the current direction.
        The new head is added to the beginning of the body list, with wrap-around at the edges.
        """
        if direction is not None:
            self.direction = direction

        new_head = (
            (self.head[0] + self.direction[0]) % GRID_WIDTH,
            (self.head[1] + self.direction[1]) % GRID_HEIGHT,
        )
        self.body.insert(0, new_head)

    def grow(self) -> None:
        """
        Grow the snake.
        This placeholder method indicates that the snake should not remove its tail when growing.
        """
        # No additional code is needed; the docstring is sufficient.
        return

    def shrink(self) -> None:
        """
        Shrink the snake by removing the tail segment.
        This method is called when the snake moves without consuming food.
        """
        self.body.pop()

    def collides_with_self(self) -> bool:
        """
        Check if the snake's head collides with any other part of its body.
        Returns True if a collision is detected, otherwise False.
        """
        return self.head in self.body[1:]
