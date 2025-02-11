from dataclasses import dataclass
from typing import List, Optional, Tuple

from .constants import (
    GRID_HEIGHT,
    GRID_WIDTH,
)


@dataclass
class Snake:
    body: List[Tuple[int, int]]
    direction: Tuple[int, int]

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]

    def move(self, direction: Optional[Tuple[int, int]] = None) -> None:
        if direction is not None:
            self.direction = direction

        new_head = (
            (self.head[0] + self.direction[0]) % GRID_WIDTH,
            (self.head[1] + self.direction[1]) % GRID_HEIGHT,
        )
        self.body.insert(0, new_head)

    def grow(self) -> None:
        """Don't remove tail when growing"""
        pass

    def shrink(self) -> None:
        """Remove tail when moving"""
        self.body.pop()

    def collides_with_self(self) -> bool:
        return self.head in self.body[1:]
