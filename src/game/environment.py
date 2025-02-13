"""
This module defines the SnakeEnvironment for the game,
which handles game state, snake movement, collision, scoring, and rendering using pygame.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pygame

from .constants import (
    BACKGROUND,
    FOOD_COLOR,
    FPS,
    GRID_COLOR,
    GRID_HEIGHT,
    GRID_SIZE,
    GRID_WIDTH,
    INITIAL_SNAKE_LENGTH,
    REWARD_DEATH,
    REWARD_FOOD,
    REWARD_MOVE,
    SNAKE_COLOR,
    TEXT_COLOR,
    WINDOW_SIZE,
)
from .snake import Snake


class SnakeEnvironment:
    """
    A class representing the snake game environment.
    Handles game state, snake movement, collision, scoring, and rendering.
    """

    def __init__(self, render_mode: bool = True):
        """Initialize the SnakeEnvironment with optional rendering."""
        self.render_mode = render_mode
        self.epsilon = 1.0
        self.rng = np.random.default_rng()  # Initialize RNG without a fixed seed
        # Group display-related attributes in a dictionary.
        self.display = None
        if render_mode:
            pygame.init()
            self.display = {
                "screen": pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE)),
                "clock": pygame.time.Clock(),
                "font": pygame.font.Font(None, 36),
            }
            pygame.display.set_caption("Snake Game")
        # Group game state attributes in a single dictionary.
        self.game: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to its initial state and return the starting state."""
        # Initialize snake in the middle.
        snake = Snake(body=[(GRID_WIDTH // 2, GRID_HEIGHT // 2)], direction=(1, 0))
        # Set initial snake length.
        for _ in range(INITIAL_SNAKE_LENGTH - 1):
            snake.body.append((snake.body[-1][0] - 1, snake.body[-1][1]))
        # First, store the snake in the game dictionary.
        self.game = {"snake": snake}
        # Now spawn food using the snake that has been added.
        self.game["food"] = self._spawn_food()
        self.game["score"] = 0
        self.game["steps"] = 0
        return self._get_state()

    def _spawn_food(self) -> Tuple[int, int]:
        """Spawn food in a random location not occupied by the snake."""
        while True:
            food = (self.rng.integers(0, GRID_WIDTH), self.rng.integers(0, GRID_HEIGHT))
            if food not in self.game["snake"].body:
                return food

    def _get_state(self) -> np.ndarray:
        """Return the current game state as a numpy array of integers."""
        # Danger straight, right, left.
        point_l = self._point_in_direction("left")
        point_r = self._point_in_direction("right")
        point_u = self._point_in_direction("up")
        point_d = self._point_in_direction("down")

        snake = self.game["snake"]
        food = self.game["food"]

        dir_l = snake.direction == (-1, 0)
        dir_r = snake.direction == (1, 0)
        dir_u = snake.direction == (0, -1)
        dir_d = snake.direction == (0, 1)

        state = [
            int(
                (dir_r and self._is_collision(point_r))
                or (dir_l and self._is_collision(point_l))
                or (dir_u and self._is_collision(point_u))
                or (dir_d and self._is_collision(point_d))
            ),
            int(
                (dir_u and self._is_collision(point_r))
                or (dir_d and self._is_collision(point_l))
                or (dir_l and self._is_collision(point_u))
                or (dir_r and self._is_collision(point_d))
            ),
            int(
                (dir_d and self._is_collision(point_r))
                or (dir_u and self._is_collision(point_l))
                or (dir_r and self._is_collision(point_u))
                or (dir_l and self._is_collision(point_d))
            ),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(food[0] < snake.head[0]),
            int(food[0] > snake.head[0]),
            int(food[1] < snake.head[1]),
            int(food[1] > snake.head[1]),
        ]

        return np.array(state, dtype=int)

    def _point_in_direction(self, direction: str) -> Tuple[int, int]:
        """
        Return the point in the given direction from the snake's head.
        The grid wraps around, so positions are computed modulo the grid dimensions.
        """
        snake = self.game["snake"]
        x, y = snake.head
        if direction == "right":
            return ((x + 1) % GRID_WIDTH, y)
        if direction == "left":
            return ((x - 1) % GRID_WIDTH, y)
        if direction == "up":
            return (x, (y - 1) % GRID_HEIGHT)
        # down case.
        return (x, (y + 1) % GRID_HEIGHT)

    def _is_collision(self, point: Tuple[int, int]) -> bool:
        """Return True if the given point collides with the snake's body (excluding the head)."""
        return point in self.game["snake"].body[1:]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, int]]:
        """
        Execute one time step within the environment given an action.
        Returns a tuple of (state, reward, game_over, info).
        """
        # Convert action (0, 1, 2) to direction: 0 = straight, 1 = right turn, 2 = left turn.
        clock_wise = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        snake = self.game["snake"]
        idx = clock_wise.index(snake.direction)

        if action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = clock_wise[idx]

        snake.move(new_dir)
        self.game["steps"] += 1

        reward = REWARD_MOVE
        game_over = False

        # Check if snake ate food.
        if snake.head == self.game["food"]:
            self.game["score"] += 1
            reward = REWARD_FOOD
            snake.grow()
            self.game["food"] = self._spawn_food()
        else:
            snake.shrink()

        # Check collision.
        if snake.collides_with_self():
            game_over = True
            reward = REWARD_DEATH

        if self.render_mode:
            self.render()

        info = {"score": self.game["score"], "steps": self.game["steps"]}
        state = self._get_state()
        return state, reward, game_over, info

    def render(self) -> None:
        """Render the current game state to the screen."""
        if not self.render_mode or self.display is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        # Clear screen.
        self.display["screen"].fill(BACKGROUND)

        # Draw grid lines.
        for x in range(0, WINDOW_SIZE, GRID_SIZE):
            pygame.draw.line(
                self.display["screen"], GRID_COLOR, (x, 0), (x, WINDOW_SIZE)
            )
        for y in range(0, WINDOW_SIZE, GRID_SIZE):
            pygame.draw.line(
                self.display["screen"], GRID_COLOR, (0, y), (WINDOW_SIZE, y)
            )

        # Draw snake segments.
        for segment in self.game["snake"].body:
            rect = pygame.Rect(
                segment[0] * GRID_SIZE,
                segment[1] * GRID_SIZE,
                GRID_SIZE - 1,
                GRID_SIZE - 1,
            )
            pygame.draw.rect(
                self.display["screen"], SNAKE_COLOR, rect, border_radius=GRID_SIZE // 4
            )

        # Draw food.
        food = self.game["food"]
        rect = pygame.Rect(
            food[0] * GRID_SIZE,
            food[1] * GRID_SIZE,
            GRID_SIZE - 1,
            GRID_SIZE - 1,
        )
        pygame.draw.rect(
            self.display["screen"], FOOD_COLOR, rect, border_radius=GRID_SIZE // 4
        )

        # Draw score.
        score_text = self.display["font"].render(
            f"Score: {self.game['score']}", True, TEXT_COLOR
        )
        self.display["screen"].blit(score_text, (10, 10))

        # Draw epsilon (exploration rate).
        epsilon_text = self.display["font"].render(
            f"Epsilon: {self.epsilon:.2f}", True, TEXT_COLOR
        )
        self.display["screen"].blit(epsilon_text, (10, 40))

        pygame.display.flip()
        self.display["clock"].tick(FPS)

    def close(self) -> None:
        """Close the game environment and quit pygame."""
        if self.render_mode:
            pygame.quit()
