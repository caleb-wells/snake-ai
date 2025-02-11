from typing import Dict, Tuple

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
    def __init__(self, render_mode: bool = True):
        self.render_mode = render_mode
        self.epsilon = 1.0
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self) -> np.ndarray:
        # Initialize snake in the middle
        self.snake = Snake(body=[(GRID_WIDTH // 2, GRID_HEIGHT // 2)], direction=(1, 0))

        # Initial snake length
        for _ in range(INITIAL_SNAKE_LENGTH - 1):
            self.snake.body.append((self.snake.body[-1][0] - 1, self.snake.body[-1][1]))

        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        return self._get_state()

    def _spawn_food(self) -> Tuple[int, int]:
        rng = np.random.default_rng(42)
        while True:
            food = (rng.integers(0, GRID_WIDTH), rng.integers(0, GRID_HEIGHT))
            if food not in self.snake.body:
                return food

    def _get_state(self) -> np.ndarray:
        # Create 11 element state vector
        state = []

        # Danger straight, right, left
        point_l = self._point_in_direction("left")
        point_r = self._point_in_direction("right")
        point_u = self._point_in_direction("up")
        point_d = self._point_in_direction("down")

        dir_l = self.snake.direction == (-1, 0)
        dir_r = self.snake.direction == (1, 0)
        dir_u = self.snake.direction == (0, -1)
        dir_d = self.snake.direction == (0, 1)

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            # Danger right
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            # Danger left
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            self.food[0] < self.snake.head[0],  # food left
            self.food[0] > self.snake.head[0],  # food right
            self.food[1] < self.snake.head[1],  # food up
            self.food[1] > self.snake.head[1],  # food down
        ]

        return np.array(state, dtype=int)

    def _point_in_direction(self, direction: str) -> Tuple[int, int]:
        x, y = self.snake.head
        if direction == "right":
            return ((x + 1) % GRID_WIDTH, y)
        elif direction == "left":
            return ((x - 1) % GRID_WIDTH, y)
        elif direction == "up":
            return (x, (y - 1) % GRID_HEIGHT)
        else:  # down
            return (x, (y + 1) % GRID_HEIGHT)

    def _is_collision(self, point: Tuple[int, int]) -> bool:
        return point in self.snake.body[1:]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, int]]:
        # Convert action (0, 1, 2) to direction
        # 0: straight, 1: right turn, 2: left turn
        clock_wise = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        idx = clock_wise.index(self.snake.direction)

        if action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = clock_wise[idx]

        self.snake.move(new_dir)
        self.steps += 1

        reward = REWARD_MOVE
        game_over = False

        # Check if snake ate food
        if self.snake.head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self.snake.grow()
            self.food = self._spawn_food()
        else:
            self.snake.shrink()

        # Check collision
        if self.snake.collides_with_self():
            game_over = True
            reward = REWARD_DEATH

        if self.render_mode:
            self.render()

        info = {"score": self.score, "steps": self.steps}

        state = self._get_state()

        return state, reward, game_over, info

    def render(self) -> None:
        if not self.render_mode:
            return

        # Handle Pygame events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        # Clear screen
        self.screen.fill(BACKGROUND)

        # Draw grid
        for x in range(0, WINDOW_SIZE, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_SIZE, y))

        # Draw snake
        for segment in self.snake.body:
            rect = pygame.Rect(
                segment[0] * GRID_SIZE,
                segment[1] * GRID_SIZE,
                GRID_SIZE - 1,
                GRID_SIZE - 1,
            )
            pygame.draw.rect(
                self.screen, SNAKE_COLOR, rect, border_radius=GRID_SIZE // 4
            )

        # Draw food
        rect = pygame.Rect(
            self.food[0] * GRID_SIZE,
            self.food[1] * GRID_SIZE,
            GRID_SIZE - 1,
            GRID_SIZE - 1,
        )
        pygame.draw.rect(self.screen, FOOD_COLOR, rect, border_radius=GRID_SIZE // 4)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))

        # Draw epsilon (exploration rate)
        epsilon_text = self.font.render(
            f"Epsilon: {self.epsilon:.2f}", True, TEXT_COLOR
        )
        self.screen.blit(epsilon_text, (10, 40))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self) -> None:
        if self.render_mode:
            pygame.quit()
