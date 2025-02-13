# pylint: disable=wrong-import-position, redefined-outer-name, line-too-long, import-error, protected-access
"""
Test module for SnakeEnvironment functionality.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.game.constants import (
    GRID_HEIGHT,
    GRID_WIDTH,
    INITIAL_SNAKE_LENGTH,
    REWARD_FOOD,
    REWARD_MOVE,
)
from src.game.environment import SnakeEnvironment
from src.game.snake import Snake


@pytest.fixture
def env_instance() -> SnakeEnvironment:
    """Fixture for SnakeEnvironment instance."""
    return SnakeEnvironment(render_mode=False)


def test_environment_initialization(env_instance: SnakeEnvironment) -> None:
    """Test environment initialization."""
    env_instance.reset()
    assert isinstance(env_instance.game["snake"], Snake)
    assert len(env_instance.game["snake"].body) == INITIAL_SNAKE_LENGTH
    assert isinstance(env_instance.game["food"], tuple)
    assert env_instance.game["score"] == 0
    assert env_instance.game["steps"] == 0


def test_reset(env_instance: SnakeEnvironment) -> None:
    """Test environment reset."""
    env_instance.step(0)
    env_instance.step(1)
    assert env_instance.game["steps"] > 0

    state = env_instance.reset()
    assert env_instance.game["steps"] == 0
    assert env_instance.game["score"] == 0
    assert isinstance(state, np.ndarray)
    assert state.shape == (11,)


def test_state_shape(env_instance: SnakeEnvironment) -> None:
    """Test state vector shape."""
    state = env_instance._get_state()
    assert isinstance(state, np.ndarray)
    assert state.shape == (11,)
    assert state.dtype == np.int64


def test_valid_actions(env_instance: SnakeEnvironment) -> None:
    """Test all possible actions."""
    for action in [0, 1, 2]:
        env_instance.reset()
        next_state, reward, done, info = env_instance.step(action)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "score" in info
        assert "steps" in info


def test_collision_detection(env_instance: SnakeEnvironment) -> None:
    """Test snake collision detection."""
    env_instance.reset()
    env_instance.game["snake"].body = [(1, 1), (1, 1)]
    assert env_instance.game["snake"].collides_with_self()


def test_food_spawning(env_instance: SnakeEnvironment) -> None:
    """Test food spawn mechanism."""
    env_instance.reset()
    food = env_instance._spawn_food()
    assert isinstance(food, tuple)
    assert len(food) == 2
    assert 0 <= food[0] < GRID_WIDTH
    assert 0 <= food[1] < GRID_HEIGHT
    assert food not in env_instance.game["snake"].body


def test_reward_system(env_instance: SnakeEnvironment) -> None:
    """Test reward mechanism."""
    env_instance.reset()
    env_instance.game["snake"].direction = (1, 0)
    env_instance.game["food"] = (
        env_instance.game["snake"].head[0] + 2,
        env_instance.game["snake"].head[1],
    )
    _, reward, _, _ = env_instance.step(0)
    assert reward == REWARD_MOVE

    env_instance.reset()
    env_instance.game["snake"].direction = (1, 0)
    env_instance.game["food"] = (
        env_instance.game["snake"].head[0] + 1,
        env_instance.game["snake"].head[1],
    )
    _, reward, _, _ = env_instance.step(0)
    assert reward == REWARD_FOOD


def test_score_system(env_instance: SnakeEnvironment) -> None:
    """Test scoring mechanism."""
    env_instance.reset()
    initial_score = env_instance.game["score"]
    env_instance.game["food"] = (
        env_instance.game["snake"].head[0] + 1,
        env_instance.game["snake"].head[1],
    )
    _, _, _, info = env_instance.step(0)
    assert info["score"] == initial_score + 1


def test_step_counter(env_instance: SnakeEnvironment) -> None:
    """Test step counting."""
    env_instance.reset()
    assert env_instance.game["steps"] == 0
    env_instance.step(0)
    assert env_instance.game["steps"] == 1
    env_instance.step(1)
    assert env_instance.game["steps"] == 2


def test_snake_growth(env_instance: SnakeEnvironment) -> None:
    """Test snake growth when eating food."""
    env_instance.reset()
    initial_length = len(env_instance.game["snake"].body)
    env_instance.game["food"] = (
        env_instance.game["snake"].head[0] + 1,
        env_instance.game["snake"].head[1],
    )
    env_instance.step(0)
    assert len(env_instance.game["snake"].body) == initial_length + 1


def test_direction_changes(env_instance: SnakeEnvironment) -> None:
    """Test snake direction changes."""
    env_instance.reset()
    initial_direction = env_instance.game["snake"].direction
    env_instance.step(1)
    assert env_instance.game["snake"].direction != initial_direction
    env_instance.step(2)
    assert env_instance.game["snake"].direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]


def test_info_dict(env_instance: SnakeEnvironment) -> None:
    """Test info dictionary contents."""
    env_instance.reset()
    _, _, _, info = env_instance.step(0)
    assert "score" in info
    assert "steps" in info
    assert isinstance(info["score"], int)
    assert isinstance(info["steps"], int)


def test_boundary_wrapping(env_instance: SnakeEnvironment) -> None:
    """Test snake wrapping around boundaries."""
    env_instance.reset()
    env_instance.game["snake"].body = [(GRID_WIDTH - 1, 0)]
    env_instance.game["snake"].direction = (1, 0)
    _, _, _, _ = env_instance.step(0)
    assert env_instance.game["snake"].head[0] == 0
