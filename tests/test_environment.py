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
def env() -> SnakeEnvironment:
    return SnakeEnvironment(render_mode=False)


def test_environment_initialization(env: SnakeEnvironment) -> None:
    """Test environment initialization."""
    env.reset()
    assert isinstance(env.game["snake"], Snake)
    assert len(env.game["snake"].body) == INITIAL_SNAKE_LENGTH
    assert isinstance(env.game["food"], tuple)
    assert env.game["score"] == 0
    assert env.game["steps"] == 0


def test_reset(env: SnakeEnvironment) -> None:
    """Test environment reset."""
    env.step(0)
    env.step(1)
    assert env.game["steps"] > 0

    state = env.reset()
    assert env.game["steps"] == 0
    assert env.game["score"] == 0
    assert isinstance(state, np.ndarray)
    assert state.shape == (11,)


def test_state_shape(env: SnakeEnvironment) -> None:
    """Test state vector shape."""
    state = env._get_state()
    assert isinstance(state, np.ndarray)
    assert state.shape == (11,)
    assert state.dtype == np.int64


def test_valid_actions(env: SnakeEnvironment) -> None:
    """Test all possible actions."""
    for action in [0, 1, 2]:
        env.reset()
        next_state, reward, done, info = env.step(action)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "score" in info
        assert "steps" in info


def test_collision_detection(env: SnakeEnvironment) -> None:
    """Test snake collision detection."""
    env.reset()
    env.game["snake"].body = [(1, 1), (1, 1)]
    assert env.game["snake"].collides_with_self()


def test_food_spawning(env: SnakeEnvironment) -> None:
    """Test food spawn mechanism."""
    env.reset()
    food = env._spawn_food()
    assert isinstance(food, tuple)
    assert len(food) == 2
    assert 0 <= food[0] < GRID_WIDTH
    assert 0 <= food[1] < GRID_HEIGHT
    assert food not in env.game["snake"].body


def test_reward_system(env: SnakeEnvironment) -> None:
    """Test reward mechanism."""
    env.reset()
    env.game["snake"].direction = (1, 0)
    env.game["food"] = (env.game["snake"].head[0] + 2, env.game["snake"].head[1])
    _, reward, _, _ = env.step(0)
    assert reward == REWARD_MOVE

    env.reset()
    env.game["snake"].direction = (1, 0)
    env.game["food"] = (env.game["snake"].head[0] + 1, env.game["snake"].head[1])
    _, reward, _, _ = env.step(0)
    assert reward == REWARD_FOOD


def test_score_system(env: SnakeEnvironment) -> None:
    """Test scoring mechanism."""
    env.reset()
    initial_score = env.game["score"]
    env.game["food"] = (env.game["snake"].head[0] + 1, env.game["snake"].head[1])
    _, _, _, info = env.step(0)
    assert info["score"] == initial_score + 1


def test_step_counter(env: SnakeEnvironment) -> None:
    """Test step counting."""
    env.reset()
    assert env.game["steps"] == 0
    env.step(0)
    assert env.game["steps"] == 1
    env.step(1)
    assert env.game["steps"] == 2


def test_snake_growth(env: SnakeEnvironment) -> None:
    """Test snake growth when eating food."""
    env.reset()
    initial_length = len(env.game["snake"].body)
    env.game["food"] = (env.game["snake"].head[0] + 1, env.game["snake"].head[1])
    env.step(0)
    assert len(env.game["snake"].body) == initial_length + 1


def test_direction_changes(env: SnakeEnvironment) -> None:
    """Test snake direction changes."""
    env.reset()
    initial_direction = env.game["snake"].direction
    env.step(1)
    assert env.game["snake"].direction != initial_direction
    env.step(2)
    assert env.game["snake"].direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]


def test_info_dict(env: SnakeEnvironment) -> None:
    """Test info dictionary contents."""
    env.reset()
    _, _, _, info = env.step(0)
    assert "score" in info
    assert "steps" in info
    assert isinstance(info["score"], int)
    assert isinstance(info["steps"], int)


def test_boundary_wrapping(env: SnakeEnvironment) -> None:
    """Test snake wrapping around boundaries."""
    env.reset()
    env.game["snake"].body = [(GRID_WIDTH - 1, 0)]
    env.game["snake"].direction = (1, 0)
    _, _, _, _ = env.step(0)
    assert env.game["snake"].head[0] == 0
