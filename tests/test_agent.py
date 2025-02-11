from pathlib import Path

import numpy as np
import pytest
import torch

from src.ai.agent import DQNAgent
from src.ai.memory import ReplayMemory


@pytest.fixture
def agent() -> DQNAgent:
    return DQNAgent()


def test_agent_initialization(agent: DQNAgent) -> None:
    """Test if agent is initialized with correct parameters"""
    assert agent.state_size == 11
    assert agent.action_size == 3
    assert isinstance(agent.policy_net, torch.nn.Module)
    assert isinstance(agent.target_net, torch.nn.Module)
    assert isinstance(agent.memory, ReplayMemory)
    assert agent.epsilon == 1


def test_agent_act_exploit(agent: DQNAgent) -> None:
    """Test agent's exploitation behavior"""
    state = np.zeros(11)
    agent.epsilon = 0.0  # Force exploitation
    action = agent.act(state)
    assert isinstance(action, int)
    assert 0 <= action <= 2


def test_memory_push(agent: DQNAgent) -> None:
    """Test replay memory functionality"""
    state = np.zeros(11)
    next_state = np.ones(11)
    agent.memory.push(state, 1, 1.0, next_state, False)
    assert len(agent.memory) == 1


def test_epsilon_decay(agent: DQNAgent) -> None:
    """Test epsilon decay mechanism"""
    initial_epsilon = agent.epsilon

    # Need to have enough samples in memory for training
    state = np.zeros(11)
    next_state = np.ones(11)
    for _ in range(agent.batch_size):
        agent.memory.push(state, 1, 1.0, next_state, False)

    # Now train_step will actually update epsilon
    for _ in range(10):
        agent.train_step()
    assert agent.epsilon < initial_epsilon


def test_save_load(agent: DQNAgent, tmp_path: Path) -> None:
    """Test model saving and loading"""
    save_path = tmp_path / "test_model.pt"
    agent.save(str(save_path))
    assert save_path.exists()

    original_epsilon = agent.epsilon
    agent.epsilon = 0.5

    agent.load(str(save_path))
    assert agent.epsilon == original_epsilon


def test_train_step_no_memory(agent: DQNAgent) -> None:
    """Test training step with insufficient memory"""
    result = agent.train_step()
    assert result is None


def test_train_step(agent: DQNAgent) -> None:
    """Test training step with sufficient memory"""
    state = np.zeros(11)
    next_state = np.ones(11)
    for _ in range(agent.batch_size):
        agent.memory.push(state, 1, 1.0, next_state, False)

    loss = agent.train_step()
    assert isinstance(loss, float)


def test_device_assignment(agent: DQNAgent) -> None:
    """Test if models are on the correct device"""
    if torch.backends.mps.is_available():
        expected_device = "mps"
    elif torch.cuda.is_available():
        expected_device = "cuda"
    else:
        expected_device = "cpu"
    assert str(next(agent.policy_net.parameters()).device).startswith(expected_device)
    assert str(next(agent.target_net.parameters()).device).startswith(expected_device)


def test_invalid_action(agent: DQNAgent) -> None:
    """Test handling of invalid actions"""
    state = np.zeros(11)
    action = agent.act(state)
    assert isinstance(action, int)
    assert 0 <= action <= 2


def test_network_shape(agent: DQNAgent) -> None:
    """Test neural network input/output shapes"""
    state = torch.zeros((1, 11)).to(agent.device)
    output = agent.policy_net(state)
    assert output.shape == (1, 3)


def test_target_network_update(agent: DQNAgent) -> None:
    """Test target network update mechanism"""
    initial_params = [param.data.clone() for param in agent.target_net.parameters()]

    state = np.zeros(11)
    next_state = np.ones(11)
    for _ in range(agent.batch_size):
        agent.memory.push(state, 1, 1.0, next_state, False)

    for _ in range(101):  # More than target update frequency
        agent.train_step()

    current_params = [param.data for param in agent.target_net.parameters()]
    assert any(not torch.equal(i, c) for i, c in zip(initial_params, current_params))
