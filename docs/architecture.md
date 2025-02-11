# Snake AI Architecture

## Project Structure

```
snake-ai/
├── src/                    # Source code
│   ├── ai/                 # AI components
│   │   ├── agent.py        # DQN Agent implementation
│   │   ├── memory.py       # Replay buffer
│   │   └── model.py        # Neural network architecture
│   ├── game/               # Game environment
│   │   ├── constants.py    # Game constants and configs
│   │   ├── environment.py  # Snake game environment
│   │   └── snake.py        # Snake game logic
│   └── utils/              # Utility functions
│       └── helpers.py      # Helper functions
├── tests/                  # Test files
├── notebooks/              # Jupyter notebooks
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── docs/                   # Documentation
```

## Core Components

### Game Environment (`src/game/`)

The game environment provides the Snake game implementation and interface for the AI agent:

- **environment.py**: Implements the game environment with OpenAI Gym-like interface
- **snake.py**: Core Snake game logic and mechanics
- **constants.py**: Game configuration and constants

### AI Components (`src/ai/`)

The AI system uses Deep Q-Learning for training:

- **agent.py**: DQN Agent implementation with experience replay
- **memory.py**: Replay buffer for storing and sampling experiences
- **model.py**: Neural network architecture for Q-value prediction

### Utilities (`src/utils/`)

Support functions and helpers:

- **helpers.py**: Training setup and utility functions

## State Representation

The game state is represented as an 11-dimensional vector:

1. Danger straight (boolean)
2. Danger right (boolean)
3. Danger left (boolean)
4. Direction left (boolean)
5. Direction right (boolean)
6. Direction up (boolean)
7. Direction down (boolean)
8. Food left (boolean)
9. Food right (boolean)
10. Food up (boolean)
11. Food down (boolean)

## Actions

The agent can perform 3 actions:
- 0: Continue straight
- 1: Turn right
- 2: Turn left

## Neural Network Architecture

- Input Layer: 11 neurons (state size)
- Hidden Layer 1: 128 neurons with ReLU
- Hidden Layer 2: 128 neurons with ReLU
- Output Layer: 3 neurons (Q-values for each action)
