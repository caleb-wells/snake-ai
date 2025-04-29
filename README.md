# Snake AI

A deep reinforcement learning project that teaches an AI agent to play Snake using PyTorch and Deep Q-Learning.

## 🚀 Features

- Deep Q-Learning implementation with PyTorch
- Custom Snake game environment built with Pygame
- Tensorboard integration for training visualization
- Cross-platform compatibility
- Comprehensive testing suite
- Built-in performance profiling tools

## 📋 Requirements

- Python 3.8 or higher
- Git
- pip (usually comes with Python)

For GPU acceleration (optional but recommended):
- CUDA-compatible GPU
- CUDA Toolkit
- cuDNN

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/caleb-wells/snake-ai.git
cd snake-ai
```

2. Run the setup script:
```bash
# Make the script executable
chmod +x setup.sh

# Run the setup
./scripts/setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

## 🎮 Usage

### Training the AI

```bash
python src/main.py
```

### Visualizing Training Progress

```bash
tensorboard --logdir runs
```

Then open your browser and navigate to `http://localhost:6006`

## 🧪 Development

### Project Structure
```
snake-ai/
├── src/
│   ├── game/           # Snake game implementation
│   ├── ai/             # AI/ML models and training
│   └── utils/          # Helper functions
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
├── data/               # Training data
├── logs/               # Training logs
└── checkpoints/        # Model checkpoints
```

### Code Quality Tools

Format code:
```bash
black .
```

Sort imports:
```bash
isort .
```

Type checking:
```bash
mypy src
```

Run tests:
```bash
pytest
```

### Performance Profiling

CPU profiling:
```bash
py-spy record -o profile.svg -- python src/main.py
```

Memory profiling:
```bash
python -m memory_profiler src/main.py
```

## 📊 Training Configuration

Key hyperparameters can be configured in `.env`:

```ini
BATCH_SIZE=64
LEARNING_RATE=0.001
GAMMA=0.99
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=0.995
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch:
```bash
git checkout -b feature/amazing-feature
```
3. Commit your changes:
```bash
git add .
git commit -m "Add some amazing feature"
```
4. Push to the branch:
```bash
git push origin feature/amazing-feature
```
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium for the environment structure inspiration
- PyTorch team for the amazing deep learning framework
- Pygame community for the game development framework
