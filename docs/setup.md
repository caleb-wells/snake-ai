# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/caleb-wells/snake-ai.git
cd snake-ai
```

2. Make setup script executable:
```bash
chmod +x setup.sh
```

3. Run setup script:
```bash
./scripts/setup.sh
```

The setup script will:
- Create virtual environment
- Install dependencies
- Set up project structure
- Configure development tools

## Virtual Environment

### Activation

```bash
source venv/bin/activate  # Unix/MacOS
.\venv\Scripts\activate   # Windows
```

### Deactivation

```bash
deactivate
```

## Development Environment

### Installed Tools

- **Code Formatting**
  - black
  - isort
  - flake8

- **Type Checking**
  - mypy

- **Testing**
  - pytest
  - pytest-cov

- **Pre-commit Hooks**
  - trailing whitespace removal
  - end-of-file fixing
  - YAML/TOML checking
  - code formatting
  - type checking

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

## Hardware Acceleration

### Apple Silicon (M1/M2)

The project automatically uses MPS (Metal Performance Shaders) if available:
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### NVIDIA GPU

CUDA support is automatically detected:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Configuration

### Environment Variables (.env)

```ini
# Training Configuration
BATCH_SIZE=64
LEARNING_RATE=0.001
GAMMA=0.99
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=0.995

# Game Configuration
WINDOW_SIZE=400
GRID_SIZE=20
FPS=30
```

## Updating Dependencies

To update project dependencies:

1. Modify `requirements.txt`
2. Run:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Setup Issues

1. **Python Version Error**
   - Install Python 3.8 or higher
   - Verify installation: `python3 --version`

2. **Virtual Environment Issues**
   - Delete `venv` directory
   - Run setup script again

3. **Package Installation Errors**
   - Upgrade pip: `pip install --upgrade pip`
   - Check internet connection
   - Verify Python version compatibility
