# Training Documentation

## Training Process

The Snake AI uses Deep Q-Learning (DQN) with the following features:
- Experience replay for stable learning
- Target network for reducing overestimation
- Epsilon-greedy exploration strategy

### Hyperparameters

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99        # Discount factor
EPSILON_START = 1.0 # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.995
```

## Running Training

### Basic Training

```bash
python src/main.py
```

### Training Parameters

- `render_training`: Toggle visual rendering (default: True)
- `game_speed`: Control game visualization speed (default: 10)
- `load_checkpoint`: Continue from previous checkpoint (default: True)

Example:
```bash
python src/main.py --render_training True --game_speed 50
```

## Checkpoints

Models are automatically saved:
- When a new best score is achieved
- On training interruption (Ctrl+C)
- At regular intervals

### Checkpoint Structure

```python
{
    'policy_net_state_dict': state_dict,
    'target_net_state_dict': state_dict,
    'optimizer_state_dict': state_dict,
    'epsilon': float,
    'steps': int,
    'memory': list
}
```

## Monitoring Progress

### Training Metrics

Metrics are saved in `logs/training_metrics.json`:
- Best score achieved
- Episode scores
- Episode lengths
- Current exploration rate
- Total training steps

### Visual Feedback

During training, you can observe:
- Current score
- Exploration rate (epsilon)
- Average score over last 100 episodes
- Training progress bar

## Troubleshooting

### Common Issues

1. **Poor Learning Performance**
   - Check epsilon decay rate
   - Verify reward structure
   - Ensure state representation is correct

2. **Slow Training**
   - Disable rendering if not needed
   - Increase game speed
   - Check hardware acceleration (MPS/CUDA)

3. **Memory Issues**
   - Reduce replay buffer size
   - Decrease batch size
   - Clear replay buffer between sessions
