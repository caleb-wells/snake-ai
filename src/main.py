"""
Main module for training the Snake AI. This script sets up the training environment,
loads/saves model checkpoints and metrics, and runs training episodes.
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pygame
import torch
from tqdm import tqdm

# pylint: disable=import-error
from ai.agent import DQNAgent
from game.environment import SnakeEnvironment
from utils.helpers import setup_training_dirs

# pylint: enable=import-error


def load_model_and_metrics(
    model_path: str, metrics_path: str, agent: DQNAgent, load_checkpoint: bool
) -> Tuple[int, List[int]]:
    """Load model and training metrics from disk."""
    best_score = 0
    episode_scores = []

    if load_checkpoint and os.path.exists(model_path):
        print(f"\nLoading saved model from: {model_path}")
        agent.load(model_path)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                best_score = metrics.get("best_score", 0)
                episode_scores = metrics.get("episode_scores", [])
                print(f"Previous best score: {best_score}")
                if episode_scores:
                    print(f"Previous training episodes: {len(episode_scores)}")
                    print(
                        f"Average of last 100 episodes: {np.mean(episode_scores[-100:]):.2f}"
                    )
    else:
        print("\nStarting fresh training")

    return best_score, episode_scores


def run_training_episode(
    env: SnakeEnvironment,
    agent: DQNAgent,
    max_steps: int,
    render_training: bool,
    game_speed: int,
) -> Tuple[int, dict]:
    """Run a single training episode."""
    state = env.reset()
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step()
        env.epsilon = agent.epsilon
        state = next_state

        if render_training:
            env.render()
            pygame.time.delay(1000 // game_speed)

        if done:
            break
    return step, info


def train(
    episodes: int = 1000,
    max_steps: int = 2000,
    render_training: bool = True,
    game_speed: int = 50,
    load_checkpoint: bool = True,
) -> None:
    """Train the Snake AI."""
    print("\n=== Snake AI Training ===")
    setup_training_dirs()
    env = SnakeEnvironment(render_mode=render_training)
    agent = DQNAgent()

    best_score, episode_scores = load_model_and_metrics(
        "checkpoints/snake_ai_model.pt",
        "logs/training_metrics.json",
        agent,
        load_checkpoint,
    )
    episode_lengths = []

    try:
        for _ in tqdm(range(episodes), desc="Training"):
            step, info = run_training_episode(
                env, agent, max_steps, render_training, game_speed
            )

            episode_scores.append(info["score"])
            episode_lengths.append(step + 1)

            # Update training progress in the terminal.
            tqdm.write(
                f"score: {info['score']}, epsilon: {agent.epsilon:.2f}, "
                f"avg_score: {np.mean(episode_scores[-100:]):.2f}"
            )

            if info["score"] > best_score:
                best_score = info["score"]
                print(f"\nNew best score! {best_score}")
                agent.save("checkpoints/snake_ai_model.pt")

            current_metrics = {
                "best_score": best_score,
                "episode_scores": episode_scores,
                "episode_lengths": episode_lengths,
                "last_epsilon": agent.epsilon,
                "total_steps": agent.steps,
            }
            with open("logs/training_metrics.json", "w", encoding="utf-8") as f:
                json.dump(current_metrics, f)

    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted. Saving current model...")
        agent.save("checkpoints/snake_ai_model.pt")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to run the model on (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--render_training",
        type=bool,
        default=True,
        help="Render training",
    )
    parser.add_argument(
        "--game_speed",
        type=int,
        default=50,
        help="Game speed",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=bool,
        default=True,
        help="Load checkpoint",
    )
    args = parser.parse_args()

    # Device selection: use argument if provided, otherwise auto-detect
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Print Game Arguments
    print(f"Device: {device}")
    print(f"Render training: {args.render_training}")
    print(f"Game speed: {args.game_speed}")
    print(f"Load checkpoint: {args.load_checkpoint}")

    train(
        render_training=args.render_training,
        game_speed=args.game_speed,
        load_checkpoint=args.load_checkpoint,
    )
