import json
import os
from typing import List, Tuple

import numpy as np
import pygame
import torch
from tqdm import tqdm

from ai.agent import DQNAgent
from game.environment import SnakeEnvironment
from utils.helpers import setup_training_dirs


def load_model_and_metrics(
    model_path: str, metrics_path: str, agent: DQNAgent, load_checkpoint: bool
) -> Tuple[int, List[int]]:
    best_score = 0
    episode_scores = []

    if load_checkpoint and os.path.exists(model_path):
        print(f"\nLoading saved model from: {model_path}")
        agent.load(model_path)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
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
    episodes: int = 10000,
    max_steps: int = 2000,
    render_training: bool = True,
    game_speed: int = 10,
    load_checkpoint: bool = True,
) -> None:
    print("\n=== Snake AI Training ===")
    print(
        f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}"
    )

    setup_training_dirs()
    model_path = "checkpoints/snake_ai_model.pt"
    metrics_path = "logs/training_metrics.json"

    env = SnakeEnvironment(render_mode=render_training)
    agent = DQNAgent()

    best_score, episode_scores = load_model_and_metrics(
        model_path, metrics_path, agent, load_checkpoint
    )
    episode_lengths = []

    try:
        progress_bar = tqdm(range(episodes), desc="Training")
        for _ in progress_bar:
            step, info = run_training_episode(
                env, agent, max_steps, render_training, game_speed
            )

            episode_scores.append(info["score"])
            episode_lengths.append(step + 1)

            progress_bar.set_postfix(
                {
                    "score": info["score"],
                    "epsilon": f"{agent.epsilon:.2f}",
                    "avg_score": f"{np.mean(episode_scores[-100:]):.2f}",
                }
            )

            if info["score"] > best_score:
                best_score = info["score"]
                print(f"\nNew best score! {best_score}")
                agent.save(model_path)

            metrics = {
                "best_score": best_score,
                "episode_scores": episode_scores,
                "episode_lengths": episode_lengths,
                "last_epsilon": agent.epsilon,
                "total_steps": agent.steps,
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted. Saving current model...")
        agent.save(model_path)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    train(render_training=True, game_speed=75, load_checkpoint=True)
