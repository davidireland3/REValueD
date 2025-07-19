"""Training utilities and helper functions."""
import random
from typing import List, Optional, Tuple, Any

import gymnasium as gym
import numpy as np
import torch

from dmc_datasets.environment_utils import make_env as dmc_make_env


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(
        domain: str,
        task: str,
        bin_size: int = 3,
        factorised: bool = True,
        seed: Optional[int] = None
) -> gym.Env:
    """Create environment using dmc_datasets.

    Args:
        domain: DMC domain name
        task: DMC task name
        bin_size: Number of bins for discretisation
        factorised: Whether to use factorised action space
        seed: Random seed

    Returns:
        Gymnasium environment
    """
    env = dmc_make_env(domain, task, bin_size, factorised)

    if seed is not None:
        env.reset(seed=seed)

    return env


def run_evaluation(
        algorithm: Any,
        env: gym.Env,
        seed: Optional[int] = None
) -> float:
    """Run evaluation episodes.

    Args:
        algorithm: Trained algorithm
        env: Environment to evaluate in
        seed: Random seed for evaluation

    Returns:
        Tuple of (mean_score, std_score)
    """
    state, _ = env.reset(seed=seed)
    done = False
    score = 0.0
    while not done:
        action = algorithm.greedy_act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
    return score


def compute_n_step_returns(
        transitions: List[Tuple[np.ndarray, np.ndarray, float]],
        gamma: float,
        n_steps: int
) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
    """Compute n-step returns from transitions.

    Args:
        transitions: List of (state, action, reward) tuples
        gamma: Discount factor
        n_steps: Number of steps for returns

    Returns:
        List of processed transitions with n-step returns
    """
    processed = []

    for i in range(len(transitions) - n_steps + 1):
        state, action, _ = transitions[i]

        # Compute n-step return
        n_step_return = 0.0
        for j in range(n_steps):
            _, _, reward = transitions[i + j]
            n_step_return += (gamma ** j) * reward

        # Get final state
        if i + n_steps < len(transitions):
            next_state = transitions[i + n_steps][0]
            done = False
        else:
            next_state = transitions[-1][0]  # Last state
            done = True

        processed.append((state, action, n_step_return, next_state, done))

    return processed
