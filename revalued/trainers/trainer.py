"""Main trainer class for RL algorithms."""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from loguru import logger

from ..algorithms.base import BaseAlgorithm
from ..replay_buffers import ReplayBuffer
from ..utils import MetricTracker, set_seeds, make_env, run_evaluation, compute_n_step_returns


class Trainer:
    """Trainer for RL algorithms.

    Handles training loop, evaluation, logging, and checkpointing.
    """

    def __init__(
            self,
            algorithm: BaseAlgorithm,
            config: Dict[str, Any],
            save_dir: Optional[Path] = None
    ):
        """Initialise trainer.

        Args:
            algorithm: RL algorithm to train
            config: Training configuration dictionary
            save_dir: Directory to save models and logs
        """
        self.algorithm = algorithm
        self.config = config
        self.save_dir = save_dir or Path('experiments')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Extract config values
        self.domain = config['environment']['domain']
        self.task = config['environment']['task']
        self.bin_size = config['environment'].get('bin_size', 3)
        self.factorised = config['environment'].get('factorised', True)

        self.max_env_steps = config['training']['max_env_steps']
        self.update_ratio = config['training'].get('update_ratio', 1)
        self.num_updates = config['training'].get('num_updates', 1)
        self.eval_frequency = config['training'].get('eval_frequency', 10000)
        self.eval_episodes = config['training'].get('eval_episodes', 5)
        self.save_frequency = config['training'].get('save_frequency', 50000)

        self.burn_in_steps = config['replay_buffer'].get('burn_in_steps', 10000)
        self.n_steps = config['algorithm'].get('n_steps', 1)
        self.gamma = config['algorithm'].get('gamma', 0.99)

        self.seed = config['experiment'].get('seed', 42)

        # Setup
        set_seeds(self.seed)
        self.env = make_env(self.domain, self.task, self.bin_size, self.factorised, self.seed)
        self.eval_env = make_env(self.domain, self.task, self.bin_size, self.factorised, self.seed + 1000)

        # Replay buffer
        state_dim = self.env.observation_space.shape[0]
        action_dim = len(self.env.action_space) if self.factorised else 1
        self.replay_buffer = ReplayBuffer(
            capacity=config['replay_buffer']['capacity'],
            state_dim=state_dim,
            action_dim=action_dim,
            batch_size=config['algorithm']['batch_size'],
            device=self.algorithm.device
        )

        # Metrics
        self.metrics = MetricTracker()

        # Tracking
        self.env_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf

    def train(self) -> None:
        """Run full training loop."""
        logger.info(f"Starting training for {self.domain}_{self.task}")
        logger.info(f"Algorithm: {self.algorithm.__class__.__name__}")
        logger.info(f"Seed: {self.seed}")

        # Burn-in phase
        self._burn_in()

        # Main training loop
        episode_transitions = []

        while self.env_steps < self.max_env_steps:
            # Collect episode
            transitions = self._collect_episode()
            episode_transitions.extend(transitions)

            # Process n-step returns and add to buffer
            if len(episode_transitions) >= self.n_steps:
                processed = compute_n_step_returns(
                    episode_transitions[-len(transitions):],
                    self.gamma,
                    self.n_steps
                )
                for state, action, reward, next_state, done in processed:
                    self.replay_buffer.push(state, action, reward, next_state, done)

            # Update algorithm
            if self.env_steps % self.update_ratio == 0:
                for _ in range(self.num_updates):
                    batch = self.replay_buffer.sample()
                    update_metrics = self.algorithm.update(*batch)
                    self.metrics.update(**update_metrics)

            # Evaluation
            if self.env_steps % self.eval_frequency == 0:
                self._evaluate()

            # Save checkpoint
            if self.env_steps % self.save_frequency == 0:
                self._save_checkpoint()

        logger.info("Training completed!")
        self._save_checkpoint(final=True)

    def _burn_in(self) -> None:
        """Fill replay buffer with random transitions."""
        logger.info(f"Starting burn-in phase ({self.burn_in_steps} steps)")

        burn_in_steps = 0

        while burn_in_steps < self.burn_in_steps:
            state, _ = self.env.reset()
            done = False
            transitions = []

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                transitions.append((state, action, reward))
                state = next_state
                burn_in_steps += 1

                if burn_in_steps >= self.burn_in_steps:
                    break

            # Process transitions
            if len(transitions) >= self.n_steps:
                processed = compute_n_step_returns(transitions, self.gamma, self.n_steps)
                for s, a, r, ns, d in processed:
                    self.replay_buffer.push(s, a, r, ns, d)

        logger.info("Burn-in phase completed")

    def _collect_episode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Collect one episode of experience.

        Returns:
            List of (state, action, reward) transitions
        """
        state, _ = self.env.reset()
        done = False
        transitions = []
        episode_reward = 0.0

        while not done:
            action = self.algorithm.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            transitions.append((state, action, reward))
            state = next_state
            episode_reward += reward
            self.env_steps += 1

        self.episodes += 1
        self.metrics.update(episode_reward=episode_reward)

        return transitions

    def _evaluate(self) -> None:
        """Run evaluation episodes."""
        mean_score, std_score = run_evaluation(
            self.algorithm,
            self.eval_env,
            self.eval_episodes,
            self.seed + 2000
        )

        self.metrics.update(eval_score=mean_score)

        # Log results
        train_metrics = self.metrics.get_all_averages()
        logger.info(
            f"Steps: {self.env_steps} | "
            f"Episodes: {self.episodes} | "
            f"Train reward: {train_metrics.get('episode_reward', 0):.2f} | "
            f"Eval score: {mean_score:.2f} ± {std_score:.2f} | "
            f"Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Q-value: {train_metrics.get('q_value', 0):.2f}"
        )

        # Save best model
        if mean_score > self.best_eval_score:
            self.best_eval_score = mean_score
            self._save_checkpoint(best=True)

    def _save_checkpoint(self, best: bool = False, final: bool = False) -> None:
        """Save model checkpoint.

        Args:
            best: Whether this is the best model so far
            final: Whether this is the final checkpoint
        """
        if best:
            path = self.save_dir / 'best_model.pt'
        elif final:
            path = self.save_dir / 'final_model.pt'
        else:
            path = self.save_dir / f'checkpoint_{self.env_steps}.pt'

        self.algorithm.save(path)
        logger.info(f"Saved checkpoint to {path}")
