"""Replay buffer implementation for off-policy RL algorithms."""
from typing import Tuple, Optional

import numpy as np
import torch

from .base import BaseReplayBuffer


class ReplayBuffer(BaseReplayBuffer):
    """Experience replay buffer for storing and sampling transitions.

    Stores transitions as tensors on specified device for efficient sampling.
    Supports n-step returns and factorised action spaces.
    """

    def __init__(
            self,
            capacity: int,
            state_dim: int,
            action_dim: int,
            batch_size: int = 128,
            device: str = 'cpu'
    ):
        """Initialise replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of action heads for factorised)
            batch_size: Default batch size for sampling
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        super().__init__(capacity, device)

        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Pre-allocate tensors for efficiency
        self.states = torch.zeros(
            size=(capacity, state_dim),
            dtype=torch.float,
            device=device
        )
        self.actions = torch.zeros(
            size=(capacity, action_dim),
            dtype=torch.long,
            device=device
        )
        self.rewards = torch.zeros(
            size=(capacity, 1),
            dtype=torch.float,
            device=device
        )
        self.next_states = torch.zeros(
            size=(capacity, state_dim),
            dtype=torch.float,
            device=device
        )
        self.dones = torch.zeros(
            size=(capacity, 1),
            dtype=torch.long,
            device=device
        )

    def push(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool
    ) -> None:
        """Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        idx = self.idx % self.capacity

        self.states[idx] = torch.from_numpy(state).float()
        self.actions[idx] = torch.from_numpy(action)
        self.rewards[idx] = reward
        self.next_states[idx] = torch.from_numpy(next_state).float()
        self.dones[idx] = float(done)

        self.idx += 1

    def sample(
            self,
            batch_size: Optional[int] = None,
            indices: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch of transitions.

        Args:
            batch_size: Number of transitions to sample (uses default if None)
            indices: Specific indices to sample (random if None)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if batch_size is None:
            batch_size = self.batch_size

        if indices is None:
            # Random sampling without replacement
            max_idx = min(self.idx, self.capacity)
            indices = np.random.randint(low=0, high=max_idx, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def to_device(self, device: str) -> None:
        """Move all tensors to specified device.

        Args:
            device: Target device ('cuda' or 'cpu')
        """
        self.device = device
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.dones = self.dones.to(device)
