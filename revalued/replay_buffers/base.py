"""Base class for replay buffers."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch


class BaseReplayBuffer(ABC):
    """Abstract base class for replay buffers."""

    def __init__(self, capacity: int, device: str = 'cpu'):
        """Initialise base replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.idx = 0

    @abstractmethod
    def push(self, *args) -> None:
        """Add transition to buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """Sample batch of transitions."""
        pass

    def __len__(self) -> int:
        """Return current size of buffer."""
        return min(self.idx, self.capacity)
