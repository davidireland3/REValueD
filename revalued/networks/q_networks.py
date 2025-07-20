"""Q-network architectures for DecQN and REValueD algorithms."""
import torch
import torch.nn as nn

from .base import BaseQNetwork
from .layers import MLPResidualLayer, VectorizedLinear, VectorizedLinearHead, VectorisedMLPResidualLayer


class DecoupledQNetwork(BaseQNetwork):
    """Decoupled Q-Network for factorised action spaces.

    This network outputs separate Q-values for each action dimension,
    allowing efficient learning in factorised action spaces.
    """

    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int, num_heads: int):
        """Initialise DecoupledQNetwork.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_actions: Maximum number of actions per head
            num_heads: Number of action dimensions (heads)
        """
        super().__init__(state_dim, hidden_dim, num_actions)
        self.num_heads = num_heads

        # Network layers
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.resnet = MLPResidualLayer(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = VectorizedLinear(hidden_dim, num_actions, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values of shape (batch_size, num_heads, num_actions)
        """
        # Shared layers
        x = self.input_layer(x)
        x = self.layer_norm(self.resnet(x))

        # Expand for vectorised computation
        x = x.unsqueeze(dim=0).repeat(self.num_heads, 1, 1)

        # Apply separate linear transformation for each head
        vals = self.output_heads(x).transpose(0, 1)

        return vals


class EnsembleDecoupledQNetwork(BaseQNetwork):
    """Ensemble of Decoupled Q-Networks for REValueD algorithm.

    This network maintains multiple Q-networks in an ensemble,
    each with separate parameters but computed in parallel for efficiency.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_heads: int,
        ensemble_size: int
    ):
        """Initialise EnsembleDecoupledQNetwork.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_actions: Maximum number of actions per head
            num_heads: Number of action dimensions (heads)
            ensemble_size: Number of networks in ensemble
        """
        super().__init__(state_dim, hidden_dim, num_actions)
        self.num_heads = num_heads
        self.ensemble_size = ensemble_size

        # Ensemble network layers
        self.input_layer = VectorizedLinear(state_dim, hidden_dim, ensemble_size)
        self.resnet_layer = VectorisedMLPResidualLayer(hidden_dim, ensemble_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = VectorizedLinearHead(hidden_dim, num_actions, ensemble_size, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble network.

        Args:
            x: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values of shape (batch_size, ensemble_size, num_heads, num_actions)
        """
        batch_size = x.shape[0]

        # Expand input for ensemble
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1).repeat(1, self.ensemble_size, 1)

        # Reshape for vectorised computation
        x = x.transpose(0, 1)

        # Shared layers (computed in parallel for all ensemble members)
        x = self.input_layer(x)
        x = self.layer_norm(self.resnet_layer(x))

        # Expand for heads
        x = x.unsqueeze(dim=1).repeat(1, self.num_heads, 1, 1)

        # Apply output heads
        q_values = self.output_heads(x)

        # Reshape to (batch_size, ensemble_size, num_heads, num_actions)
        q_values = q_values.transpose(1, 2).transpose(0, 1)

        return q_values
