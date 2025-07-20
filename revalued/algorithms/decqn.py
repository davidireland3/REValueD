"""Decoupled Q-Network (DecQN) implementation."""
import copy
from typing import Dict

import numpy as np
import torch
from torch.nn import HuberLoss
from torch.nn.utils import clip_grad_norm_

from .base import BaseAlgorithm


class DecQN(BaseAlgorithm):
    """Decoupled Q-Network for factorised action spaces.

    This algorithm learns separate Q-values for each action dimension,
    enabling efficient learning in high-dimensional discrete action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        action_space,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        n_steps: int = 1,
        grad_clip: float = 40.0,
        **kwargs
    ):
        """Initialise DecQN.

        Args:
            state_dim: Dimension of state space
            action_space: MultiDiscrete action space
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            n_steps: Number of steps for n-step returns
            grad_clip: Gradient clipping threshold
            **kwargs: Additional arguments passed to BaseAlgorithm
        """
        super().__init__(state_dim, action_space, **kwargs)

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Algorithm parameters
        self.n_steps = n_steps
        self.grad_clip = grad_clip

        # Action space info
        self.num_heads = len(action_space)
        self.max_action_dim = max(space.n for space in action_space)

        # Loss function
        self.loss_fn = HuberLoss()

        # Build networks
        self.build_networks()

    def build_networks(self) -> None:
        """Initialise Q-networks and optimiser."""
        # Import here to avoid circular imports
        from ..networks import DecoupledQNetwork

        # Create action mask for invalid actions
        self.action_mask = self._create_action_mask()

        # Initialise networks
        self.critic = DecoupledQNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_size,
            num_actions=self.max_action_dim,
            num_heads=self.num_heads
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.optimiser = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate
        )

    def act(self, state: np.ndarray) -> np.ndarray:
        """Select action using epsilon-greedy exploration.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore
            self.epsilon = max(
                self.epsilon * self.epsilon_decay,
                self.epsilon_min
            )
            return self.action_space.sample()
        else:
            # Exploit
            return self.greedy_act(state)

    def greedy_act(self, state: np.ndarray) -> np.ndarray:
        """Select action greedily based on Q-values.

        Args:
            state: Current state

        Returns:
            Greedy action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.critic(state_tensor) + self.action_mask
            actions = q_values.argmax(dim=-1).cpu().numpy().flatten()

        return actions

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one gradient update on Q-network.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of metrics
        """
        # Get Q-values for selected actions
        q_values = self.critic(states) + self.action_mask
        selected_q_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_value = selected_q_values.mean(dim=-1, keepdim=True)

        # Compute targets using double Q-learning
        with torch.no_grad():
            # Get best actions from online network
            next_q_values = self.critic(next_states) + self.action_mask
            next_actions = next_q_values.argmax(dim=-1)

            # Evaluate actions using target network
            next_q_values_target = self.critic_target(next_states) + self.action_mask
            next_q_value = next_q_values_target.gather(
                -1, next_actions.unsqueeze(-1)
            ).squeeze(-1).mean(dim=-1, keepdim=True)

            # Compute n-step targets
            targets = rewards + (self.gamma ** self.n_steps) * (1 - dones) * next_q_value

        # Compute loss
        loss = self.loss_fn(q_value, targets)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimiser.step()

        # Update counters and target network
        self.grad_steps += 1
        self.update_target_networks()

        return {
            'loss': loss.item(),
            'q_value': q_value.mean().item(),
            'epsilon': self.epsilon
        }

    def _create_action_mask(self) -> torch.Tensor:
        """Create mask for invalid actions in factorised action space.

        Returns:
            Action mask tensor
        """
        mask = []
        for subaction_space in self.action_space:
            sub_mask = []
            for j in range(self.max_action_dim):
                if j < subaction_space.n:
                    sub_mask.append(0)
                else:
                    sub_mask.append(-np.inf)
            mask.append(sub_mask)
        return torch.FloatTensor([mask]).to(self.device)
