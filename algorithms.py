from buffer_utils import ReplayBuffer
import copy
from networks import DecoupledQNetwork, EnsembleDecoupledQNetwork
import numpy as np
import pickle
import torch
from torch.nn import HuberLoss
from torch.nn.utils import clip_grad_norm_
import wandb


class DecQN:
    def __init__(self, state_dim, num_heads, num_actions, hidden_size, batch_size=512, gamma=0.99, tau=0.005,
                 epsilon_min=0.05, lr=1e-3, task_name=None, task=None,  n_steps=1, seed=None, memory_size=100_000,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self._device = device
        self._state_dim = state_dim
        self._critic_lr = lr
        self._epsilon_min = epsilon_min
        self._state_dim = state_dim
        self._num_heads = num_heads
        self._num_actions = num_actions
        self.critic = DecoupledQNetwork(self._state_dim, hidden_size, self._num_actions, self._num_heads).to(self._device)
        self.optimiser = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target = copy.deepcopy(self.critic).to(self._device)
        self._batch_size = batch_size
        self._gamma = gamma
        self._n_steps = n_steps
        self.grad_steps = 0
        self._tau = tau
        self._SEED = seed
        self.epsilon = 1
        self._exploration_decay = 0.999
        self._hidden_size = hidden_size

        self.task_name = task_name
        self.task = task

        self.test_scores = []
        self._alg_name = "DecQN"

        self.memory = ReplayBuffer(memory_size, state_dim, num_heads, batch_size, device)
        self.loss_fn = HuberLoss()

    def greedy_act(self, state):
        state = torch.FloatTensor(state).view(1, -1).to(self._device)
        with torch.no_grad():
            values = self.critic(state).squeeze(dim=0)
        action = values.argmax(dim=1).cpu().numpy()
        return action

    def act(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(low=0, high=self._num_actions, size=self._num_heads)
            self.epsilon = max(self.epsilon * self._exploration_decay, self._epsilon_min)
        else:
            state = torch.FloatTensor(state).view(1, -1).to(self._device)
            with torch.no_grad():
                values = self.critic(state).squeeze(dim=0)
            action = values.argmax(dim=1).cpu().numpy()
        return action

    def experience_replay(self):
        states, actions, rewards, next_states, dones = self.get_batch()

        utility_values = self.critic.forward(states)
        selected_utility_values = utility_values.gather(-1, actions.unsqueeze(dim=-1)).squeeze(dim=-1)
        q_vals = selected_utility_values.mean(dim=-1, keepdim=True)
        with torch.no_grad():
            idx = self.critic.forward(next_states).argmax(dim=-1)
            target_utilities = self.critic_target.forward(next_states).gather(-1, idx.unsqueeze(dim=-1)).squeeze(dim=-1)
            target_q_vals = target_utilities.mean(dim=-1, keepdim=True)
            targets = (rewards + (self._gamma ** self._n_steps) * (1 - dones) * target_q_vals)

        self.optimiser.zero_grad()
        loss = self.loss_fn(q_vals, targets)
        loss.backward()
        clip_grad_norm_(self.critic.parameters(), 40)
        self._store_critic_grads()
        self.optimiser.step()

        self.grad_steps += 1
        self._update_target()

    def get_batch(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device)

        return states, actions, rewards, next_states, dones

    def _update_target(self):
        if 1 > self._tau >= 0:
            for real, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.copy_(real.data * self._tau + target.data * (1 - self._tau))

        else:
            assert int(self._tau) == self._tau, "tau is not an integer"
            if self.grad_steps % self._tau == 0:
                self.critic_target = copy.deepcopy(self.critic).to(self._device)

    def save_model(self, task_name=None, task=None, level=None, seed=None):
        torch.save(self.critic.state_dict(), f"networks/{self._alg_name}-{task_name}-{task}-{level}-seed-{seed}")

    def load_model(self, task_name=None, task=None, level=None, seed=None):
        self.critic.load_state_dict(torch.load(f"networks/{self._alg_name}-{task_name}-{task}-{level}-seed-{seed}",
                                               map_location=torch.device(self._device)))

    def remember(self, memory):
        for state, action, reward, next_state, done in memory:
            self.memory.push(state, action, reward, next_state, done)


class REValueD(DecQN):
    def __init__(self, update_type='Mean', ensemble_size=10, **kwargs):
        super(REValueD, self).__init__(**kwargs)
        self.alg_name = 'REValueD'
        self.update_type = update_type

        self.critic = EnsembleDecoupledQNetwork(self._state_dim, self._hidden_size, self._num_actions,
                                                self._num_heads, ensemble_size).to(self._device)
        self.optimiser = torch.optim.Adam(self.critic.parameters(), lr=self._critic_lr)
        self.critic_target = copy.deepcopy(self.critic).to(self._device)
        self._ensemble_size = ensemble_size

    def get_batch(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = states.to(self._device)
        actions = actions.to(self._device).view(self._batch_size, 1, -1).repeat(1, self._ensemble_size, 1)
        rewards = rewards.to(self._device).view(self._batch_size, 1).repeat(1, self._ensemble_size)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device).view(self._batch_size, 1).repeat(1, self._ensemble_size)

        return states, actions, rewards, next_states, dones

    def experience_replay(self, use_cql=False, alpha=1):
        states, actions, rewards, next_states, dones = self.get_batch()
        utilities = self.critic.forward(states)
        selected_utilities = utilities.gather(-1, actions.unsqueeze(dim=-1)).squeeze(dim=-1)
        q_vals = selected_utilities.mean(dim=-1)
        with torch.no_grad():
            targets = self._get_targets(next_states)
            targets = (rewards + (self._gamma ** self._n_steps) * (1 - dones) * targets)

        self.optimiser.zero_grad()
        loss = self.loss_fn(q_vals, targets).mean(dim=0).sum()
        loss.backward()
        clip_grad_norm_(self.critic.parameters(), 40)
        self.optimiser.step()
        self.grad_steps += 1
        self._update_target()

    def _get_targets(self, next_states):
        """
        method for obtaining targets for the q-network. options are: DecQN, which essentially trains an ensemble of indept
        deceqn networks. redq, which uses the approach from https://openreview.net/forum?id=AY8zfZm0tDd and takes the minimum
        of two randomly sampled estimates from the ensemble, and mean which uses the mean value of the ensemble.
        :param next_states: next states used to bootstrap from
        :return: targets: target q-values
        """
        with torch.no_grad():
            if self.update_type == "DecQN":
                idx = self.critic.forward(next_states).argmax(dim=-1)
                targets = self.critic_target.forward(next_states).gather(-1, idx.unsqueeze(dim=-1)).squeeze(dim=-1).mean(dim=-1)
            elif self.update_type == "REDQ":
                targets = self.critic_target.forward(next_states).max(dim=-1)[0].mean(dim=-1)
                idx = np.random.choice(range(self._ensemble_size), size=2, replace=False)
                targets = targets[:, idx].min(dim=-1, keepdim=True)[0].repeat(1, self._ensemble_size)
            elif self.update_type == "Mean":
                targets = self.critic_target.forward(next_states).mean(dim=1).max(dim=-1)[0].mean(dim=-1, keepdim=True).repeat(1, self._ensemble_size)
            else:
                raise TypeError('update type not supported')
        return targets

    def greedy_act(self, state):
        state = torch.FloatTensor(state).view(1, -1).to(self._device)
        with torch.no_grad():
            values = self.critic.forward(state)  # 1 x ensemble_size x num_heads x num_actions
        action = values.mean(dim=1).argmax(dim=-1).cpu().flatten().numpy()
        return action

    def act(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(low=0, high=self._num_actions, size=self._num_heads)
            self.epsilon = max(self.epsilon * self._exploration_decay, self._epsilon_min)
        else:
            state = torch.FloatTensor(state).view(1, -1).to(self._device)
            with torch.no_grad():
                values = self.critic.forward(state)
            action = values[0][np.random.randint(self._ensemble_size)].argmax(dim=-1).cpu().flatten().numpy()
        return action
