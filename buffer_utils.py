import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, batch_size=128, device='cpu'):
        self.capacity = capacity
        self.idx = 0
        self.batch_size = batch_size
        self.device = device
        self.states = torch.zeros(size=(capacity, state_dim), dtype=torch.float).to(self.device)
        self.actions = torch.zeros(size=(capacity, action_dim), dtype=torch.long).to(self.device)
        self.rewards = torch.zeros(size=(capacity, 1), dtype=torch.float).to(self.device)
        self.next_states = torch.zeros(size=(capacity, state_dim), dtype=torch.float).to(self.device)
        self.dones = torch.zeros(size=(capacity, 1), dtype=torch.long).to(self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def push(self, state, action, reward, next_state, done):
        self.states[self.idx % self.capacity] = torch.from_numpy(state).float().to(self.device)
        self.actions[self.idx % self.capacity] = torch.from_numpy(action).to(self.device)
        self.rewards[self.idx % self.capacity] = reward
        self.next_states[self.idx % self.capacity] = torch.from_numpy(next_state).float().to(self.device)
        self.dones[self.idx % self.capacity] = int(done)
        self.idx += 1

    def sample(self, batch_size=None, idx=None):
        if not batch_size:
            batch_size = self.batch_size
        if idx is None:
            # When buffer large the probability of sampling a transition more than once -> 0
            idx = np.random.randint(low=0, high=min(self.idx, self.capacity), size=batch_size)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]

    def __len__(self):
        return min(self.idx, self.capacity)

    def to_device(self, device=None):
        if device is None:
            device = self.device
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.dones = self.dones.to(device)
