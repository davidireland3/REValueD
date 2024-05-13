import dm_control.suite as suite
import gymnasium as gym
from gymnasium import spaces
from itertools import product
import numpy as np


def run_test(alg, env, seed):
    state, _ = env.reset(seed=seed)
    done = False
    score = 0
    while not done:
        action = alg.greedy_act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
    return score


class DMSuiteWrapper(gym.Env):
    def __init__(self, domain_name, task_name, episode_len=None, seed=None):
        if seed is not None:
            self.env = suite.load(domain_name, task_name, task_kwargs={'random': seed})
        else:
            self.env = suite.load(domain_name, task_name)
        num_actions = self.env.action_spec().shape[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
        # Calculate the size of the state space
        time_step = self.env.reset()
        state_size = np.concatenate([v.flatten() for v in time_step.observation.values()]).shape[0]
        obs_high = np.array([np.inf for _ in range(state_size)], dtype=np.float32)
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high)
        self.episode_len = episode_len or self.env._step_limit
        self._time_step = None
        self.task_name = task_name
        self.domain_name = domain_name

    def reset(self, seed=None, options=None):
        if seed is None:
            self._time_step = self.env.reset()
        else:
            self.env = suite.load(self.domain_name, self.task_name, task_kwargs={'random': seed})
            self._time_step = self.env.reset()
        return np.concatenate([v.flatten() for v in self._time_step.observation.values()]), {}

    def step(self, action):
        self._time_step = self.env.step(action)
        observation, reward, termination, info = (
            np.concatenate([v.flatten() for v in self._time_step.observation.values()]),
            self._time_step.reward,
            self._time_step.last(),
            {}
        )
        if self._time_step.last():
            info['truncated'] = not self._time_step.step_type.last()
        return observation, reward, False, self._time_step.last(), info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class AtomicWrapper(gym.Wrapper):
    def __init__(self, env, bin_size=3):
        super(AtomicWrapper, self).__init__(env)
        lows = self.env.action_space.low
        highs = self.env.action_space.high
        self.action_lookups = {}
        bins = []
        for low, high in zip(lows, highs):
            bins.append(np.linspace(low, high, bin_size).tolist())
        for count, action in enumerate(product(*bins)):
            self.action_lookups[count] = list(action)
        self.num_actions = len(self.action_lookups)
        self.action_space = spaces.Discrete(self.num_actions)

    def step(self, action):
        action = self.get_continuous_action(action)
        return super().step(action)

    def get_continuous_action(self, action):
        continuous_action = self.action_lookups[action]
        return continuous_action


class FactorisedWrapper(gym.Wrapper):
    def __init__(self, env, bin_size=3):
        super(FactorisedWrapper, self).__init__(env)
        self.num_subaction_spaces = self.env.action_space.shape[0]
        if isinstance(bin_size, int):
            self.bin_size = [bin_size] * self.num_subaction_spaces
        elif isinstance(bin_size, list) or isinstance(bin_size, np.ndarray):
            assert len(bin_size) == self.num_subaction_spaces
            self.bin_size = bin_size
        lows = self.env.action_space.low
        highs = self.env.action_space.high
        self.action_lookups = {}
        for a, l, h in zip(range(self.num_subaction_spaces), lows, highs):
            self.action_lookups[a] = {}
            bins = np.linspace(l, h, self.bin_size[a])
            for count, b in enumerate(bins):
                self.action_lookups[a][count] = b
        self.action_space = spaces.MultiDiscrete(self.bin_size)

    def step(self, action):
        action = self.get_continuous_action(action)
        return super().step(action)

    def get_continuous_action(self, action):
        continuous_action = []
        for action_id, a in enumerate(action):
            continuous_action.append(self.action_lookups[action_id][a])
        return continuous_action


def make_env(task_name, task, bin_size=3, factorised=True):
    if factorised:
        return FactorisedWrapper(DMSuiteWrapper(task_name, task), bin_size=bin_size)
    else:
        return AtomicWrapper(DMSuiteWrapper(task_name, task), bin_size=bin_size)
