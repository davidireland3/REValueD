from algorithms import REValueD
from environment_utils import make_env, run_test
import numpy as np
import random
import torch

SEED = 174428
offset = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 256
hidden_size = 512
critic_lr = 1e-4
gamma = 0.99
bin_size = [2, 3, 4, 5, 6, 7]  # this can be an integer for fixed bin sizes in each dim, or a list of desired bin sizes.
n_steps = 3
task_name = "walker"
task = "walk"
tau = 0.005
max_env_interactions = 10_000_000
update_ratio = 5
num_updates = 1
ensemble_size = 10

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

env = make_env(task_name, task, bin_size)
test_env = make_env(task_name, task, bin_size)
state_dim = test_env.observation_space.shape[0]
action_space = env.action_space
dqn = REValueD(state_dim=state_dim, action_space=action_space, hidden_size=hidden_size,
               batch_size=batch_size, gamma=gamma, tau=tau, lr=critic_lr, task_name=task_name, task=task, n_steps=n_steps,
               seed=SEED, memory_size=100_000, device=device, ensemble_size=ensemble_size, update_type='REDQ')
dqn.make_critic()

ep_count = 0
memory = []
while len(memory) < 10000:
    done = False
    state, _ = env.reset()
    score = 0
    ep_count += 1
    n_step_buffer = []
    while not done:

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        n_step_buffer.append((state, action, reward))
        if len(n_step_buffer) == n_steps:
            state_0, action_0, _ = n_step_buffer[0]
            disc_returns = np.sum([r * gamma ** count for count, (_, _, r) in enumerate(n_step_buffer)], axis=0)
            memory.append((state_0, action_0, disc_returns, next_state, terminated))
            n_step_buffer.pop(0)
        if terminated:
            while n_step_buffer:
                state_0, action_0, _ = n_step_buffer[0]
                disc_returns = np.sum([r * gamma ** count for count, (_, _, r) in enumerate(n_step_buffer)], axis=0)
                memory.append((state_0, action_0, disc_returns, next_state, terminated))
                n_step_buffer.pop(0)

        state = next_state
        score += reward
    print(f"Burn-in episode {ep_count}")
dqn.remember(memory)

episode = 0
env_interactions = 0
memory = []
while env_interactions < max_env_interactions:
    episode += 1
    done = False
    state, _ = env.reset()
    score = 0
    n_step_buffer = []
    while not done:
        action = dqn.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        env_interactions += 1
        done = terminated or truncated

        n_step_buffer.append((state, action, reward))
        if len(n_step_buffer) == n_steps:
            state_0, action_0, _ = n_step_buffer[0]
            disc_returns = np.sum([r * gamma ** count for count, (_, _, r) in enumerate(n_step_buffer)], axis=0)
            memory.append((state_0, action_0, disc_returns, next_state, terminated))
            n_step_buffer.pop(0)
        if terminated:
            while n_step_buffer:
                state_0, action_0, _ = n_step_buffer[0]
                disc_returns = np.sum([r * gamma ** count for count, (_, _, r) in enumerate(n_step_buffer)], axis=0)
                memory.append((state_0, action_0, disc_returns, next_state, terminated))
                n_step_buffer.pop(0)

        if env_interactions % update_ratio == 0:
            dqn.remember(memory)
            memory = []
            for i in range(num_updates):
                dqn.experience_replay()
            if env_interactions % 5000 == 0:
                test_scores = []
                for j in range(5):
                    test_scores.append(run_test(dqn, test_env, SEED + j * offset))
                print(f"Env steps {env_interactions}. Grad steps: {dqn.grad_steps}. Score: {np.mean(test_scores)}")
        state = next_state
        score += reward
