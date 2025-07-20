# REValueD: Regularised Ensemble Value Decomposition

A deep reinforcement learning framework implementing the REValueD algorithm for improved exploration and stability in factorised action spaces. This project extends Decoupled Q-Networks (DecQN) with ensemble methods to reduce overestimation bias and enhance exploration through regularised value functions.

## Overview

REValueD is designed for environments with high-dimensional discrete action spaces, particularly those that can be factorised into multiple independent action dimensions. The algorithm combines:

- **Ensemble Q-Networks**: Multiple Q-networks with shared experience for improved stability
- **Factorised Action Spaces**: Efficient learning in multi-dimensional discrete action spaces
- **Randomised Exploration**: Enhanced exploration through randomised ensemble member selection
- **Multiple Update Strategies**: Support for Mean, REDQ, and DecQN update methods

## Key Features

- **DecQN Implementation**: Baseline algorithm for factorised action spaces
- **REValueD Algorithm**: Ensemble extension with configurable update strategies
- **Efficient Neural Networks**: Vectorised implementations for parallel ensemble computation
- **Comprehensive Training Framework**: Complete training loop with evaluation and checkpointing
- **DeepMind Control Suite Integration**: Built-in support for DMC environments
- **Flexible Configuration**: Configurable hyperparameters and training settings

## Algorithm Details

### DecQN (Decoupled Q-Network)
- Learns separate utility values for each action dimension, which are averaged to a single Q-value
- Enables efficient learning in factorised action spaces
- Uses double Q-learning for reduced overestimation bias

### REValueD (Regularised Ensemble Value Decomposition)
- Extends DecQN with an ensemble of Q-networks
- Supports three update strategies:
  - **Mean**: Uses ensemble mean for target computation
  - **REDQ**: Takes the minimum of a random sample of the ensemble for the target
  - **DecQN**: Independent training of ensemble members
- Randomised action selection using different ensemble members

## Project Structure

```
revalued/
├── algorithms/          # RL algorithm implementations
│   ├── base.py         # Abstract base class for algorithms
│   ├── decqn.py        # Decoupled Q-Network implementation
│   └── revalued.py     # REValueD algorithm
├── networks/           # Neural network architectures
│   ├── base.py         # Base network classes
│   ├── layers.py       # Custom layers (vectorised, residual)
│   └── q_networks.py   # Q-network implementations
├── replay_buffers/     # Experience replay implementations
│   ├── base.py         # Abstract base replay buffer
│   └── replay_buffer.py # Standard replay buffer
├── training/           # Training framework
│   └── trainer.py      # Main training loop and evaluation
└── utils/             # Utilities and helper functions
    ├── metrics.py      # Metrics tracking and aggregation
    └── training.py     # Training utilities and environment setup
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Gymnasium
- NumPy
- Loguru
- DMC Datasets (for DeepMind Control Suite environments)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/davidireland3/REValueD.git
cd REValueD
git checkout refactor
```

2. Install dependencies:
```bash
pip install torch gymnasium numpy loguru
pip install dmc-datasets  # For DMC environment support
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Basic Training Example

```python
from revalued.algorithms import REValueD
from revalued.training import Trainer
from revalued.utils import make_env

# Configuration
config = {
    'environment': {
        'domain': 'cartpole',
        'task': 'swingup',
        'bin_size': 3,
        'factorised': True
    },
    'algorithm': {
        'ensemble_size': 10,
        'update_type': 'Mean',
        'batch_size': 256,
        'learning_rate': 1e-3,
        'gamma': 0.99
    },
    'training': {
        'max_env_steps': 1000000,
        'eval_frequency': 10000,
        'save_frequency': 50000
    },
    'replay_buffer': {
        'capacity': 1000000,
        'burn_in_steps': 10000
    },
    'experiment': {
        'seed': 42
    }
}

# Create environment and algorithm
env = make_env(config['environment']['domain'], 
               config['environment']['task'])
state_dim = env.observation_space.shape[0]

algorithm = REValueD(
    state_dim=state_dim,
    action_space=env.action_space,
    ensemble_size=config['algorithm']['ensemble_size'],
    update_type=config['algorithm']['update_type'],
    **config['algorithm']
)

# Train
trainer = Trainer(algorithm, config)
trainer.train()
```

### Algorithm Comparison

```python
# DecQN baseline
decqn = DecQN(state_dim=state_dim, action_space=env.action_space)

# REValueD variants
revalued_mean = REValueD(state_dim, env.action_space, update_type='Mean')
revalued_redq = REValueD(state_dim, env.action_space, update_type='REDQ')
revalued_decqn = REValueD(state_dim, env.action_space, update_type='DecQN')
```

## Configuration

### Algorithm Parameters

- `ensemble_size`: Number of Q-networks in ensemble (default: 10)
- `update_type`: Target computation strategy ('Mean', 'REDQ', 'DecQN')
- `epsilon_start`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Exploration decay rate (default: 0.999)
- `n_steps`: N-step return horizon (default: 1)
- `grad_clip`: Gradient clipping threshold (default: 40.0)

### Training Parameters

- `max_env_steps`: Maximum environment steps
- `update_ratio`: Environment steps per algorithm update
- `eval_frequency`: Steps between evaluations
- `burn_in_steps`: Random exploration steps before training

### Network Architecture

- `hidden_size`: Hidden layer dimension (default: 256)
- `learning_rate`: Adam optimiser learning rate (default: 1e-3)
- `tau`: Target network update rate (default: 0.005)

## Evaluation

The training framework includes built-in evaluation:

```python
from revalued.utils import run_evaluation

# Evaluate trained algorithm
mean_score, std_score = run_evaluation(
    algorithm=trained_algorithm,
    env=eval_env,
    num_episodes=10,
    seed=42
)
print(f"Evaluation: {mean_score:.2f} ± {std_score:.2f}")
```

## Logging and Checkpoints

- Training metrics are logged using Loguru
- Model checkpoints saved at specified intervals
- Best performing models automatically saved
- Metrics include loss, Q-values, episode rewards, and evaluation scores
