# Deep Reinforcement Learning Algorithms Framework

A  Python framework for implementing and testing t deep reinforcement learning algorithms on OpenAI Gymnasium environments.

## 📋 Overview

This project provides  implementations of several RL algorithms including DQN, DDQN, PPO, SAC, and TD3. It features a flexible configuration system, automated experiment tracking, and comprehensive evaluation tools.

**Repository:** [marziof/deep-rl-project](https://github.com/marziof/deep-rl-project)

## Supported Algorithms & Environments

| Algorithm | Environments |
|-----------|--------------|
| **DQN** | CartPole-v1, LunarLander-v3|
| **DDQN** | CartPole-v1, LunarLander-v3 |
| **PPO** | LunarLander-v3, Pendulum-v1, CartPole-v1 |
| **SAC** | Pendulum-v1, InvertedDoublePendulum-v5|
| **TD3** | Pendulum-v1, InvertedDoublePendulum-v5|

##  Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/marziof/deep-rl-project.git
cd deep-rl-project

# Create virtual environment
python -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
Some of us ran into some issues when installing the dependencies. We do not guarantee that our requirements will work perfectly. 
## Quick Start

### 1. Configure an experiment

Edit or create a configuration file in `configs/`:

```yaml
experiment:
  env_name: "Pendulum-v1"        # Choose environment
  algo: "sac"                    # Choose algorithm
  n_episodes: 300                # Number of training episodes
  n_iterations: 61               # Iterations per episode
  eval_interval: 10              # Evaluation frequency
  seeds: [0, 42, 100]            # Random seeds for reproducibility
  exp_name: "my_experiment"      # Experiment name for results folder
  create_videos: false

algos:
  sac:
    lr: 0.0003
    tau: 0.005
    alpha: 0.9
    batch_size: 256
    buffer_capacity: 200000
    gamma: 0.99
```

### 2. Run an experiment

```bash
python experiment_runner.py --config configs/conf.yaml
```

Results are saved to `exp_name/` directory containing:
- Training logs and metrics
- Configuration file snapshot
- Performance plots
- Optional videos (if enabled)

##Project structure

```
├── src/                          # Core framework
│   ├── algorithms/               # Algorithm implementations
│   │   ├── dqn.py
│   │   ├── ddqn.py
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   └── td3.py
│   ├── networks/                 # Neural network architectures
│   │   ├── mlp.py
│   │   └── actor_net.py
│   ├── buffers/                  # Experience replay buffers
│   │   └── replay_buffer.py
│   ├── utils/                    # Utilities
│   │   ├── env.py
│   │   ├── plotting.py
│   │   ├── logger.py
│   │   ├── seed.py
│   │   └── stats.py
│   ├── evaluation.py             # Evaluation utilities
│   └── train.py                  # Training orchestration
├── configs/                      # Configuration files
│   └── conf.yaml                 # Default configuration
├── experiments/                  # Experiment-specific scripts
├── experiment_runner.py          # Main entry point
├── plot_generator.py             # Visualization tools
└── requirements.txt              # Dependencies
```

## How it works

### Training pipeline

1. **experiment_runner.py** - Main entry point
   - Reads configuration file
   - Orchestrates experiment execution
   - Generates plots and saves results

2. **src/train.py** - Training logic
   - `run_experiments()` - Manages multiple experimental runs
   - `run_experiment()` - Single experiment execution
   - Environment creation and algorithm selection

3. **Algorithm-specific scripts** - In `src/algorithms/`
   - `run_episode()` - Handles training loops specific to each algorithm
   - Network updates and policy learning

## Example usage

```bash
# Run with default config
python experiment_runner.py --config configs/conf.yaml

# Custom configuration
python experiment_runner.py --config configs/my_config.yaml
```

Results are automatically saved with timestamped folders if `exp_name` is not specified (format: `YYYY-MM-DD-HH-MM`).

## Configuration guide

Key parameters for `configs/conf.yaml`:

| Parameter | Description |
|-----------|------------|
| `env_name` | OpenAI Gymnasium environment |
| `algo` | Algorithm choice: dqn, ddqn, ppo, sac, td3 |
| `n_episodes` | Total training episodes |
| `n_iterations` | Steps per episode |
| `eval_interval` | Evaluate every N episodes |
| `seeds` | List of random seeds for reproducibility |
| `exp_name` | Output folder name |
| `create_videos` | Record training videos |

Algorithm-specific hyperparameters are configured under `algos.<algorithm_name>` in the config file.

## Logging
- Training metrics logged to stdout

## Generating plots
The framework contains a plot generator, called `plot_generator.py`, which aggregates training logs across different algorithms and hyperparameters to generate comparative figures. The `plot_generator.py` script is explicitly tailored to look for the specific experimental directories and log filenames structured within our project directory (`results/data/`). If you alter the default file naming or folder structure layout, you may need to update the file path variables directly in the script. Currently, it is possible to reproduce all the plots for hyperparameter tuning in case of SAC, amd the comparison of algos with different environments. 

# Compare algorithms on Pendulum-v1
```bash
python plot_generator.py --env Pendulum-v1
```

# Compare algorithms on InvertedDoublePendulum-v5
```bash
python plot_generator.py --env InvertedDoublePendulum-v1
```

To plot results from a hyperparameter tuning sweep (such as comparing variations of the temperature parameter $\alpha$ in SAC or exploration noise $\sigma$ in TD3), specify the respective parameter keyword as the target value:

# Compare algorithms on Pendulum-v1
```bash
python plot_generator.py --env Alpha
```

# Compare algorithms on InvertedDoublePendulum-v5
```bash
python plot_generator.py --env Sigma
```
Notice that to test other hyperparameters on other environments, you must customize the paths and plot_generator.py to be compatible with the desired hyperparameter.



