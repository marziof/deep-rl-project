# deep-rl-project

Implementation and testing of four deep reinforcement-learning algorithms on
[Gymnasium](https://gymnasium.farama.org/) environments:

| Algorithm | Action space | Key paper |
|-----------|-------------|-----------|
| **DQN**   | Discrete    | Mnih et al. (2015) |
| **PPO**   | Discrete    | Schulman et al. (2017) |
| **SAC**   | Continuous  | Haarnoja et al. (2018) |
| **TD3**   | Continuous  | Fujimoto et al. (2018) |

---

## Project structure

```
deep-rl-project/
├── configs/           # YAML hyper-parameter files
│   ├── dqn.yaml
│   ├── ppo.yaml
│   ├── sac.yaml
│   └── td3.yaml
├── src/
│   ├── algorithms/    # DQN, PPO, SAC, TD3 implementations
│   ├── networks/      # Shared MLP building blocks
│   ├── buffers/       # Replay buffer
│   ├── utils/         # Logger, seed, plotting helpers
│   └── train.py       # Unified training entry-point
├── experiments/       # Per-environment experiment notes
├── results/
│   ├── logs/          # CSV training logs
│   └── plots/         # Saved training-curve figures
└── report/            # Written report
```

---

## Installation

```bash
pip install -r requirements.txt
# (optional) editable install
pip install -e .
```

---

## Training

```bash
# DQN on CartPole
python src/train.py --algo dqn --config configs/dqn.yaml

# PPO on CartPole
python src/train.py --algo ppo --config configs/ppo.yaml

# SAC on MountainCarContinuous
python src/train.py --algo sac --config configs/sac.yaml

# TD3 on MountainCarContinuous
python src/train.py --algo td3 --config configs/td3.yaml

# Override env or seed at the command line
python src/train.py --algo dqn --config configs/dqn.yaml --env LunarLander-v3 --seed 0
```

Logs are written to `results/logs/<run_name>/progress.csv`.

---

## Plotting results

```python
from src.utils.plotting import plot_training_curve

plot_training_curve(
    "results/logs/<run_name>/progress.csv",
    y_col="eval_mean_reward",
    save_path="results/plots/dqn_cartpole.png",
    window=5,
)
```
