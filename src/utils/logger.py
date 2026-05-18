# File to log training data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, algo_name, env_name, seed=0):
        self.episode_rewards = []
        self.eval_rewards = []
        self.losses = []
        self.epsilons = []
        self.step_rewards = []
        self.algo_name = algo_name
        self.env_name = env_name
        self.seed = seed

    def reset(self):
        self.episode_rewards = []
        self.eval_rewards = []
        self.losses = []
        self.epsilons = []
        self.step_rewards = []

    def log_episode_reward(self, r):
        self.episode_rewards.append(r)

    def log_eval_reward(self, r):
        self.eval_rewards.append(r)

    def log_loss(self, loss):
        self.losses.append(loss)

    def log_epsilon(self, eps):
        self.epsilons.append(eps)

    def moving_average(self, window=100):
        return np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
    
    def log_step_rewards(self, rewards):
        self.step_rewards.append(rewards)

    def to_dataframe(self):
        rows = []

        for i, r in enumerate(self.episode_rewards):
            rows.append({"algo": self.algo_name, "env": self.env_name, "seed": self.seed, "step": i, "metric": "episode_reward", "value": r})

        for i, r in enumerate(self.step_rewards): 
            rows.append({ "algo": self.algo_name, "env": self.env_name, "seed": self.seed, "step": i, "metric": "step_reward", "value": r})

        for i, r in enumerate(self.eval_rewards): 
            rows.append({"algo": self.algo_name, "env": self.env_name, "seed": self.seed, "step": i, "metric": "eval_reward", "value": r})

        for i, l in enumerate(self.losses): 
            rows.append({"algo": self.algo_name, "env": self.env_name, "seed": self.seed, "step": i, "metric": "loss", "value": l})

        for i, e in enumerate(self.epsilons):
            rows.append({"algo": self.algo_name, "env": self.env_name, "seed": self.seed, "step": i, "metric": "epsilon", "value": e})

        return pd.DataFrame(rows)