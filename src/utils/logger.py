# File to log training data

import numpy as np


class Logger:
    def __init__(self):
        self.episode_rewards = []
        self.eval_rewards = []
        self.losses = []
        self.epsilons = []

    def reset(self):
        self.episode_rewards = []
        self.eval_rewards = []
        self.losses = []
        self.epsilons = []

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