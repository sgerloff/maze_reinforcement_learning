from src.learner.greedy_q_learner import GreedyQLearner
import numpy as np


class SoftmaxQLearner(GreedyQLearner):
    def __init__(self, env, temperature=1.):
        super().__init__(env)
        self.temperature = temperature

    def get_policy(self, epsilon):
        exp_quality = np.exp(self.quality / self.temperature)
        sum_exp_quality = np.sum(exp_quality, axis=-1, keepdims=True)
        exp_quality = exp_quality / sum_exp_quality

        return exp_quality
