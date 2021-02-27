import matplotlib.pyplot as plt
from matplotlib import interactive

from src.utility.plot import setup_plot, plot_values_and_policy, plot_simulation
from src.environment.maze import Maze

import numpy as np

import copy
import tqdm


class GreedyQLearner:
    def __init__(self, env):
        self.env = env
        self.reset()
        self.policy = None
        self.quality = None

        self.ax = None
        #Hyperparameters
        self.epsilon_0 = 1.
        self.epsilon_decay_window = 200
        self.gamma = 0.98
        self.alpha = 0.9

        self.reset()

    def reset(self):
        self.policy = self.__get_random_policy(epsilon=10)
        self.quality = np.zeros(self.policy.shape)

    def __get_random_policy(self, epsilon=0.1):
        random_policy = np.zeros((self.env.shape[0], self.env.shape[1], self.env.action_space.n))
        # Set random actions to 1
        for i in range(random_policy.shape[0]):
            for j in range(random_policy.shape[1]):
                random_index = np.random.randint(0, high=self.env.action_space.n - 1)
                random_policy[random_index] = 1
        random_policy = random_policy + epsilon  # Add some salt
        # Normalize policy
        normalization_factor = np.sum(random_policy, axis=-1)
        normalization_factor = np.expand_dims(normalization_factor, -1)
        return random_policy / normalization_factor

    def episode(self, learning_episode_id=0):
        terminated, epsilon = self.__initialize_episode(learning_episode_id)

        while not terminated:
            terminated = self.__run_episode(epsilon)

    def __initialize_episode(self, learning_episode_id):
        terminated = False
        self.env.reset()
        if np.array_equal(self.env.state, self.env.goal_state):
            terminated = True

        epsilon = self.epsilon_0 * np.exp(-1. * learning_episode_id / self.epsilon_decay_window)

        return terminated, epsilon

    def __run_episode(self, epsilon):
        state = copy.deepcopy(self.env.state)

        self.policy = self.__get_policy(epsilon)
        action_id = self.choose_action(self.env.state)

        state_prime, reward, terminated, info = self.env.step(action_id)
        self.learn_quality(state, action_id, reward, state_prime, self.quality, self.gamma, self.alpha)

        return terminated

    def __get_policy(self, epsilon):
        policy = np.full(self.quality.shape, epsilon / self.env.action_space.n)
        for state in self.env.get_all_states():
            i, j = state
            state_quality = self.quality[i, j, :]
            idx_max = np.argwhere(state_quality == np.max(state_quality))
            policy[i, j, np.random.choice(idx_max.flatten())] += 1. - epsilon
        return policy

    def choose_action(self, state):
        i, j = state
        # print(self.policy[i,j,:])
        return np.random.choice(self.env.action_space.n, p=self.policy[i, j, :])

    @staticmethod
    def learn_quality(state, action_id, reward, state_prime, quality, gamma, alpha):
        '''
        updates the Q table using one s,a,r,s'
        '''
        i, j = state.astype(int)
        iprime, jprime = state_prime.astype(int)

        correction_term = reward + gamma * np.max(quality[iprime, jprime, :]) - quality[i, j, action_id]

        quality[i, j, action_id] = quality[i, j, action_id] + alpha * correction_term

        return quality

    def learn(self, number_of_episodes=100):
        self.__initialize_plots()
        for learning_episode_id in tqdm.tqdm(range(number_of_episodes), 'learning episode'):
            self.episode(learning_episode_id)
            self.__plot_episode(learning_episode_id, simulation_interval=50)


    def __initialize_plots(self):
        self.ax = setup_plot(self.env.shape)
        plt.ion()
        interactive(True)
        plt.cla()
        self.ax.axis('off')

    def __plot_episode(self, learning_episode_id, simulation_interval=50):
        plot_values_and_policy(self)
        if (learning_episode_id % simulation_interval == simulation_interval - 1):
            if learning_episode_id > 2.*simulation_interval - 1: # Skip the first two intervals
                plot_simulation(self)



if __name__ == "__main__":
    learner = GreedyQLearner(
        Maze()
    )
    learner.learn( number_of_episodes=7*7*20*4 )