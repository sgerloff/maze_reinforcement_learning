from src.learner.greedy_q_learner import GreedyQLearner
import tqdm
import copy

from src.obstacles.prim_maze_obstacles import PrimMaze
from src.maze import Maze

import tensorflow as tf
import numpy as np


class NNQLeaner(GreedyQLearner):
    def __init__(self, env, model):
        super().__init__(env)
        # Define Training Examples
        self.model = model
        self.state_buffer = []
        self.action_buffer = []
        self.target_buffer = []
        self.buffer_size = 10000

    def learn(self, number_of_episodes=100, simulation_interval=50, buffer_size=10000):
        self.buffer_size = buffer_size
        self.initialize_trajectory_tape()
        self.initialize_plots()
        for learning_episode_id in tqdm.tqdm(range(number_of_episodes), 'learning episode'):
            self.train_quality_net()
            self.episode(learning_episode_id)
            self.plot_episode(learning_episode_id, simulation_interval=simulation_interval)

    def initialize_trajectory_tape(self):
        self.state_buffer = []
        self.action_buffer = []
        self.target_buffer = []
        print("Generating initial trajectory training data...")
        while len(self.state_buffer) < self.buffer_size:
            self.add_episode_to_tape()
            print(f"Buffer: {len(self.state_buffer)}")

    def add_episode_to_tape(self):
        states, actions, targets = self.episode()
        self.state_buffer.extend(states)
        self.action_buffer.extend(actions)
        self.target_buffer.extend(targets)
        return self.crop_to_tape_length()

    def crop_to_tape_length(self):
        self.state_buffer = self.state_buffer[-self.buffer_size:]
        self.action_buffer = self.action_buffer[-self.buffer_size:]
        self.target_buffer = self.target_buffer[-self.buffer_size:]
        return None

    def train_quality_net(self):
        self.model.fit(
            np.array(self.state_buffer), np.array(self.target_buffer), epochs=10
        )

    def episode(self, learning_episode_id=0):
        terminated, epsilon = self.initialize_episode(learning_episode_id)
        states = []
        actions = []
        targets = []
        while not terminated:
            states.append(copy.deepcopy(self.env.state))
            terminated, action_id, target = self.episode_step(epsilon)
            actions.append(action_id)
            targets.append(target)
        return states, actions, targets

    def episode_step(self, epsilon):
        state = copy.deepcopy(self.env.state)

        self.policy = self.get_policy(epsilon, self.env.state)
        action_id = self.choose_action(self.env.state)

        state_prime, reward, terminated, info = self.env.step(action_id)
        self.learn_quality(state, action_id, reward, state_prime, self.quality, self.gamma, self.alpha)
        target = self.quality[state[0], state[1], :]

        return terminated, action_id, target

    def get_policy(self, epsilon, state):
        i, j = state
        policy = np.full(self.quality.shape, epsilon / self.env.action_space.n)
        state_quality = self.model.predict(np.reshape(state, (1,2)))
        idx_max = np.argwhere(state_quality == np.max(state_quality))
        policy[i, j, : ] = np.full(self.env.action_space.n, epsilon / self.env.action_space.n)
        policy[i, j, np.random.choice(idx_max.flatten())] += 1. - epsilon

        return policy



if __name__ == "__main__":
    obstacles = PrimMaze(shape=(7, 7))
    maze = Maze(obstacles=obstacles)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=2, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="linear"))
    model.compile(
        loss="mean_squared_error",
        optimizer="adam"
    )

    learner = NNQLeaner(maze, model)
    learner.learn()
    print(learner.target_buffer[0])
