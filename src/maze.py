import copy
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

from src.obstacles.obstacle_template import ObstacleTemplate


class Maze(gym.Env):
    '''
    An Amazing Maze environment

    Actions:
    Type: Discrete(4)
    Num    Action
    0         Forward
    1         Backward
    2         Right
    3         Left

    Observation:
    Type: Box

    Observation                   presented as      Shape                                  Range
    -----------                   -----             -------                                -------
    position                      [x, y]            (2,)                                   [0, N - 1]
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, obstacles=ObstacleTemplate(), seed=None):
        super(Maze, self).__init__()
        '''
        Creates a new instant of the maze environment.

        Arguments:

        N -- the size of the maze
        '''

        self.shape = obstacles.shape
        self.goal_state = obstacles.goal_state
        self.start_state = obstacles.start_state

        # Lets first do the actions
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.shape[0] - 1, self.shape[1] - 1]),
                                            shape=(2,),
                                            dtype=np.int64)

        self.action_dict = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([1, 0]),
            3: np.array([-1, 0]),
        }
        # Set geometry of the obstacles
        self.obstacle = obstacles.get_obstacles()

        # state
        self.state = None
        self.seed()
        self.steps_beyond_done = None
        self.reset()

    def seed(self, seed=None):
        '''
        create the random generator

        copy-pasted from the cartpole
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, initial_state=None):
        '''
        resets the agent to a random initial state

        Arguments:
        initial_state -- the agents state is set to this argument. If None is provided it is set randomly.
        '''
        # reset time

        self.steps_beyond_done = False

        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = self.__set_random_valid_state()

        return self.state

    def __set_random_valid_state(self):
        state = self.__get_random_state()
        while not self.is_the_new_state_allowed(state):
            state = self.__get_random_state()
        return state

    def __get_random_state(self):
        return np.random.randint(low=np.array([0, 0]),
                                 high=np.array([self.shape[0] - 1, self.shape[1] - 1]),
                                 size=(2,))

    def is_the_new_state_allowed(self, new_state):
        '''
        checks if the state is allowed:
        the state is not allowed if the agent is steping out the grid
        or on the obstacles.

        returns:
        res -- a boolean showing that the state is allowed (True) or not (False).
        '''

        res = True

        i, j = new_state

        # checking if it is getting out the world.
        if i not in range(self.shape[0]) or j not in range(self.shape[1]):
            res = False
        else:
            # setting the obstacles
            if self.obstacle[i, j] == 1:
                res = False

        return res

    def step(self, action):
        '''
        one step in the maze!

        Argument:
        action -- the chosen action

        Returns:

        state -- the new state
        done -- True if the process is terminated. The process is terminated if the agent hits the wall, or reaches the goal
        reward -- the reward is -1 for an accepted step and large value (-2 * self.N) for hitting the wall, and 0 for reaching the goal.
        info -- forget about it!
        '''

        # Let's make a copy of the state first
        obsv = copy.deepcopy(self.state)

        # Lets implement different actions
        obsv += self.action_dict[action]

        if self.is_the_new_state_allowed(obsv):
            # when the new state is allowed
            if np.array_equal(obsv, self.goal_state):
                # if we are the goal
                reward = 0
                done = True
                self.state = obsv
            else:
                # if we are not at the goal
                reward = -1
                done = False
                self.state = obsv
        else:
            # when it hitting a wall or an obstacle
            reward = -1
            done = False
            # dont do anything for the state; it remains where it was

        return self.state, reward, done, {}

    def render(self, mode='human'):
        '''
        not implemented
        '''
        if mode == "human":
            env = -1 * (self.obstacle - 1)
            plt.imshow(env, cmap="gray")
            plt.scatter([self.goal_state[0]], [self.goal_state[1]], c="r", marker="x", s=100)
            plt.scatter([self.state[0]], [self.state[1]], c="b", s=100)
            plt.show()
        if mode == 'text':
            print(self.state)

    def close(self):
        print('Good Bye!')

    def get_all_states(self):
        return np.array(
            np.meshgrid(np.arange(self.shape[0]),
                        np.arange(self.shape[1]))
        ).T.reshape(-1,2)


if __name__ == "__main__":
    maze = Maze()
    maze.render()
