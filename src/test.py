from src.maze import Maze

from src.obstacles.prim_maze_obstacles import PrimMaze
from src.obstacles.obstacles_gui import ObstaclesGUI

from src.learner.softmax_q_learner import SoftmaxQLearner
from src.learner.greedy_q_learner import GreedyQLearner


# obstacles = PrimMaze(shape=(13,13))
obstacles = ObstaclesGUI(shape=(9,9))
maze = Maze(obstacles=obstacles)
# learner = GreedyQLearner(maze)
learner = SoftmaxQLearner(maze, temperature=0.6)

learner.learn(number_of_episodes=2000, simulation_interval=100)