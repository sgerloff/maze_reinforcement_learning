from src.maze import Maze

from src.obstacles.prim_maze_obstacles import PrimMaze
from src.obstacles.obstacles_gui import ObstaclesGUI

from src.learner.softmax_q_learner import SoftmaxQLearner
from src.learner.greedy_q_learner import GreedyQLearner


obstacles = PrimMaze(shape=(13,13))
# obstacles = ObstaclesGUI(shape=(7,7))
maze = Maze(obstacles=obstacles)
# learner = GreedyQLearner(maze)
learner = SoftmaxQLearner(maze)

learner.learn(number_of_episodes=5000, simulation_interval=250)