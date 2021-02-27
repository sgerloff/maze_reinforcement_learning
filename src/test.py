from src.environment.maze import Maze
from src.learner.greedy_q_learner import GreedyQLearner
from src.obstacles.prim_maze_obstacles import PrimMaze
from src.obstacles.obstacles_gui import ObstaclesGUI

# obstacles = PrimMaze(shape=(13,13))
obstacles = ObstaclesGUI(shape=(7,7))
maze = Maze(obstacles=obstacles)
learner = GreedyQLearner(maze)

learner.learn(number_of_episodes=500)