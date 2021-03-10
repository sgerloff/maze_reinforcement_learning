from src.obstacles.obstacle_template import ObstacleTemplate
from src.obstacles.hoshen_kopelman import HoshenKopelmanCluster

import matplotlib.pyplot as plt
import numpy as np
import random


class PrimMaze(ObstacleTemplate):
    def __init__(self, xdim=7, ydim=7, start_state=None, goal_state=None):
        super().__init__(xdim, ydim, goal_state, start_state)
        self.obstacles = np.ones(self.shape)
        self.maze = np.zeros(self.shape)
        self.geometry_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        self.walls = []
        self.hk_cluster = HoshenKopelmanCluster()
        self.cluster_map = None

    def get_obstacles(self):
        self.obstacles = np.ones(self.shape)
        self.maze = np.zeros(self.shape)
        self.walls = []

        self.__add_cell(self.goal_state)
        self.__add_cell(self.start_state)

        print("Generating maze...")
        while self.walls:
            random.shuffle(self.walls)
            random_wall = self.walls[-1]  # Take random item
            self.walls = self.walls[:-1]  # Remove last item
            self.__try_wall(random_wall)

        is_disconnect = True
        while is_disconnect:
            is_disconnect = self.__connect_disconnected_mazes()

        return self.obstacles

    def __add_cell(self, cell):
        i, j = cell
        self.maze[i, j] = 1
        #Find valid neighbor cells and flag potential passages:
        for n in self.geometry_directions:
            passage = (i + n[0], j + n[1])
            target = (i + 2 * n[0], j + 2 * n[1])
            if self.__pos_in_range(target) and self.maze[target[0], target[1]] == 0:
                self.__add_wall(cell, target, passage)

    def __add_wall(self, cell, target, passage):
        self.walls.append(
            {
                "cell": cell,
                "target": target,
                "passage": passage
            }
        )

    def __pos_in_range(self, cell):
        return cell[0] in range(self.shape[0]) and cell[1] in range(self.shape[1])

    def __try_wall(self, wall):
        i, j = wall["target"]
        k, l = wall["passage"]
        if self.maze[i, j] == 0: # Neighbor cell is not part of the maze yet
            self.maze[k, l] = 1  # Add Passage
            self.__add_cell(wall["target"])

    def __connect_disconnected_mazes(self):
        self.obstacles = np.ones(self.shape) - self.maze
        #Get connected mazes via Hoshen Kopelman algorithm
        self.cluster_map = self.hk_cluster.cluster_map(self.obstacles)

        self.cluster_ids = set(self.cluster_map.flatten())
        self.cluster_ids = [c for c in self.cluster_ids if c != -1]

        print(f"Identified {len(self.cluster_ids)} disconnected mazes. Connecting two of them ...")

        if len(self.cluster_ids) > 1:
            valid_passages = []
            for i in range(self.cluster_map.shape[0]):
                for j in range(self.cluster_map.shape[1]):
                    valid_passages.extend(self.__find_valid_passages((i, j)))
            #Choose one random passage
            i, j = random.choice(valid_passages)
            self.maze[i, j] = 1
            self.obstacles[i, j] = 0

        if len(self.cluster_ids) - 1 > 1:
            return True
        else:
            return False

    def __find_valid_passages(self, cell):
        i, j = cell
        tmp_passages = []
        if self.cluster_map[i, j] == self.cluster_ids[0]:
            for direction in self.geometry_directions:
                tmp_passages.extend(self.__check_direction(cell, direction))
        return tmp_passages

    def __check_direction(self, cell, direction):
        tmp_passages = []
        i, j = cell
        tmp = (i + 2 * direction[0], j + 2 * direction[1])
        if self.__pos_in_range(tmp):
            if self.cluster_map[tmp[0], tmp[1]] in self.cluster_ids[1:]:
                tmp_passages.append((i + direction[0], j + direction[1]))
        return tmp_passages

    def plot_obstacles(self):
        plt.imshow(np.ones(self.shape)-self.obstacles, cmap="inferno")
        plt.show()


if __name__ == "__main__":
    maze = PrimMaze(shape=(31, 51))
    maze.get_obstacles()
    maze.plot_obstacles()
