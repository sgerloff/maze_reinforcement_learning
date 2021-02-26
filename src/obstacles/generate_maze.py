from src.obstacles.obstacle_template import ObstacleTemplate
import numpy as np
import matplotlib.pyplot as plt
import random

class GenerateMaze(ObstacleTemplate):
    def __init__(self, shape=(7,7), goal_state=None, start_state=None):
        super().__init__(shape,goal_state,start_state)
        pass

    def generate_maze(self):
        self.obstacles = np.ones(self.shape)
        self.maze = np.zeros(self.shape)
        self.walls = []

        self.__add_cell_to_maze(self.goal_state)
        self.__add_cell_to_maze(self.start_state)

        while self.walls:
            random.shuffle(self.walls)
            # print(self.walls)
            random_wall_cell = self.walls[-1] #Take random item
            self.walls = self.walls[:-1] #Remove last item
            self.__check_new_wall(random_wall_cell)


        plt.imshow(self.maze)
        plt.show()
        print(self.obstacles)

    def __add_cell_to_maze(self, cell):
        i, j = cell
        self.obstacles[i, j] = 0
        self.maze[i,j] = 1

        for n in [(1,0), (-1,0), (0,1), (0,-1)]:
            k, l = i+n[0], j+n[1]
            if self.__pos_in_range(k,l):
                if self.maze[k,l] == 0:
                    self.__add_wall( cell, (k,l) )

    def __add_wall(self, cell, wall_cell):
        self.walls.append(
            {
                "wall": (wall_cell[0], wall_cell[1]),
                "direction": (wall_cell[0]-cell[0], wall_cell[1]-cell[1])
            }
        )

    def __check_new_wall(self, wall):
        wall_cell = wall["wall"]
        direction = wall["direction"]
        target_cell = (wall_cell[0]+direction[0], wall_cell[1] + direction[1])
        if self.__is_valid_target_cell(target_cell, direction):
            self.__add_passage(wall)
            self.__add_cell_to_maze(target_cell)

    def __is_valid_target_cell(self, cell, direction):
        orth1 = (cell[0]+direction[1], cell[1]+direction[0])
        orth2 = (cell[0]-direction[1], cell[1]-direction[0])
        valid = True
        for c in [cell, orth1, orth2]:
            i, j = c
            if self.__pos_in_range(i, j):
                if self.maze[i, j] == 1:
                    valid = False
        if not self.__pos_in_range(cell[0], cell[1]):
            valid = False

        return valid

    def __add_passage(self, wall):
        i,j = wall["wall"]
        self.maze[i,j] = 1
        k, l = wall["direction"]
        for n in [ (l,k), (-l, -k) ]: #Orthogonal Cells
            tmp_i, tmp_j = i+n[0], j+n[1]
            if self.__pos_in_range(tmp_i, tmp_j):
                if self.maze[tmp_i, tmp_j] == 0:
                    self.__add_wall(wall["wall"], (tmp_i, tmp_j))
                    self.maze[i,j] = 1
                else:
                    print("This should not happen!")

    def __pos_in_range(self, i,j):
        return i in range(self.shape[0]) and j in range(self.shape[1])

if __name__ == "__main__":
    maze = GenerateMaze(shape=(30,30))
    maze.generate_maze()