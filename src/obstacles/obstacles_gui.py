import tkinter as tk
from tkinter import ttk
import numpy as np
from src.obstacles.obstacle_template import ObstacleTemplate

import traceback

class ObstaclesGUI(ObstacleTemplate):
    def __init__(self, xdim=7, ydim=7, goal_state=None, start_state=None):
        super().__init__(xdim, ydim, goal_state, start_state)

        self.check_boxes = []
        self.states = ""

        self.root = tk.Tk()
        self.root.title("Enter Obstacles")

        self.setup_boxes()

        btn2 = ttk.Button(self.root, text="Done", command=self.get_states)
        btn2.grid(row=self.shape[0] + 1, column=self.shape[1] + 1)


    def setup_boxes(self):
        self.check_boxes = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                row.append(tk.IntVar())
            self.check_boxes.append(row)

        for i in range(len(self.check_boxes)):
            for j in range(len(self.check_boxes[i])):
                if (i,j) != self.goal_state and (i,j) != self.start_state:
                    checkb = tk.Checkbutton(self.root, text=None, variable=self.check_boxes[i][j], height=2, width=1)
                    checkb.grid(row=i, column=j)

    def get_obstacles(self):
        self.root.mainloop()
        return self.states

    def get_states(self):
        self.states = np.zeros(self.shape)
        for i in range(self.states.shape[0]):
            for j in range(self.states.shape[0]):
                self.states[i, j] = self.check_boxes[i][j].get()
        self.root.destroy()

if __name__ == "__main__":
    obsgui = ObstaclesGUI(shape=(7,11))
    print(obsgui.get_obstacles())