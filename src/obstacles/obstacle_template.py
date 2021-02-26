
class ObstacleTemplate:
    def __init__(self, shape=(7,7), goal_state=None, start_state=None):
        self.shape = shape
        self.set_start_state(start_state)
        self.set_goal_state(goal_state)

    def set_goal_state(self, goal_state):
        if goal_state is None:
            self.goal_state = (self.shape[0]-1, self.shape[1]-1)
        else:
            self.goal_state = goal_state

    def set_start_state(self, start_state):
        if start_state is None:
            self.start_state = (self.shape[0]-1, 0)
        else:
            self.start_state = start_state

    def get_obstacles(self):
        pass