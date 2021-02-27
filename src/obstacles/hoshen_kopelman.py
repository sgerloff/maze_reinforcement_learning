import numpy as np


class HoshenKopelmanCluster:
    def __init__(self):
        pass

    def cluster_map(self, obstacles):
        array = np.zeros(obstacles.shape)
        array = array - obstacles
        valid_indices = self.get_valid_indices(array)
        array = self.initialize_array(array, valid_indices)
        self.get_cluster_indices(array, valid_indices)

        return array

    def get_valid_indices(self, array):
        valid_indices = []
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j] != -1:
                    valid_indices.append((i, j))
        return valid_indices

    def initialize_array(self, array, valid_indices):
        for i, idx in enumerate(valid_indices):
            array[idx[0], idx[1]] = i + 1
        return array


    def get_cluster_indices(self, array, valid_indices):
        change = True
        while change:
            change = False
            for i in valid_indices:
                neighbors = []
                for c in [(1,0), (-1,0), (0,1), (0,-1)]:
                    new_idx = (i[0]+c[0], i[1]+c[1])
                    if new_idx in valid_indices:
                        try:
                            neighbors.append(array[ new_idx[0], new_idx[1] ])
                        except:
                            pass
                if len(neighbors) > 0:
                    if array[i[0],i[1]] > min(neighbors):
                        array[i[0], i[1]] = min(neighbors)
                        change=True
        return array