import numpy as np
from numba import jit, njit, typeof
from numba.typed import List
from operator import itemgetter

class jitknn:
    def __init__(self):
        self.accuracy = 0

    @staticmethod
    @njit(nogil=True, cache=True)
    def find_distances(train, testGroup):
        distances = []
        for i in range(0, len(train)):
            difference = np.sqrt(np.sum(np.power(train[i]-testGroup, 2)))
            distances.append((train[i], difference))
        return distances

    @staticmethod
    @njit(nogil=True, cache=True)
    def find_neighbors(k, distances):
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    @staticmethod
    @njit(nogil=True, cache=True)
    def get_votes(neighbors):
        votes = {}
        for neighbor in neighbors:
            flag = neighbor[-1]
            if flag in votes:
                votes[flag] += 1
            else:
                votes[flag] = 1
        return votes

    @staticmethod
    @njit(nogil=True,cache=True)
    def get_accuracy(predicted, total_sz):
        accuracy = (predicted/total_sz)
        return accuracy
      
    def run_model(self, train, test, k):
        print("Running Model")
        avg_accuracy = []
        for group in test:
            distances = self.find_distances(train, group)
            distances.sort(key=itemgetter(1))
            neighbors = self.find_neighbors(k, distances)
            votes = self.get_votes(neighbors)
            accuracy = self.get_accuracy(votes[group[-1]], len(neighbors))
            avg_accuracy.append(accuracy)
        self.accuracy =  np.average(np.array(avg_accuracy))
        print("Successful Run!")
