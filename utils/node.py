import numpy as np

class Node:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])

        self.g = float('inf')
        self.lmc = float('inf')

        self.parent = None
        self.children = []

        self.N0_in = []
        self.N0_out = []
        self.Nr_in = []
        self.Nr_out = []

    def __lt__(self, other):
        key1_self = min(self.g, self.lmc)
        key1_other = min(other.g, other.lmc)
        
        if key1_self != key1_other:
            return key1_self < key1_other
        
        if self.g != other.g:
            return self.g < other.g
        
        return id(self) < id(other)

    def __repr__(self):
        return f"Node({self.pos[0]:.2f}, {self.pos[1]:.2f} | g={self.g:.2f}, lmc={self.lmc:.2f})"