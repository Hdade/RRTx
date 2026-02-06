import numpy as np
from utils.node import *
from utils.Config import *
from utils.geometry import *

class HolonomicModel:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def distance(self, node1, node2):
        return np.linalg.norm(node1.pos - node2.pos)
    
    def steer(self, nodeFrom, nodeTo, delta=DELTA):
        dist = self.distance(nodeFrom, nodeTo)

        if dist <= delta:
            return nodeTo.pos
        
        diff = nodeTo.pos - nodeFrom.pos
        newPos = nodeFrom.pos + (diff / dist) * delta
        return newPos
    
    def checkCollision(self, node1, node2):
        for obs in self.obstacles:
            if obs.intersectSegment(node1.pos, node2.pos):
                return True
            
        return False