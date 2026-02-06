import numpy as np
from utils.config import *

class HolonomicModel:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def distance(self, node1, node2):
        return np.linalg.norm(node1.pos - node2.pos)
    
    def steer(self, nodeFrom, nodeTo, delta=DELTA):
        dist = self.distance(nodeFrom, nodeTo)

        if dist <= delta or dist < 1e-9:
            return nodeTo.pos
        
        diff = nodeTo.pos - nodeFrom.pos
        newPos = nodeFrom.pos + (diff / dist) * delta
        return newPos
    
    def checkCollision(self, node1, node2):
        p1 = node1.pos
        p2 = node2.pos
        for obs in self.obstacles:
            if obs.intersectSegment(p1, p2):
                return True
            
        return False