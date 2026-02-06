import math
import numpy as np

class Circle:
    def __init__(self, x, y, r):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)
        self._hash = hash((self.x, self.y, self.r, 'circle'))

    def __eq__(self, other):
        return (
            isinstance(other, Circle) 
            and self.x == other.x
            and self.y == other.y
            and self.r == other.r
        )

    def __hash__(self):
        return self._hash

    def isInside(self, x, y):
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.r ** 2
    
    def intersectSegment(self, point1, point2):
        d = point2 - point1
        f = point1 - np.array([self.x, self.y])

        a = np.dot(d, d)
        if a == 0:
            return self.isInside(point1[0], point1[1])

        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - self.r ** 2

        delta = b ** 2 - 4 * a * c
        if delta < 0:
            return False
        
        sqrt_delta = math.sqrt(delta)
        t1 = (-b - sqrt_delta) / (2 * a)
        t2 = (-b + sqrt_delta) / (2 * a)

        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (0 <= t1 and t2 <= 1) or (0 <= t2 and t1 <= 1)

class Rectangle:
    def __init__(self, x_center, y_center, width, height, angle_deg):
        self.x = float(x_center)
        self.y = float(y_center)
        self.width = float(width)
        self.height = float(height)
        self.angle = np.deg2rad(angle_deg)
        
        w2 = self.width / 2.0
        h2 = self.height / 2.0
        
        local_v = np.array([
            [-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]
        ])
        
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        self.vertices = local_v @ rot.T + np.array([self.x, self.y])
        
        self.bounding_r = np.sqrt(w2**2 + h2**2)
        
        self._hash = hash((self.x, self.y, self.width, self.height, self.angle, 'rect'))

    def __eq__(self, other):
        return (
            isinstance(other, Rectangle) 
            and self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
            and self.angle == other.angle
        )

    def __hash__(self):
        return self._hash

    def get_vertices(self):
        return self.vertices
    
    def isInside(self, x, y):
        dx = x - self.x
        dy = y - self.y
        
        cos_a = np.cos(-self.angle)
        sin_a = np.sin(-self.angle)
        
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        return abs(local_x) <= self.width / 2.0 and abs(local_y) <= self.height / 2.0

    def intersectSegment(self, point1, point2):
        if np.linalg.norm((point1 + point2) / 2 - np.array([self.x, self.y])) > self.bounding_r + np.linalg.norm(point2-point1) / 2:
            return False

        rect_v = self.vertices
        axes = [
            rect_v[1] - rect_v[0],
            rect_v[2] - rect_v[1] 
        ]

        edge_seg = point2 - point1
        axes.append(np.array([-edge_seg[1], edge_seg[0]])) 

        for axis in axes:
            if np.dot(axis, axis) == 0: 
                continue
            
            projs_rect = [np.dot(v, axis) for v in rect_v]
            min_r, max_r = min(projs_rect), max(projs_rect)
            
            p1_proj = np.dot(point1, axis)
            p2_proj = np.dot(point2, axis)
            min_s, max_s = min(p1_proj, p2_proj), max(p1_proj, p2_proj)
            if max_s < min_r or min_s > max_r:
                return False
        
        return True