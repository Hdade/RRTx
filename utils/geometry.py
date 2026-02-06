import math
import numpy as np

class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def isInside(self, x, y):
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.r ** 2
    
    def intersectSegment(self, point1, point2):
        d = point2 - point1
        f = point1 - np.array([self.x, self.y])

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - self.r ** 2

        delta = b ** 2 - 4 * a * c

        if delta < 0:
            # line not cut circle
            return False
        
        delta = math.sqrt(delta)
        t1 = (-b - delta) / (2 * a)
        t2 = (-b + delta) / (2 * a)

        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            # intersection on line
            return True
        
        if (0 <= t1 and t2 <= 1) or (0 <= t2 and t1 <= 1):
            # line in circle
            return True
        
        return False
    
class Rectangle:
    def __init__(self, x_center, y_center, width, height, angle_deg):
        self.x = x_center
        self.y = y_center
        self.width = width
        self.height = height
        self.angle = np.deg2rad(angle_deg)
    
    def get_vertices(self):
        w2 = self.width / 2
        h2 = self.height / 2
        
        vertices = np.array([
            [-w2, -h2],
            [w2, -h2],
            [w2, h2],
            [-w2, h2]
        ])
        
        rot = np.array([
            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ])
        
        vertices = vertices @ rot.T + np.array([self.x, self.y])
        return vertices
    
    def intersectSegment(self, point1, point2):
        bounding_r = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        d = point2 - point1
        f = point1 - np.array([self.x, self.y])
        
        d_len_sq = np.dot(d, d)
        if d_len_sq == 0:
            return np.linalg.norm(f) <= bounding_r

        t = -np.dot(f, d) / d_len_sq
        t = max(0, min(1, t))
        
        closest_point = point1 + t * d
        distance_sq = np.sum((closest_point - np.array([self.x, self.y]))**2)
        
        if distance_sq > bounding_r ** 2:
            return False
            
        vertices = self.get_vertices()
        edges = [
            vertices[1] - vertices[0],
            vertices[2] - vertices[1],
            vertices[3] - vertices[2],
            vertices[0] - vertices[3]
        ]
        
        segment = point2 - point1
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            
            proj_vertices = vertices @ normal
            proj_p1 = point1 @ normal
            proj_p2 = point2 @ normal
            
            min_rect = np.min(proj_vertices)
            max_rect = np.max(proj_vertices)
            min_seg = min(proj_p1, proj_p2)
            max_seg = max(proj_p1, proj_p2)
            
            if max_seg < min_rect or min_seg > max_rect:
                return False
        
        return True
    
    def isInside(self, x, y):
        point = np.array([x, y])
        vertices = self.get_vertices()
        local_point = point - np.array([self.x, self.y])
        rot_inv = np.array([
            [np.cos(-self.angle), -np.sin(-self.angle)],
            [np.sin(-self.angle), np.cos(-self.angle)]
        ])
        
        local_point = local_point @ rot_inv.T
        return abs(local_point[0]) <= self.width/2 and abs(local_point[1]) <= self.height/2
    
    def get_bounding_circle(self):
        r = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        return Circle(self.x, self.y, r)