import numpy as np
import pygame, sys, random
from utils.Config import *
from utils.node import Node
from utils.model import HolonomicModel
from utils.RRTx import RRTx
import utils.Config

utils.Config.SCREEN_WIDTH = 800
utils.Config.SCREEN_HEIGHT = 600
utils.Config.X_DIM = 800
utils.Config.Y_DIM = 600
utils.Config.GAMMA = 5000.0 

COLOR_BG = (50, 50, 50)
COLOR_OBS_FILL = (0, 0, 0)
COLOR_OBS_BORDER = (255, 255, 255)
COLOR_TREE = (100, 100, 100)
COLOR_PATH = (255, 0, 0)
COLOR_ROBOT = (0, 255, 255)  
COLOR_GOAL = (255, 255, 255)
COLOR_START = (0, 255, 0)

class Obstacle:
    def __init__(self, x, y, w, h, vx=0.0, vy=0.0):
        self.rect = pygame.Rect(x, y, w, h)
        self.x = float(x); self.y = float(y)
        self.vx = vx; self.vy = vy
        self.color = COLOR_OBS_FILL
        self.border_color = COLOR_OBS_BORDER
        self.visible = True

    def move(self, boundary_w, boundary_h):
        if self.vx == 0 and self.vy == 0: return False
        
        self.x += self.vx
        self.y += self.vy
        
        if self.x < 0 or self.x + self.rect.w > boundary_w: self.vx *= -1
        if self.y < 0 or self.y + self.rect.h > boundary_h: self.vy *= -1
        
        self.x = max(0, min(self.x, boundary_w - self.rect.w))
        self.y = max(0, min(self.y, boundary_h - self.rect.h))
        
        old_rect = self.rect.copy()
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)
        
        return old_rect != self.rect

    def draw(self, surface):
        if self.visible:
            pygame.draw.rect(surface, self.color, self.rect)
            pygame.draw.rect(surface, self.border_color, self.rect, 2)

    def intersectSegment(self, p1, p2):
        if not self.visible: return False
        start = (p1[0], p1[1])
        end = (p2[0], p2[1])
        return bool(self.rect.clipline(start, end))

    def isInside(self, x, y):
        if not self.visible: return False
        return self.rect.collidepoint(x, y)

    def __eq__(self, other):
        return isinstance(other, Obstacle) and self.rect == other.rect
    
    def __hash__(self):
        return hash((self.rect.x, self.rect.y, self.rect.w, self.rect.h, self.visible))

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []
        self.start = (400, 550)
        self.goal = (400, 250)
        self.dynamic_event_timer = 0
        self._init_dynamic_layout()

    def _init_dynamic_layout(self):
        self.obstacles.append(Obstacle(200, 150, 50, 300))
        self.obstacles.append(Obstacle(550, 150, 50, 300))
        self.obstacles.append(Obstacle(200, 150, 400, 50))
        self.obstacles.append(Obstacle(200, 400, 150, 50))
        self.obstacles.append(Obstacle(450, 400, 150, 50))
        self.obstacles.append(Obstacle(50, 100, 60, 60, vx=2.0, vy=1.5))
        self.obstacles.append(Obstacle(650, 100, 60, 60, vx=-2.5, vy=1.0))
        self.obstacles.append(Obstacle(100, 500, 80, 40, vx=3.0, vy=0))
        self.obstacles.append(Obstacle(620, 500, 80, 40, vx=-3.0, vy=0))
        self.obstacles.append(Obstacle(380, 460, 40, 40, vx=0, vy=1.0))

    def update(self):
        for obs in self.obstacles:
            obs.move(self.width, self.height)
        
        self.dynamic_event_timer += 1
        if self.dynamic_event_timer > 120:
            self.dynamic_event_timer = 0
            if len(self.obstacles) > 5:
                idx = random.randint(5, len(self.obstacles) - 1)
                self.obstacles[idx].visible = not self.obstacles[idx].visible

    def draw(self, surface):
        pygame.draw.circle(surface, COLOR_START, self.start, 8)
        pygame.draw.rect(surface, COLOR_GOAL, (self.goal[0]-10, self.goal[1]-10, 20, 20))
        
        for obs in self.obstacles:
            obs.draw(surface)

def draw_rrt_graph(surface, rrt):
    if rrt.V:
        for node in rrt.V:
            if node.parent:
                start_pos = (int(node.pos[0]), int(node.pos[1]))
                end_pos = (int(node.parent.pos[0]), int(node.parent.pos[1]))
                pygame.draw.line(surface, COLOR_TREE, start_pos, end_pos, 5)

    curr = rrt.v_bot
    path_points = []
    while curr is not None:
        path_points.append((int(curr.pos[0]), int(curr.pos[1])))
        if curr == rrt.v_goal:
            break
        curr = curr.parent
    
    if len(path_points) > 1:
        pygame.draw.lines(surface, COLOR_PATH, False, path_points, 3)

    robot_pos = (int(rrt.v_bot.pos[0]), int(rrt.v_bot.pos[1]))
    pygame.draw.circle(surface, COLOR_ROBOT, robot_pos, 8)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RRTX Dynamic Replanning Demo")
    clock = pygame.time.Clock()

    env = Environment(SCREEN_WIDTH, SCREEN_HEIGHT)
    start_node = Node(env.start[0], env.start[1])
    goal_node = Node(env.goal[0], env.goal[1])

    model = HolonomicModel(env.obstacles)
    rrt = RRTx(start_node, goal_node, model)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()
                    return

        env.update()

        steps_per_frame = 30
        for _ in range(steps_per_frame):
            rrt.step()

        screen.fill(COLOR_BG)
        
        draw_rrt_graph(screen, rrt)
        env.draw(screen)
        
        font = pygame.font.SysFont("Arial", 18)
        info = f"Nodes: {len(rrt.V)} | Robot Cost: {rrt.v_bot.g:.1f}"
        text_surf = font.render(info, True, (255, 255, 0))
        screen.blit(text_surf, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()