import pygame, sys
import numpy as np
import rrtx_cpp

Config = rrtx_cpp.config
Node = rrtx_cpp.Node
Rectangle = rrtx_cpp.Rectangle
HolonomicModel = rrtx_cpp.HolonomicModel
RRTx = rrtx_cpp.RRTx

COLOR_BG = (20, 20, 30)
COLOR_OBSTACLE = (50, 50, 60)
COLOR_OBSTACLE_BORDER = (200, 200, 200)
COLOR_TREE = (0, 100, 255, 30)
COLOR_PATH = (255, 50, 50)
COLOR_ROBOT = (255, 165, 0)
COLOR_GOAL = (0, 255, 127)
COLOR_START = (0, 191, 255)
COLOR_ORPHAN = (148, 0, 211)
COLOR_TEXT = (220, 220, 220)

STATE_SETUP_OBSTACLES = 0
STATE_SET_START = 1
STATE_SET_GOAL = 2
STATE_RUNNING = 3

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption(f"RRTX C++ Backend Interactive Planner")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.large_font = pygame.font.SysFont("Consolas", 20, bold=True)

        self.current_state = STATE_SETUP_OBSTACLES
        self.obstacles = []
        self.start_node = None
        self.goal_node = None
        
        self.model = None
        self.rrtx = None
        
        self.create_borders()
        self.last_move_time = 0
        self.move_delay = 50
        self.show_tree = True
        self.paused = False
        self.start_time = 0

    def create_borders(self):
        w, h = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        thickness = 40
        self.obstacles.append(Rectangle(w/2, thickness/2, w, thickness, 0))
        self.obstacles.append(Rectangle(w/2, h - thickness/2, w, thickness, 0))
        self.obstacles.append(Rectangle(thickness/2, h/2, thickness, h, 0))
        self.obstacles.append(Rectangle(w - thickness/2, h/2, thickness, h, 0))

    def init_algorithm(self):
        print(">>> Initializing RRTx Algorithm...")
        self.model = HolonomicModel(self.obstacles)
        self.rrtx = RRTx(self.start_node, self.goal_node, self.model)
        self.start_time = pygame.time.get_ticks()

    def is_point_inside_polygon(self, point, vertices):
        x, y = point
        n = len(vertices)
        inside = False
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def handle_obstacle_click(self, pos, is_left_click):
        mx, my = pos
        if is_left_click:
            new_obs = Rectangle(mx, my, 60, 60, 0)
            self.obstacles.append(new_obs)
            if self.current_state == STATE_RUNNING and self.rrtx:
                r = self.rrtx.shrinking_ball_radius()
                self.rrtx.update_obstacles(r, self.obstacles)
        else:
            for i in range(len(self.obstacles) - 1, -1, -1):
                obs = self.obstacles[i]
                vertices = obs.get_vertices() 
                if self.is_point_inside_polygon((mx, my), vertices):
                    self.obstacles.pop(i)
                    if self.current_state == STATE_RUNNING and self.rrtx:
                        r = self.rrtx.shrinking_ball_radius()
                        self.rrtx.update_obstacles(r, self.obstacles)
                    break

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    if event.key == pygame.K_RETURN:
                        if self.current_state == STATE_SETUP_OBSTACLES:
                            self.current_state = STATE_SET_START
                            print("Transition: Set Start Node")
                        elif self.current_state == STATE_RUNNING:
                            pass

                    if self.current_state == STATE_RUNNING:
                        if event.key == pygame.K_t:
                            self.show_tree = not self.show_tree
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    
                    if self.current_state == STATE_SETUP_OBSTACLES:
                        self.handle_obstacle_click((mx, my), event.button == 1)
                    elif self.current_state == STATE_SET_START:
                        if event.button == 1:
                            self.start_node = Node(float(mx), float(my))
                            self.current_state = STATE_SET_GOAL
                            print(f"Start set at: {mx}, {my}")
                    elif self.current_state == STATE_SET_GOAL:
                        if event.button == 1:
                            self.goal_node = Node(float(mx), float(my))
                            self.current_state = STATE_RUNNING
                            print(f"Goal set at: {mx}, {my}")
                            self.init_algorithm()
                    elif self.current_state == STATE_RUNNING:
                        self.handle_obstacle_click((mx, my), event.button == 1)

            if self.current_state == STATE_RUNNING and not self.paused and self.rrtx:
                current_time = pygame.time.get_ticks()
                should_move = False
                if current_time - self.last_move_time > self.move_delay:
                    should_move = True
                    self.last_move_time = current_time
                
                self.rrtx.step(move_robot=should_move)

            self.screen.fill(COLOR_BG)
            for obs in self.obstacles:
                vertices = obs.get_vertices()
                pygame.draw.polygon(self.screen, COLOR_OBSTACLE, vertices)
                pygame.draw.lines(self.screen, COLOR_OBSTACLE_BORDER, True, vertices, 2)

            if self.start_node:
                pygame.draw.circle(self.screen, COLOR_START, (int(self.start_node.pos[0]), int(self.start_node.pos[1])), 8)
            
            if self.goal_node:
                pygame.draw.circle(self.screen, COLOR_GOAL, (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])), 8)

            if self.current_state == STATE_RUNNING and self.rrtx:
                if self.show_tree:
                    for node in self.rrtx.V:
                        if node.parent:
                            s = (int(node.pos[0]), int(node.pos[1]))
                            e = (int(node.parent.pos[0]), int(node.parent.pos[1]))
                            pygame.draw.line(self.screen, COLOR_TREE, s, e, 1)
                            
                bot_pos = (int(self.rrtx.v_bot.pos[0]), int(self.rrtx.v_bot.pos[1]))
                pygame.draw.circle(self.screen, COLOR_ROBOT, bot_pos, 8)
                
                path = []
                curr = self.rrtx.v_bot
                while curr and curr.parent:
                    path.append((curr.pos[0], curr.pos[1]))
                    curr = curr.parent
                    if curr == self.goal_node: break
                path.append((curr.pos[0], curr.pos[1]))
                
                if len(path) > 1:
                    pygame.draw.lines(self.screen, COLOR_PATH, False, path, 3)

            self.draw_ui_overlay()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

    def draw_ui_overlay(self):
        status_text = ""
        instruct_text = ""
        
        if self.current_state == STATE_SETUP_OBSTACLES:
            status_text = "MODE: MAP EDITING"
            instruct_text = "[L-Click]: Add Obstacle | [R-Click]: Remove | [ENTER]: Done"
        elif self.current_state == STATE_SET_START:
            status_text = "MODE: SET START"
            instruct_text = "[L-Click]: Place Start Position"
        elif self.current_state == STATE_SET_GOAL:
            status_text = "MODE: SET GOAL"
            instruct_text = "[L-Click]: Place Goal Position -> RUN"
        elif self.current_state == STATE_RUNNING:
            status_text = "MODE: RRTx RUNNING"
            cost = self.rrtx.v_bot.lmc if self.rrtx else 0
            cost_str = f"{cost:.2f}" if cost < float('inf') else "Inf"
            instruct_text = f"Cost: {cost_str} | [L-Click]: Add | [R-Click]: Remove | [Space]: Pause"

        pygame.draw.rect(self.screen, (0,0,0), (0, 0, Config.SCREEN_WIDTH, 60))
        
        surf_status = self.large_font.render(status_text, True, COLOR_ORPHAN)
        surf_instruct = self.font.render(instruct_text, True, COLOR_TEXT)
        
        self.screen.blit(surf_status, (10, 5))
        self.screen.blit(surf_instruct, (10, 35))

if __name__ == "__main__":
    viz = Visualizer()
    viz.run()