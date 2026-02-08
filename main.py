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

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption(f"RRTX C++ Backend - Complex Maze")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)

        self.init_simulation()

        self.last_move_time = 0
        self.move_delay = 50
        
        self.show_tree = True
        self.paused = False
        self.dynamic_triggered = False

    def init_simulation(self):
        self.start_node = Node(400.0, 100.0)
        self.goal_node = Node(400.0, 570.0)

        self.obstacles = []
        self.create_map()
        self.model = HolonomicModel(self.obstacles)
        self.rrtx = RRTx(self.start_node, self.goal_node, self.model)
        self.start_time = pygame.time.get_ticks()

    def create_map(self):
        self.obstacles.append(Rectangle(40, 300, 40, 500, 10)) 
        self.obstacles.append(Rectangle(760, 300, 40, 500, -10))
        self.obstacles.append(Rectangle(400, 40, 700, 40, 0))
        
        self.obstacles.append(Rectangle(150, 560, 250, 40, 10))
        self.obstacles.append(Rectangle(650, 560, 250, 40, -10))

        self.obstacles.append(Rectangle(250, 300, 40, 200, 15))
        self.obstacles.append(Rectangle(260, 200, 40, 100, -10))
        
        self.obstacles.append(Rectangle(550, 300, 40, 200, -15))
        self.obstacles.append(Rectangle(540, 200, 40, 100, 10))
        
        self.obstacles.append(Rectangle(400, 150, 300, 40, 0))
        
        self.obstacles.append(Rectangle(320, 450, 150, 40, 20))
        self.obstacles.append(Rectangle(480, 450, 150, 40, -20))

        self.obstacles.append(Rectangle(400, 250, 60, 60, 45))
        
        self.obstacles.append(Rectangle(200, 100, 60, 150, 30))
        self.obstacles.append(Rectangle(600, 100, 60, 150, -30))

        self.obstacles.append(Rectangle(350, 350, 50, 50, 0))
        self.obstacles.append(Rectangle(450, 350, 50, 50, 0))

    def trigger_dynamic_event(self):
        print(">>> WARNING: DYNAMIC OBSTACLE DETECTED! <<<")
        new_obs = Rectangle(400, 400, 120, 30, 0) 
        self.obstacles.append(new_obs)
        
        r = self.rrtx.shrinking_ball_radius()
        self.rrtx.update_obstacles(r, self.obstacles)
        self.dynamic_triggered = True

    def draw_obstacles(self):
        for obs in self.obstacles:
            vertices = obs.get_vertices()
            pygame.draw.polygon(self.screen, COLOR_OBSTACLE, vertices)
            pygame.draw.lines(self.screen, COLOR_OBSTACLE_BORDER, True, vertices, 2)

    def draw_tree(self):
        if not self.show_tree:
            return

        nodes_to_draw = self.rrtx.V
        for node in nodes_to_draw:
            if node.parent:
                start_pos = (int(node.pos[0]), int(node.pos[1]))
                end_pos = (int(node.parent.pos[0]), int(node.parent.pos[1]))
                pygame.draw.line(self.screen, COLOR_TREE, start_pos, end_pos, 1)

    def draw_orphans(self):
        for node in self.rrtx.V:
            if node.lmc == float('inf'): 
                pos = (int(node.pos[0]), int(node.pos[1]))
                pygame.draw.circle(self.screen, COLOR_ORPHAN, pos, 2)

    def draw_path(self):
        path = []
        curr = self.rrtx.v_bot
        found_goal = False
        limit = 0
        
        while curr is not None and limit < 5000:
            path.append((curr.pos[0], curr.pos[1]))
            if curr == self.goal_node: 
                found_goal = True
                break

            curr = curr.parent
            limit += 1
            
        if found_goal and len(path) > 1:
            pygame.draw.lines(self.screen, COLOR_PATH, False, path, 4)

    def draw_ui(self):
        fps = int(self.clock.get_fps())
        pygame.display.set_caption(f"RRTX C++ Backend - FPS: {fps}")
        nodes_count = len(self.rrtx.V)
        orphan_count = len(self.rrtx.Orphans)
        cost = self.rrtx.v_bot.lmc
        cost_str = f"{cost:.2f}" if cost < float('inf') else "Inf"
        texts = [
            f"FPS: {fps} | Nodes: {nodes_count} | Orphans: {orphan_count}",
            f"Robot Cost (LMC): {cost_str}",
            f"Controls: [Space] Pause | [T] Toggle Tree | [Click] Add Obstacle",
        ]
        
        if self.dynamic_triggered:
            texts.append("STATUS: OBSTACLE UPDATE DETECTED! REWIRING...")

        for i, line in enumerate(texts):
            s = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(s, (10, 10 + i * 20))

    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.show_tree = not self.show_tree
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    new_obs = Rectangle(mx, my, 50, 50, 0)
                    self.obstacles.append(new_obs)
                    r = self.rrtx.shrinking_ball_radius()
                    self.rrtx.update_obstacles(r, self.obstacles)

            if not self.paused:
                current_time = pygame.time.get_ticks()
                should_move = False
                if current_time - self.last_move_time > self.move_delay:
                    should_move = True
                    self.last_move_time = current_time
                
                self.rrtx.step(move_robot=should_move)
                if not self.dynamic_triggered and (current_time - self.start_time > 4000):
                     if self.rrtx.v_bot.lmc < float('inf'):
                          self.trigger_dynamic_event()

            self.screen.fill(COLOR_BG)
            self.draw_obstacles()
            self.draw_tree()
            self.draw_orphans()
            self.draw_path()
            
            pygame.draw.circle(self.screen, COLOR_GOAL, (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])), 8)
            pygame.draw.circle(self.screen, (255, 255, 0), (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])), int(Config.GOAL_RADIUS), 1)
            
            bot_pos = (int(self.rrtx.v_bot.pos[0]), int(self.rrtx.v_bot.pos[1]))
            pygame.draw.circle(self.screen, COLOR_ROBOT, bot_pos, 8)
            pygame.draw.circle(self.screen, (255, 255, 0), bot_pos, int(Config.GOAL_RADIUS), 1)
            
            r_search = self.rrtx.shrinking_ball_radius()
            pygame.draw.circle(self.screen, (100, 100, 100), bot_pos, int(r_search), 1)

            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(120) 

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    viz = Visualizer()
    viz.run()