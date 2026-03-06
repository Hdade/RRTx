import numpy as np
import pygame, sys, cv2, rrtx_cpp, multiprocessing, queue, argparse
from gan_worker import gan_worker_loop
from sfd_worker import sfd_worker_loop

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
COLOR_GOAL = (0, 0, 255)
COLOR_START = (255, 0, 0)
COLOR_ORPHAN = (148, 0, 211)
COLOR_TEXT = (220, 220, 220)
COLOR_HEURISTIC_TREE = (0, 255, 0)

STATE_SETUP_OBSTACLES = 0
STATE_SET_START = 1
STATE_SET_GOAL = 2
STATE_TEST_MODEL = 3
STATE_RUNNING = 4

class Visualizer:
    def __init__(self, model_type="gan"):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        self.model_type = model_type.upper()
        pygame.display.set_caption(f"RRTX C++ Backend Interactive Planner - {self.model_type} Model")
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

        self.model_input_queue = multiprocessing.Queue()
        self.model_output_queue = multiprocessing.Queue()
        
        if self.model_type == "GAN":
            print(">>> [Main] Starting GAN Worker Process...")
            checkpoint_path = "checkpoints/GAN_checkpoint/netG_epoch_40.pth"
            target_func = gan_worker_loop
            target_args = (self.model_input_queue, self.model_output_queue, checkpoint_path, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        else:
            print(">>> [Main] Starting SFD Worker Process...")
            checkpoint_path = "checkpoints/SFD_checkpoints"
            target_func = sfd_worker_loop
            target_args = (self.model_input_queue, self.model_output_queue, checkpoint_path, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))

        self.worker_process = multiprocessing.Process(
            target=target_func,
            args=target_args
        )
        self.worker_process.daemon = True
        self.worker_process.start()
        
        self.pending_model_request = False
        self.sampling_map_updated = False
        self.heuristic_debug_surface = None

    def create_heatmap_surface_from_data(self, flat_map):
        w, h = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        
        heatmap_2d = flat_map.reshape((h, w))
        heatmap_color = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_uint8 = (heatmap_2d * 255).astype(np.uint8)
        heatmap_color[:, :, 1] = heatmap_uint8
        
        heatmap_surf = pygame.surfarray.make_surface(heatmap_color.transpose(1, 0, 2))
        heatmap_surf.set_alpha(150)
        return heatmap_surf

    def check_model_result(self):
        try:
            flat_map = self.model_output_queue.get_nowait()
            self.heuristic_debug_surface = self.create_heatmap_surface_from_data(flat_map)
            if self.rrtx:
                self.rrtx.update_sampling_distribution(flat_map, Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
                self.rrtx.update_node_heuristics()
                
            print(f">>> [Main] Received Heatmap from {self.model_type} Worker!")
            self.pending_model_request = False
            self.sampling_map_updated = True
        except queue.Empty:
            pass

    def get_model_input_from_pygame(self):
        w, h = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        map_surf = pygame.Surface((w, h))
        map_surf.fill((255, 255, 255)) 
        
        for obs in self.obstacles:
            vertices = obs.get_vertices()
            if len(vertices) > 2:
                pygame.draw.polygon(map_surf, (0, 0, 0), vertices)
        
        map_arr = pygame.surfarray.array3d(map_surf)
        map_arr = map_arr.transpose(1, 0, 2)
        
        points_surf = pygame.Surface((w, h))
        points_surf.fill((255, 255, 255)) 
        
        start_pos = None
        if self.rrtx and self.rrtx.v_bot:
            start_pos = (int(self.rrtx.v_bot.pos[0]), int(self.rrtx.v_bot.pos[1]))
        elif self.start_node:
            start_pos = (int(self.start_node.pos[0]), int(self.start_node.pos[1]))

        if start_pos:
            pygame.draw.circle(points_surf, COLOR_START, start_pos, 10)

        if self.goal_node:
            goal_pos = (int(self.goal_node.pos[0]), int(self.goal_node.pos[1]))
            pygame.draw.circle(points_surf, COLOR_GOAL, goal_pos, 10)
        
        points_arr = pygame.surfarray.array3d(points_surf)
        points_arr = points_arr.transpose(1, 0, 2)
        return map_arr, points_arr
    
    def update_model_heuristic(self):
        if self.pending_model_request:
            return

        map_img, points_img = self.get_model_input_from_pygame()
        self.model_input_queue.put((map_img, points_img))
        self.pending_model_request = True
        print(f">>> [Main] Sent request to {self.model_type} Worker...")

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
        self.update_model_heuristic()
        self.start_time = pygame.time.get_ticks()

    def is_point_inside_polygon(self, point, vertices):
        x, y = point
        n = len(vertices)
        inside = False
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
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
                self.sampling_map_updated = False 
        else:
            for i in range(len(self.obstacles) - 1, -1, -1):
                obs = self.obstacles[i]
                vertices = obs.get_vertices() 
                if self.is_point_inside_polygon((mx, my), vertices):
                    self.obstacles.pop(i)
                    if self.current_state == STATE_RUNNING and self.rrtx:
                        r = self.rrtx.shrinking_ball_radius()
                        self.rrtx.update_obstacles(r, self.obstacles)
                        self.sampling_map_updated = False
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
                        elif self.current_state == STATE_TEST_MODEL:
                            self.current_state = STATE_RUNNING
                            self.init_algorithm()
                            print("Transition: RRTx Running")

                    if self.current_state == STATE_TEST_MODEL:
                        if event.key == pygame.K_SPACE:
                            print(">>> Sending request to Visualize...")
                            self.update_model_heuristic()

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
                            self.current_state = STATE_TEST_MODEL
                            print(f"Goal set at: {mx}, {my}. Now in {self.model_type} TEST MODE. Press ENTER to Run RRTx.")
                            self.update_model_heuristic()

                    elif self.current_state == STATE_RUNNING:
                        self.handle_obstacle_click((mx, my), event.button == 1)
                    
                    elif self.current_state == STATE_TEST_MODEL:
                        self.handle_obstacle_click((mx, my), event.button == 1)
                        self.update_model_heuristic()

            self.check_model_result()
            if self.current_state == STATE_RUNNING and self.rrtx:
                if not self.sampling_map_updated:
                    self.update_model_heuristic()

                if not self.paused:
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

            if self.current_state == STATE_TEST_MODEL and self.heuristic_debug_surface:
                self.screen.blit(self.heuristic_debug_surface, (0, 0))

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
                            if node.heuristic_val > 0.4:
                                pygame.draw.line(self.screen, COLOR_HEURISTIC_TREE, s, e, 2)
                            else:
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

        print(">>> Stopping Worker Process...")
        self.model_input_queue.put('STOP')
        self.worker_process.join()
        pygame.quit()
        sys.exit()

    def draw_ui_overlay(self):
        status_text = ""
        instruct_text = ""
        
        if self.current_state == STATE_SETUP_OBSTACLES:
            status_text = "MODE: MAP EDITING"
            instruct_text = "[L-Click]: Add Obs | [ENTER]: Done"
        
        elif self.current_state == STATE_SET_START:
            status_text = "MODE: SET START"
            instruct_text = "[L-Click]: Place Start"
        
        elif self.current_state == STATE_SET_GOAL:
            status_text = "MODE: SET GOAL"
            instruct_text = "[L-Click]: Place Goal"
        
        elif self.current_state == STATE_TEST_MODEL:
            status_text = f"MODE: {self.model_type} VISUALIZATION CHECK"
            instruct_text = "[L-Click]: Move Obs | [SPACE]: Refresh | [ENTER]: Run RRTx"

        elif self.current_state == STATE_RUNNING:
            status_text = "MODE: RRTx RUNNING"
            cost = self.rrtx.v_bot.lmc if self.rrtx else 0
            cost_str = f"{cost:.2f}" if cost < float('inf') else "Inf"
            instruct_text = f"Cost: {cost_str} | [L-Click]: Add Obs | [SPACE]: Pause"

        pygame.draw.rect(self.screen, (0,0,0), (0, 0, Config.SCREEN_WIDTH, 60))
        
        surf_status = self.large_font.render(status_text, True, COLOR_ORPHAN)
        surf_instruct = self.font.render(instruct_text, True, COLOR_TEXT)
        
        self.screen.blit(surf_status, (10, 5))
        self.screen.blit(surf_instruct, (10, 35))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RRTx Path Planning Visualization with GAN/SFD Heuristic")
    parser.add_argument("--model", type=str, choices=["gan", "sfd"], default="gan", 
                        help="Choose the model to use for path heuristic: 'gan' or 'sfd'. Default is 'gan'.")
    args = parser.parse_args()

    viz = Visualizer(model_type=args.model)
    viz.run()