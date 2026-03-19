import numpy as np
import pygame, sys, cv2, rrtx_cpp, multiprocessing, queue, argparse, json, math
from gan_worker import gan_worker_loop
from sfd_worker import sfd_worker_loop

Config = rrtx_cpp.config
Node = rrtx_cpp.Node
Rectangle = rrtx_cpp.Rectangle
Circle = rrtx_cpp.Circle
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
        
        # Bảo vệ RAM khỏi Garbage Collector
        self._gc_protector = [] 

        self.py_obstacles = [] 
        self.obstacles = []
        
        self.start_node = None
        self.goal_node = None
        self.model = None
        self.rrtx = None
        
        # --- TÁCH BIỆT THỜI GIAN (FIX LỖI ĐỨNG YÊN) ---
        self.obstacle_move_delay = 50  # Chướng ngại vật cập nhật mỗi 50ms
        self.last_obs_move_time = 0
        
        self.robot_move_delay = 100    # Robot bước đi mỗi 100ms (cho C++ kịp tính toán)
        self.last_robot_move_time = 0
        
        self.show_tree = True
        self.paused = False
        self.use_heuristic = True

        self.model_input_queue = multiprocessing.Queue()
        self.model_output_queue = multiprocessing.Queue()
        
        if self.model_type == "GAN":
            print(">>> [Main] Starting GAN Worker Process...")
            checkpoint_path = "checkpoints/GAN_checkpoint/netG_epoch_40.pth"
            target_func = gan_worker_loop
            target_args = (self.model_input_queue, self.model_output_queue, checkpoint_path, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        elif self.model_type == "SFD":
            print(">>> [Main] Starting SFD Worker Process...")
            checkpoint_path = "checkpoints/SFD_checkpoints"
            target_func = sfd_worker_loop
            target_args = (self.model_input_queue, self.model_output_queue, checkpoint_path, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        else:
            self.use_heuristic = False
            self.model_input_queue = None
            self.model_output_queue = None

        if self.use_heuristic:
            self.worker_process = multiprocessing.Process(target=target_func, args=target_args)
            self.worker_process.daemon = True
            self.worker_process.start()
        else:
            self.worker_process = None
            
        self.pending_model_request = False
        self.sampling_map_updated = False
        self.initial_map_loaded = False # Cờ quan trọng để chống giật
        self.heuristic_debug_surface = None
        
        self.load_scenario("annotations1.json")

    def get_rect_vertices(self, x, y, w, h, angle):
        w2, h2 = w / 2.0, h / 2.0
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        local_v = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
        return [(lx * cos_a - ly * sin_a + x, lx * sin_a + ly * cos_a + y) for lx, ly in local_v]

    def create_borders(self):
        w, h = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        thickness = 40
        borders = [
            (w/2, thickness/2, w, thickness),
            (w/2, h - thickness/2, w, thickness),
            (thickness/2, h/2, thickness, h),
            (w - thickness/2, h/2, thickness, h)
        ]
        for bx, by, bw, bh in borders:
            obs_cpp = Rectangle(bx, by, bw, bh, 0)
            self._gc_protector.append(obs_cpp)
            py_obj = {'shape': 'rectangle', 'x': bx, 'y': by, 'w': bw, 'h': bh, 'angle': 0, 'type': 'static', 'cpp_ref': obs_cpp}
            self.py_obstacles.append(py_obj)
            self.obstacles.append(obs_cpp)

    def load_scenario(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.start_node = Node(float(data['start'][0]), float(data['start'][1]))
        self.goal_node = Node(float(data['goal'][0]), float(data['goal'][1]))

        self.py_obstacles = []
        self.obstacles = []
        self.create_borders() 

        for obs_data in data['obstacles']:
            shape = obs_data.get('shape', 'rectangle')
            current_pos = obs_data.get('center') if shape == 'circle' else obs_data.get('position')
            x, y = float(current_pos[0]), float(current_pos[1])
            
            if shape == 'circle':
                r = float(obs_data['radius'])
                py_obj = {'shape': 'circle', 'x': x, 'y': y, 'r': r}
                obs_cpp = Circle(x, y, r)
            else:
                if shape == 'square':
                    w = float(obs_data['size'])
                    h = float(obs_data['size'])
                else:
                    w = float(obs_data['width'])
                    h = float(obs_data.get('height', w))
                angle = float(obs_data.get('rotation', 0))
                py_obj = {'shape': 'rectangle', 'x': x, 'y': y, 'w': w, 'h': h, 'angle': angle}
                obs_cpp = Rectangle(x, y, w, h, angle)

            self._gc_protector.append(obs_cpp)
            py_obj['type'] = obs_data['type']
            if py_obj['type'] == 'dynamic':
                py_obj['velocity'] = list(obs_data['velocity'])
                py_obj['bounds'] = obs_data['movement_bounds']

            if py_obj['type'] == 'proximity':
                py_obj['trigger_dist'] = obs_data.get('trigger_dist', 100.0)
                py_obj['behavior'] = obs_data.get('behavior', 'appear_when_near')
                py_obj['active'] = False if py_obj['behavior'] == 'appear_when_near' else True
                py_obj['triggered'] = False
            
            py_obj['cpp_ref'] = obs_cpp
            self.py_obstacles.append(py_obj)
            if py_obj.get('active', True):
                self.obstacles.append(obs_cpp)

        self.current_state = STATE_RUNNING
        self.init_algorithm()

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
        if not self.use_heuristic: return
        try:
            flat_map = self.model_output_queue.get_nowait()
            safe_flat_map = np.ascontiguousarray(flat_map.flatten(), dtype=np.float64)
            self.heuristic_debug_surface = self.create_heatmap_surface_from_data(flat_map)
            
            if self.rrtx:
                self.rrtx.update_sampling_distribution(safe_flat_map, Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
                self.rrtx.update_node_heuristics()
                
            print(f">>> [Main] Received Heatmap from {self.model_type} Worker!")
            self.pending_model_request = False
            self.sampling_map_updated = True
            self.initial_map_loaded = True # Đã nhận được map đầu tiên
        except queue.Empty:
            pass
        except Exception as e:
            print(f">>> [Main] Lỗi lúc nạp heatmap: {e}")

    def get_model_input_from_pygame(self):
        w, h = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        map_surf = pygame.Surface((w, h))
        map_surf.fill((255, 255, 255)) 
        
        for py_obs in self.py_obstacles:
            if not py_obs.get('active', True): continue
            if py_obs['shape'] == 'rectangle':
                vertices = self.get_rect_vertices(py_obs['x'], py_obs['y'], py_obs['w'], py_obs['h'], py_obs['angle'])
                pygame.draw.polygon(map_surf, (0, 0, 0), vertices)
            elif py_obs['shape'] == 'circle':
                pygame.draw.circle(map_surf, (0, 0, 0), (int(py_obs['x']), int(py_obs['y'])), int(py_obs['r']))
        
        map_arr = pygame.surfarray.array3d(map_surf).transpose(1, 0, 2)
        
        points_surf = pygame.Surface((w, h))
        points_surf.fill((255, 255, 255)) 
        
        start_pos = None
        if self.rrtx and hasattr(self.rrtx, 'v_bot') and self.rrtx.v_bot is not None:
            start_pos = (int(self.rrtx.v_bot.pos[0]), int(self.rrtx.v_bot.pos[1]))
        elif self.start_node:
            start_pos = (int(self.start_node.pos[0]), int(self.start_node.pos[1]))

        if start_pos: pygame.draw.circle(points_surf, COLOR_START, start_pos, 10)
        if self.goal_node: pygame.draw.circle(points_surf, COLOR_GOAL, (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])), 10)
        
        points_arr = pygame.surfarray.array3d(points_surf).transpose(1, 0, 2)
        return map_arr, points_arr
    
    def update_model_heuristic(self):
        if not self.use_heuristic or self.pending_model_request: return
        map_img, points_img = self.get_model_input_from_pygame()
        self.model_input_queue.put((map_img, points_img))
        self.pending_model_request = True
        print(f">>> [Main] Sent request to {self.model_type} Worker...")

    def init_algorithm(self):
        print(">>> Initializing RRTx Algorithm...")
        self.model = HolonomicModel(self.obstacles)
        self.rrtx = RRTx(self.start_node, self.goal_node, self.model)
        self.update_model_heuristic()

    def is_point_inside_polygon(self, point, vertices):
        x, y = point
        n = len(vertices)
        inside = False
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters: inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def handle_obstacle_click(self, pos, is_left_click):
        mx, my = pos
        if is_left_click:
            w, h, angle = 60, 60, 0
            obs_cpp = Rectangle(mx, my, w, h, angle)
            self._gc_protector.append(obs_cpp)
            self.py_obstacles.append({'shape': 'rectangle', 'x': mx, 'y': my, 'w': w, 'h': h, 'angle': angle, 'type': 'static', 'cpp_ref': obs_cpp})
            self.obstacles.append(obs_cpp)
            if self.current_state == STATE_RUNNING and self.rrtx:
                r = self.rrtx.shrinking_ball_radius()
                self.rrtx.update_obstacles(r, self.obstacles)
                self.sampling_map_updated = False 
        else:
            for i in range(len(self.py_obstacles) - 1, -1, -1):
                py_obs = self.py_obstacles[i]
                inside = False
                if py_obs['shape'] == 'rectangle':
                    verts = self.get_rect_vertices(py_obs['x'], py_obs['y'], py_obs['w'], py_obs['h'], py_obs['angle'])
                    inside = self.is_point_inside_polygon((mx, my), verts)
                elif py_obs['shape'] == 'circle':
                    inside = ((mx - py_obs['x'])**2 + (my - py_obs['y'])**2) <= (py_obs['r']**2)
                
                if inside:
                    self.py_obstacles.pop(i)
                    self.obstacles = [p['cpp_ref'] for p in self.py_obstacles]
                    if self.current_state == STATE_RUNNING and self.rrtx:
                        r = self.rrtx.shrinking_ball_radius()
                        self.rrtx.update_obstacles(r, self.obstacles)
                        self.sampling_map_updated = False
                    break

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    if event.key == pygame.K_RETURN:
                        if self.current_state == STATE_SETUP_OBSTACLES:
                            self.current_state = STATE_SET_START
                        elif self.current_state == STATE_TEST_MODEL:
                            self.current_state = STATE_RUNNING
                            self.init_algorithm()
                    if self.current_state == STATE_TEST_MODEL and event.key == pygame.K_SPACE:
                        self.update_model_heuristic()
                    if self.current_state == STATE_RUNNING:
                        if event.key == pygame.K_t: self.show_tree = not self.show_tree
                        elif event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if self.current_state == STATE_SETUP_OBSTACLES: self.handle_obstacle_click((mx, my), event.button == 1)
                    elif self.current_state == STATE_SET_START and event.button == 1:
                        self.start_node = Node(float(mx), float(my))
                        self.current_state = STATE_SET_GOAL
                    elif self.current_state == STATE_SET_GOAL and event.button == 1:
                        self.goal_node = Node(float(mx), float(my))
                        self.current_state = STATE_TEST_MODEL
                        self.update_model_heuristic()
                    elif self.current_state == STATE_RUNNING: self.handle_obstacle_click((mx, my), event.button == 1)
                    elif self.current_state == STATE_TEST_MODEL:
                        self.handle_obstacle_click((mx, my), event.button == 1)
                        self.update_model_heuristic()

            self.check_model_result()
            
            # Cờ cho phép chạy mượt mà ngay sau khi map đầu tiên được nạp
            is_model_ready = (not self.use_heuristic) or self.initial_map_loaded

            if self.current_state == STATE_RUNNING and self.rrtx:
                if not self.sampling_map_updated: self.update_model_heuristic()

                if is_model_ready and not self.paused:
                    current_time = pygame.time.get_ticks()
                    
                    # 1. CẬP NHẬT CHƯỚNG NGẠI VẬT (Tần số nhanh)
                    if current_time - self.last_obs_move_time > self.obstacle_move_delay:
                        self.last_obs_move_time = current_time
                        needs_cpp_update = False
                        new_obstacles_cpp = []

                        # Lấy vị trí hiện tại của robot
                        bot_pos = None
                        if hasattr(self.rrtx, 'v_bot') and self.rrtx.v_bot:
                            bot_pos = self.rrtx.v_bot.pos
                        
                        for py_obs in self.py_obstacles:
                            # 1. KIỂM TRA LOGIC PROXIMITY
                            if py_obs['type'] == 'proximity' and bot_pos and not py_obs.get('triggered', False):
                                dist = math.hypot(py_obs['x'] - bot_pos[0], py_obs['y'] - bot_pos[1])
                                
                                if dist < py_obs['trigger_dist']:
                                    # Chuyển đổi trạng thái
                                    if py_obs['behavior'] == 'appear_when_near':
                                        py_obs['active'] = True
                                    else: # disappear_when_near
                                        py_obs['active'] = False

                                    py_obs['triggered'] = True
                                    needs_cpp_update = True

                            # 2. CHỈ ĐƯA VÀO C++ NẾU ĐANG ACTIVE (Đây là mấu chốt)
                            if py_obs.get('active', True):
                                # Xử lý di chuyển cho dynamic
                                if py_obs['type'] == 'dynamic':
                                    needs_cpp_update = True
                                    py_obs['x'] += py_obs['velocity'][0]
                                    py_obs['y'] += py_obs['velocity'][1]
                                    b = py_obs['bounds']
                                    
                                    if py_obs['x'] < b['x_min'] or py_obs['x'] > b['x_max']: py_obs['velocity'][0] *= -1
                                    if py_obs['y'] < b['y_min'] or py_obs['y'] > b['y_max']: py_obs['velocity'][1] *= -1

                                    if py_obs['shape'] == 'circle':
                                        new_cpp = Circle(py_obs['x'], py_obs['y'], py_obs['r'])
                                    else:
                                        new_cpp = Rectangle(py_obs['x'], py_obs['y'], py_obs['w'], py_obs['h'], py_obs['angle'])
                                    
                                    self._gc_protector.append(new_cpp)
                                    py_obs['cpp_ref'] = new_cpp
                                
                                # Đưa vào danh sách gửi xuống thuật toán
                                new_obstacles_cpp.append(py_obs['cpp_ref'])

                        if needs_cpp_update:
                            self.obstacles = new_obstacles_cpp
                            r = self.rrtx.shrinking_ball_radius()
                            self.rrtx.update_obstacles(r, self.obstacles)
                            self.sampling_map_updated = False
                    
                    should_move_robot = False
                    reached_goal = False
                    if hasattr(self.rrtx, 'v_bot') and self.rrtx.v_bot is not None and self.goal_node is not None:
                        dist_to_goal = math.hypot(
                            self.rrtx.v_bot.pos[0] - self.goal_node.pos[0], 
                            self.rrtx.v_bot.pos[1] - self.goal_node.pos[1]
                        )
                        if dist_to_goal < 5.0:
                            reached_goal = True

                    if current_time - self.last_robot_move_time > self.robot_move_delay:
                        self.last_robot_move_time = current_time
                        if not reached_goal:
                            should_move_robot = True
                        else:
                            if self.current_state == STATE_RUNNING:
                                print("\n>>> [Success] Robot đã chạm đích an toàn!")
                                self.current_state = STATE_TEST_MODEL
                                
                    self.rrtx.step(move_robot=should_move_robot)

            self.screen.fill(COLOR_BG)

            for py_obs in self.py_obstacles:
                if not py_obs.get('active', True): continue
                if py_obs['shape'] == 'rectangle':
                    verts = self.get_rect_vertices(py_obs['x'], py_obs['y'], py_obs['w'], py_obs['h'], py_obs['angle'])
                    pygame.draw.polygon(self.screen, COLOR_OBSTACLE, verts)
                    pygame.draw.lines(self.screen, COLOR_OBSTACLE_BORDER, True, verts, 2)
                elif py_obs['shape'] == 'circle':
                    cx, cy, r = int(py_obs['x']), int(py_obs['y']), int(py_obs['r'])
                    pygame.draw.circle(self.screen, COLOR_OBSTACLE, (cx, cy), r)
                    pygame.draw.circle(self.screen, COLOR_OBSTACLE_BORDER, (cx, cy), r, 2)

            if self.current_state == STATE_TEST_MODEL and self.heuristic_debug_surface:
                self.screen.blit(self.heuristic_debug_surface, (0, 0))

            if self.start_node: pygame.draw.circle(self.screen, COLOR_START, (int(self.start_node.pos[0]), int(self.start_node.pos[1])), 8)
            if self.goal_node: pygame.draw.circle(self.screen, COLOR_GOAL, (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])), 8)

            if self.current_state == STATE_RUNNING and self.rrtx and is_model_ready:
                if self.show_tree:
                    for node in self.rrtx.V:
                        if node.parent:
                            s, e = (int(node.pos[0]), int(node.pos[1])), (int(node.parent.pos[0]), int(node.parent.pos[1]))
                            if node.heuristic_val > 0.4: pygame.draw.line(self.screen, COLOR_HEURISTIC_TREE, s, e, 2)
                            else: pygame.draw.line(self.screen, COLOR_TREE, s, e, 1)
                                
                if hasattr(self.rrtx, 'v_bot') and self.rrtx.v_bot is not None:
                    bot_pos = (int(self.rrtx.v_bot.pos[0]), int(self.rrtx.v_bot.pos[1]))
                    pygame.draw.circle(self.screen, COLOR_ROBOT, bot_pos, 8)
                    
                    path = []
                    curr = self.rrtx.v_bot
                    while curr and curr.parent:
                        path.append((curr.pos[0], curr.pos[1]))
                        curr = curr.parent
                        if curr == self.goal_node: break
                    path.append((curr.pos[0], curr.pos[1]))
                    
                    if len(path) > 1: pygame.draw.lines(self.screen, COLOR_PATH, False, path, 3)

            self.draw_ui_overlay(is_model_ready if self.current_state == STATE_RUNNING else True)
            pygame.display.flip()
            self.clock.tick(60)

    def draw_ui_overlay(self, is_ready=True):
        status_text, instruct_text = "", ""
        if self.current_state == STATE_SETUP_OBSTACLES:
            status_text, instruct_text = "MODE: MAP EDITING", "[L-Click]: Add Obs | [ENTER]: Done"
        elif self.current_state == STATE_SET_START:
            status_text, instruct_text = "MODE: SET START", "[L-Click]: Place Start"
        elif self.current_state == STATE_SET_GOAL:
            status_text, instruct_text = "MODE: SET GOAL", "[L-Click]: Place Goal"
        elif self.current_state == STATE_TEST_MODEL:
            status_text, instruct_text = f"MODE: {self.model_type} VISUALIZATION", "[L-Click]: Move Obs | [SPACE]: Refresh | [ENTER]: Run"
        elif self.current_state == STATE_RUNNING:
            if not is_ready:
                status_text, instruct_text = "MODE: INITIALIZING AI...", "Waiting for Model to load and map heuristics..."
            else:
                cost = self.rrtx.v_bot.lmc if (hasattr(self.rrtx, 'v_bot') and self.rrtx.v_bot is not None) else 0
                cost_str = f"{cost:.2f}" if cost < float('inf') else "Inf"
                status_text, instruct_text = "MODE: RRTx RUNNING", f"Cost: {cost_str} | [L-Click]: Add Obs | [SPACE]: Pause"

        pygame.draw.rect(self.screen, (0,0,0), (0, 0, Config.SCREEN_WIDTH, 60))
        self.screen.blit(self.large_font.render(status_text, True, COLOR_ORPHAN), (10, 5))
        self.screen.blit(self.font.render(instruct_text, True, COLOR_TEXT), (10, 35))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RRTx Path Planning")
    parser.add_argument("--model", type=str, choices=["gan", "sfd", "none"], default="none")
    args = parser.parse_args()

    viz = None
    try:
        viz = Visualizer(model_type=args.model)
        viz.run()
    except Exception as e:
        print(f"\n>>> [Main] Chương trình văng lỗi: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        if viz and viz.use_heuristic and viz.worker_process:
            if viz.worker_process.is_alive():
                viz.worker_process.terminate()
                viz.worker_process.join(timeout=1)
                if viz.worker_process.is_alive(): viz.worker_process.kill()
        pygame.quit()
        sys.exit()