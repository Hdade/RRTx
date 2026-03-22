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
RRTStar = rrtx_cpp.RRTStar

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
    def __init__(self, model_type="gan", planner_type="rrtx"):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))

        self.model_type = model_type.upper()
        self.planner_type = planner_type.lower()

        planner_name = "RRTx" if self.planner_type == "rrtx" else "RRT*"
        pygame.display.set_caption(f"{planner_name} C++ Backend Interactive Planner - {self.model_type} Model")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.large_font = pygame.font.SysFont("Consolas", 20, bold=True)

        self.current_state = STATE_SETUP_OBSTACLES

        self._gc_protector = []
        self.py_obstacles = []
        self.obstacles = []

        self.start_node = None
        self.goal_node = None
        self.model = None
        self.planner = None

        self.obstacle_move_delay = 50
        self.last_obs_move_time = 0

        self.robot_move_delay = 100
        self.last_robot_move_time = 0

        self.show_tree = True
        self.paused = False
        self.use_heuristic = True

        self.model_input_queue = multiprocessing.Queue()
        self.model_output_queue = multiprocessing.Queue()

        if self.model_type == "GAN":
            print(">>> [Main] Starting GAN Worker Process.")
            checkpoint_path = "checkpoints/GAN_checkpoint/netG_epoch_40.pth"
            target_func = gan_worker_loop
            target_args = (
                self.model_input_queue,
                self.model_output_queue,
                checkpoint_path,
                (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT),
            )
        elif self.model_type == "SFD":
            print(">>> [Main] Starting SFD Worker Process.")
            checkpoint_path = "checkpoints/SFD_checkpoint/best_model.pth"
            target_func = sfd_worker_loop
            target_args = (
                self.model_input_queue,
                self.model_output_queue,
                checkpoint_path,
                (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT),
            )
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
        self.initial_map_loaded = False
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
            (w / 2, thickness / 2, w, thickness),
            (w / 2, h - thickness / 2, w, thickness),
            (thickness / 2, h / 2, thickness, h),
            (w - thickness / 2, h / 2, thickness, h),
        ]
        for bx, by, bw, bh in borders:
            obs_cpp = Rectangle(bx, by, bw, bh, 0)
            self._gc_protector.append(obs_cpp)
            py_obj = {
                "shape": "rectangle",
                "x": bx,
                "y": by,
                "w": bw,
                "h": bh,
                "angle": 0,
                "type": "static",
                "cpp_ref": obs_cpp,
            }
            self.py_obstacles.append(py_obj)
            self.obstacles.append(obs_cpp)

    def load_scenario(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.planner = None
        self.model = None
        self.pending_model_request = False
        self.sampling_map_updated = False
        self.initial_map_loaded = False
        self.heuristic_debug_surface = None

        self.start_node = Node(float(data["start"][0]), float(data["start"][1]))
        self.goal_node = Node(float(data["goal"][0]), float(data["goal"][1]))

        self.py_obstacles = []
        self.obstacles = []
        self._gc_protector = []

        self.create_borders()

        for obs_data in data["obstacles"]:
            shape = obs_data.get("shape", "rectangle")
            current_pos = obs_data.get("center") if shape == "circle" else obs_data.get("position")
            x, y = float(current_pos[0]), float(current_pos[1])

            if shape == "circle":
                r = float(obs_data["radius"])
                py_obj = {"shape": "circle", "x": x, "y": y, "r": r}
                obs_cpp = Circle(x, y, r)
            else:
                if shape == "square":
                    w = float(obs_data["size"])
                    h = float(obs_data["size"])
                else:
                    w = float(obs_data["width"])
                    h = float(obs_data.get("height", w))
                angle = float(obs_data.get("rotation", 0))
                py_obj = {"shape": "rectangle", "x": x, "y": y, "w": w, "h": h, "angle": angle}
                obs_cpp = Rectangle(x, y, w, h, angle)

            self._gc_protector.append(obs_cpp)

            py_obj["type"] = obs_data.get("type", "static")

            if py_obj["type"] == "dynamic":
                py_obj["velocity"] = list(obs_data.get("velocity", [1.0, 1.0]))
                py_obj["bounds"] = obs_data.get(
                    "movement_bounds",
                    obs_data.get(
                        "bounds",
                        {
                            "x_min": 0,
                            "x_max": Config.SCREEN_WIDTH,
                            "y_min": 0,
                            "y_max": Config.SCREEN_HEIGHT,
                        },
                    ),
                )

            if py_obj["type"] == "proximity":
                py_obj["trigger_dist"] = obs_data.get("trigger_dist", 100.0)
                py_obj["behavior"] = obs_data.get("behavior", "appear_when_near")
                py_obj["active"] = False if py_obj["behavior"] == "appear_when_near" else True
                py_obj["triggered"] = False

            py_obj["cpp_ref"] = obs_cpp
            self.py_obstacles.append(py_obj)

        self.obstacles = [p["cpp_ref"] for p in self.py_obstacles if p.get("active", True)]

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
        if not self.use_heuristic:
            return
        try:
            flat_map = self.model_output_queue.get_nowait()
            safe_flat_map = np.ascontiguousarray(flat_map.flatten(), dtype=np.float64)
            self.heuristic_debug_surface = self.create_heatmap_surface_from_data(flat_map)

            if self.planner:
                self.planner.update_sampling_distribution(
                    safe_flat_map, Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
                )
                self.planner.update_node_heuristics()

            print(f">>> [Main] Received Heatmap from {self.model_type} Worker!")
            self.pending_model_request = False
            self.sampling_map_updated = True
            self.initial_map_loaded = True
        except queue.Empty:
            pass
        except Exception as e:
            print(f">>> [Main] Lỗi lúc nạp heatmap: {e}")

    def get_model_input_from_pygame(self):
        w, h = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        map_surf = pygame.Surface((w, h))
        map_surf.fill((255, 255, 255))

        for py_obs in self.py_obstacles:
            if not py_obs.get("active", True):
                continue
            if py_obs["shape"] == "rectangle":
                vertices = self.get_rect_vertices(
                    py_obs["x"], py_obs["y"], py_obs["w"], py_obs["h"], py_obs["angle"]
                )
                pygame.draw.polygon(map_surf, (0, 0, 0), vertices)
            elif py_obs["shape"] == "circle":
                pygame.draw.circle(
                    map_surf,
                    (0, 0, 0),
                    (int(py_obs["x"]), int(py_obs["y"])),
                    int(py_obs["r"]),
                )

        map_arr = pygame.surfarray.array3d(map_surf).transpose(1, 0, 2)

        points_surf = pygame.Surface((w, h))
        points_surf.fill((255, 255, 255))

        start_pos = None
        if self.planner and hasattr(self.planner, "v_bot") and self.planner.v_bot is not None:
            start_pos = (int(self.planner.v_bot.pos[0]), int(self.planner.v_bot.pos[1]))
        elif self.start_node:
            start_pos = (int(self.start_node.pos[0]), int(self.start_node.pos[1]))

        if start_pos:
            pygame.draw.circle(points_surf, COLOR_START, start_pos, 10)
        if self.goal_node:
            pygame.draw.circle(
                points_surf,
                COLOR_GOAL,
                (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])),
                10,
            )

        points_arr = pygame.surfarray.array3d(points_surf).transpose(1, 0, 2)
        return map_arr, points_arr

    def update_model_heuristic(self):
        if not self.use_heuristic or self.pending_model_request:
            return
        map_img, points_img = self.get_model_input_from_pygame()
        self.model_input_queue.put((map_img, points_img))
        self.pending_model_request = True
        print(f">>> [Main] Sent request to {self.model_type} Worker.")

    def init_algorithm(self):
        planner_name = "RRTx" if self.planner_type == "rrtx" else "RRT*"
        print(f">>> Initializing {planner_name} Algorithm.")

        self.model = HolonomicModel(self.obstacles)

        if self.planner_type == "rrtstar":
            self.planner = RRTStar(self.start_node, self.goal_node, self.model)
        else:
            self.planner = RRTx(self.start_node, self.goal_node, self.model)

        self.update_model_heuristic()

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
            w, h, angle = 60, 60, 0
            obs_cpp = Rectangle(mx, my, w, h, angle)
            self._gc_protector.append(obs_cpp)
            self.py_obstacles.append(
                {
                    "shape": "rectangle",
                    "x": mx,
                    "y": my,
                    "w": w,
                    "h": h,
                    "angle": angle,
                    "type": "static",
                    "cpp_ref": obs_cpp,
                }
            )
            self.obstacles.append(obs_cpp)
            if self.current_state == STATE_RUNNING and self.planner:
                r = self.planner.shrinking_ball_radius()
                self.planner.update_obstacles(r, self.obstacles)
                self.sampling_map_updated = False
        else:
            for i in range(len(self.py_obstacles) - 1, -1, -1):
                py_obs = self.py_obstacles[i]
                inside = False
                if py_obs["shape"] == "rectangle":
                    verts = self.get_rect_vertices(
                        py_obs["x"], py_obs["y"], py_obs["w"], py_obs["h"], py_obs["angle"]
                    )
                    inside = self.is_point_inside_polygon((mx, my), verts)
                elif py_obs["shape"] == "circle":
                    inside = ((mx - py_obs["x"]) ** 2 + (my - py_obs["y"]) ** 2) <= (py_obs["r"] ** 2)

                if inside:
                    self.py_obstacles.pop(i)
                    self.obstacles = [p["cpp_ref"] for p in self.py_obstacles if p.get("active", True)]
                    if self.current_state == STATE_RUNNING and self.planner:
                        r = self.planner.shrinking_ball_radius()
                        self.planner.update_obstacles(r, self.obstacles)
                        self.sampling_map_updated = False
                    break

    def draw_ui_overlay(self, is_ready=True):
        status_text, instruct_text = "", ""
        planner_label = "RRTx" if self.planner_type == "rrtx" else "RRT*"

        if self.current_state == STATE_SETUP_OBSTACLES:
            status_text, instruct_text = "MODE: MAP EDITING", "[L-Click]: Add Obs | [ENTER]: Done"
        elif self.current_state == STATE_SET_START:
            status_text, instruct_text = "MODE: SET START", "[L-Click]: Place Start"
        elif self.current_state == STATE_SET_GOAL:
            status_text, instruct_text = "MODE: SET GOAL", "[L-Click]: Place Goal"
        elif self.current_state == STATE_TEST_MODEL:
            status_text, instruct_text = (
                f"MODE: {self.model_type} VISUALIZATION",
                "[L-Click]: Move Obs | [SPACE]: Refresh | [ENTER]: Run",
            )
        elif self.current_state == STATE_RUNNING:
            if not is_ready:
                status_text, instruct_text = "MODE: INITIALIZING AI...", "Waiting for first heatmap..."
            else:
                cost = (
                    self.planner.v_bot.lmc
                    if (self.planner and hasattr(self.planner, "v_bot") and self.planner.v_bot is not None)
                    else 0.0
                )
                cost_str = f"{cost:.2f}" if cost < float("inf") else "Inf"
                status_text = f"MODE: {planner_label} RUNNING"
                instruct_text = f"Cost: {cost_str} | [P]: Pause | [T]: Toggle Tree"

        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, Config.SCREEN_WIDTH, 60))
        self.screen.blit(self.large_font.render(status_text, True, COLOR_ORPHAN), (10, 5))
        self.screen.blit(self.font.render(instruct_text, True, COLOR_TEXT), (10, 35))

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
                        elif self.current_state == STATE_TEST_MODEL:
                            self.current_state = STATE_RUNNING
                            self.init_algorithm()

                    if self.current_state == STATE_TEST_MODEL and event.key == pygame.K_SPACE:
                        self.update_model_heuristic()

                    if self.current_state == STATE_RUNNING and event.key == pygame.K_p:
                        self.paused = not self.paused

                    if event.key == pygame.K_t:
                        self.show_tree = not self.show_tree

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.current_state == STATE_SETUP_OBSTACLES:
                        if event.button == 1:
                            self.handle_obstacle_click(event.pos, True)
                        elif event.button == 3:
                            self.handle_obstacle_click(event.pos, False)

                    elif self.current_state == STATE_SET_START and event.button == 1:
                        self.start_node = Node(float(event.pos[0]), float(event.pos[1]))
                        self.current_state = STATE_SET_GOAL

                    elif self.current_state == STATE_SET_GOAL and event.button == 1:
                        self.goal_node = Node(float(event.pos[0]), float(event.pos[1]))
                        self.current_state = STATE_TEST_MODEL
                        self.update_model_heuristic()

            self.check_model_result()

            current_time = pygame.time.get_ticks()
            is_model_ready = (not self.use_heuristic) or self.initial_map_loaded

            if self.current_state == STATE_RUNNING and self.planner and not self.paused:
                if current_time - self.last_obs_move_time > self.obstacle_move_delay:
                    self.last_obs_move_time = current_time

                    needs_cpp_update = False
                    new_obstacles_cpp = []

                    bot_x = None
                    bot_y = None
                    if hasattr(self.planner, "v_bot") and self.planner.v_bot is not None:
                        bot_x = self.planner.v_bot.pos[0]
                        bot_y = self.planner.v_bot.pos[1]

                    for py_obs in self.py_obstacles:
                        if py_obs.get("type") == "dynamic":
                            needs_cpp_update = True
                            py_obs["x"] += py_obs["velocity"][0]
                            py_obs["y"] += py_obs["velocity"][1]
                            b = py_obs["bounds"]

                            if py_obs["x"] < b["x_min"] or py_obs["x"] > b["x_max"]:
                                py_obs["velocity"][0] *= -1
                                py_obs["x"] = max(b["x_min"], min(py_obs["x"], b["x_max"]))
                            if py_obs["y"] < b["y_min"] or py_obs["y"] > b["y_max"]:
                                py_obs["velocity"][1] *= -1
                                py_obs["y"] = max(b["y_min"], min(py_obs["y"], b["y_max"]))

                        if py_obs.get("type") == "proximity":
                            if bot_x is not None and bot_y is not None:
                                dist = math.hypot(py_obs["x"] - bot_x, py_obs["y"] - bot_y)
                                if py_obs["behavior"] == "appear_when_near":
                                    new_active = dist <= py_obs["trigger_dist"]
                                else:
                                    new_active = dist > py_obs["trigger_dist"]
                                if py_obs.get("active", True) != new_active:
                                    needs_cpp_update = True
                                py_obs["active"] = new_active

                        if py_obs["shape"] == "circle":
                            new_cpp = Circle(py_obs["x"], py_obs["y"], py_obs["r"])
                        else:
                            new_cpp = Rectangle(
                                py_obs["x"], py_obs["y"], py_obs["w"], py_obs["h"], py_obs["angle"]
                            )

                        self._gc_protector.append(new_cpp)
                        py_obs["cpp_ref"] = new_cpp

                        if py_obs.get("active", True):
                            new_obstacles_cpp.append(new_cpp)

                    if needs_cpp_update:
                        self.obstacles = new_obstacles_cpp
                        r = self.planner.shrinking_ball_radius()
                        self.planner.update_obstacles(r, self.obstacles)
                        self.sampling_map_updated = False
                        if self.use_heuristic:
                            self.update_model_heuristic()

                should_move_robot = False
                reached_goal = False

                if hasattr(self.planner, "v_bot") and self.planner.v_bot is not None and self.goal_node is not None:
                    dist_to_goal = math.hypot(
                        self.planner.v_bot.pos[0] - self.goal_node.pos[0],
                        self.planner.v_bot.pos[1] - self.goal_node.pos[1],
                    )
                    if dist_to_goal < 5.0:
                        reached_goal = True

                if current_time - self.last_robot_move_time > self.robot_move_delay:
                    self.last_robot_move_time = current_time
                    if not reached_goal:
                        should_move_robot = True
                    else:
                        print("\n>>> [Success] Robot đã chạm đích an toàn!")
                        self.current_state = STATE_TEST_MODEL

                self.planner.step(move_robot=should_move_robot)

            self.screen.fill(COLOR_BG)

            for py_obs in self.py_obstacles:
                if not py_obs.get("active", True):
                    continue
                if py_obs["shape"] == "rectangle":
                    verts = self.get_rect_vertices(
                        py_obs["x"], py_obs["y"], py_obs["w"], py_obs["h"], py_obs["angle"]
                    )
                    pygame.draw.polygon(self.screen, COLOR_OBSTACLE, verts)
                    pygame.draw.lines(self.screen, COLOR_OBSTACLE_BORDER, True, verts, 2)
                elif py_obs["shape"] == "circle":
                    cx, cy, r = int(py_obs["x"]), int(py_obs["y"]), int(py_obs["r"])
                    pygame.draw.circle(self.screen, COLOR_OBSTACLE, (cx, cy), r)
                    pygame.draw.circle(self.screen, COLOR_OBSTACLE_BORDER, (cx, cy), r, 2)

            if self.current_state == STATE_TEST_MODEL and self.heuristic_debug_surface:
                self.screen.blit(self.heuristic_debug_surface, (0, 0))

            if self.start_node:
                pygame.draw.circle(
                    self.screen,
                    COLOR_START,
                    (int(self.start_node.pos[0]), int(self.start_node.pos[1])),
                    8,
                )
            if self.goal_node:
                pygame.draw.circle(
                    self.screen,
                    COLOR_GOAL,
                    (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])),
                    8,
                )

            if self.current_state == STATE_RUNNING and self.planner and is_model_ready:
                if self.show_tree:
                    for node in self.planner.V:
                        if node.parent:
                            s = (int(node.pos[0]), int(node.pos[1]))
                            e = (int(node.parent.pos[0]), int(node.parent.pos[1]))
                            if getattr(node, "heuristic_val", 0.0) > 0.4:
                                pygame.draw.line(self.screen, COLOR_HEURISTIC_TREE, s, e, 2)
                            else:
                                pygame.draw.line(self.screen, COLOR_TREE, s, e, 1)

                if hasattr(self.planner, "v_bot") and self.planner.v_bot is not None:
                    bot_pos = (int(self.planner.v_bot.pos[0]), int(self.planner.v_bot.pos[1]))
                    pygame.draw.circle(self.screen, COLOR_ROBOT, bot_pos, 8)

                    path = []
                    curr = self.planner.v_bot
                    visited = set()
                    while curr is not None and curr not in visited:
                        visited.add(curr)
                        path.append((curr.pos[0], curr.pos[1]))
                        if curr.parent is None:
                            break
                        curr = curr.parent

                    if len(path) > 1:
                        pygame.draw.lines(self.screen, COLOR_PATH, False, path, 3)

            self.draw_ui_overlay(is_model_ready if self.current_state == STATE_RUNNING else True)
            pygame.display.flip()
            self.clock.tick(60)

        if self.worker_process is not None:
            self.worker_process.terminate()
            self.worker_process.join()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Interactive RRTx / RRT* Path Planning Visualization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gan",
        choices=["gan", "sfd", "none"],
        help="Heuristic model to guide planner",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="rrtx",
        choices=["rrtx", "rrtstar"],
        help="Planner algorithm",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="annotations1.json",
        help="Path to scenario JSON file",
    )

    args = parser.parse_args()

    print("\n==============================")
    print(" RRT Planner Configuration")
    print("==============================")
    print(f"Planner   : {args.planner}")
    print(f"Model     : {args.model}")
    print(f"Scenario  : {args.scenario}")
    print("==============================\n")

    vis = Visualizer(model_type=args.model, planner_type=args.planner)
    if args.scenario:
        vis.load_scenario(args.scenario)
    vis.run()