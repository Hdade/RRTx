import pygame
import sys
import numpy as np

from utils.config import *
from utils.node import Node
from utils.geometry import Rectangle
from utils.model import HolonomicModel
from utils.RRTx import RRTx

COLOR_BG = (20, 20, 30)           # Dark Navy (Nền tối cho ngầu)
COLOR_OBSTACLE = (50, 50, 60)     # Xám đậm
COLOR_OBSTACLE_BORDER = (200, 200, 200)
COLOR_TREE = (0, 100, 255, 50)    # Xanh dương nhạt (alpha thấp)
COLOR_PATH = (255, 50, 50)        # Đỏ tươi (Path)
COLOR_ROBOT = (255, 165, 0)       # Cam
COLOR_GOAL = (0, 255, 127)        # Xanh lá mạ
COLOR_START = (0, 191, 255)       # Xanh biển
COLOR_ORPHAN = (148, 0, 211)      # Tím (Node bị cô lập)

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("RRTX Algorithm - Senior Robotics Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)

        self.init_simulation()
        
        # UI Flags
        self.show_tree = True
        self.paused = False
        self.dynamic_triggered = False

    def init_simulation(self):
        # 1. Map Setup (Figure 1 Layout)
        self.start_node = Node(400, 550)
        self.goal_node = Node(400, 280)

        self.obstacles = []
        self.create_map()

        # 2. Algorithm Init
        self.model = HolonomicModel(self.obstacles)
        self.rrtx = RRTx(self.start_node, self.goal_node, self.model)
        
        # Tracking Stats
        self.start_time = pygame.time.get_ticks()

    def create_map(self):
        """Tạo map mô phỏng 'Bug Trap'"""
        # Tường bao
        self.obstacles.append(Rectangle(100, 300, 40, 580, 0)) # Trái
        self.obstacles.append(Rectangle(700, 300, 40, 580, 0)) # Phải
        self.obstacles.append(Rectangle(400, 50, 640, 40, 0))  # Trên
        
        # Đáy (hở giữa)
        self.obstacles.append(Rectangle(240, 580, 320, 40, 0))
        self.obstacles.append(Rectangle(560, 580, 320, 40, 0))

        # Hộp chữ U ngược ở giữa (Bẫy)
        self.obstacles.append(Rectangle(400, 180, 300, 40, 0)) # Nóc hộp
        self.obstacles.append(Rectangle(270, 280, 40, 240, 0)) # Cạnh trái hộp
        self.obstacles.append(Rectangle(530, 280, 40, 240, 0)) # Cạnh phải hộp

        # Obstacles nhỏ ngẫu nhiên (Noise)
        self.obstacles.append(Rectangle(300, 400, 30, 30, 45))
        self.obstacles.append(Rectangle(500, 400, 30, 30, 15))

    def trigger_dynamic_event(self):
        """Thả cửa chặn đường robot"""
        print(">>> WARNING: DYNAMIC OBSTACLE DETECTED! <<<")
        # Một thanh ngang xuất hiện chặn ngay lối vào của cái hộp
        new_obs = Rectangle(400, 420, 200, 30, 0)
        self.obstacles.append(new_obs)
        
        # Notify RRTX
        r = self.rrtx.shrinkingBallRadius()
        self.rrtx.updateObstacles(r, self.obstacles)
        self.dynamic_triggered = True

    def draw_obstacles(self):
        for obs in self.obstacles:
            # Lấy vertices để vẽ đa giác (hỗ trợ xoay)
            vertices = obs.get_vertices()
            pygame.draw.polygon(self.screen, COLOR_OBSTACLE, vertices)
            pygame.draw.lines(self.screen, COLOR_OBSTACLE_BORDER, True, vertices, 2)

    def draw_tree(self):
        """
        Vẽ cây RRTX. 
        Lưu ý: Để tối ưu, ta vẽ lên một Surface trong suốt thay vì vẽ trực tiếp.
        """
        if not self.show_tree:
            return

        # Chỉ vẽ 2000 node gần nhất nếu quá đông để giữ FPS
        nodes_to_draw = self.rrtx.V
        
        for node in nodes_to_draw:
            if node.parent:
                # Vẽ line từ node -> parent
                start_pos = (int(node.pos[0]), int(node.pos[1]))
                end_pos = (int(node.parent.pos[0]), int(node.parent.pos[1]))
                pygame.draw.line(self.screen, (50, 100, 100), start_pos, end_pos, 1)

    def draw_orphans(self):
        """Vẽ các node bị cô lập (Orphans) - Đặc trưng của RRTX"""
        if not self.rrtx.Orphans:
            return
        
        for node in self.rrtx.Orphans:
            pos = (int(node.pos[0]), int(node.pos[1]))
            pygame.draw.circle(self.screen, COLOR_ORPHAN, pos, 2)

    def draw_path(self):
        """Truy vết từ Robot về Goal (theo parent pointers)"""
        path = []
        curr = self.rrtx.v_bot
        
        # Safety limit để tránh vòng lặp vô tận nếu bug
        limit = 0
        while curr is not None and limit < 5000:
            path.append(curr.pos)
            if curr == self.rrtx.v_goal:
                break
            curr = curr.parent
            limit += 1
            
        if len(path) > 1:
            pygame.draw.lines(self.screen, COLOR_PATH, False, path, 4)

    def draw_ui(self):
        # Thông tin FPS và Status
        fps = int(self.clock.get_fps())
        nodes_count = len(self.rrtx.V)
        orphan_count = len(self.rrtx.Orphans)
        cost = self.rrtx.v_bot.lmc if self.rrtx.v_bot.lmc != float('inf') else "Inf"
        
        texts = [
            f"FPS: {fps} | Nodes: {nodes_count} | Orphans: {orphan_count}",
            f"Robot Cost (LMC): {cost}",
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
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.show_tree = not self.show_tree
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Thêm vật cản bằng chuột
                    mx, my = pygame.mouse.get_pos()
                    new_obs = Rectangle(mx, my, 50, 50, 0)
                    self.obstacles.append(new_obs)
                    # Notify RRTX update ngay lập tức
                    r = self.rrtx.shrinkingBallRadius()
                    self.rrtx.updateObstacles(r, self.obstacles)

            # 2. Logic Update
            if not self.paused:
                # Chạy 1 bước thuật toán
                # (Có thể gọi loop step nhiều lần để tăng tốc độ phát triển cây)
                for _ in range(1000): 
                    self.rrtx.step()

                # Trigger sự kiện động sau 3 giây
                current_time = pygame.time.get_ticks()
                if not self.dynamic_triggered and (current_time - self.start_time > 3000):
                    # Chỉ trigger khi robot đã đi được một chút (ví dụ cost < vô cực)
                    if self.rrtx.v_bot.lmc < float('inf'):
                         self.trigger_dynamic_event()

            # 3. Visualization Loop
            self.screen.fill(COLOR_BG)
            
            self.draw_obstacles()
            self.draw_tree()      # Vẽ cây (nền)
            self.draw_orphans()   # Vẽ các node bị gãy (nếu có)
            self.draw_path()      # Vẽ đường đi (nổi bật)
            
            # Vẽ Start / Goal
            pygame.draw.circle(self.screen, COLOR_START, (int(self.start_node.pos[0]), int(self.start_node.pos[1])), 8)
            pygame.draw.circle(self.screen, COLOR_GOAL, (int(self.goal_node.pos[0]), int(self.goal_node.pos[1])), 8)
            
            # Vẽ Robot
            bot_pos = (int(self.rrtx.v_bot.pos[0]), int(self.rrtx.v_bot.pos[1]))
            pygame.draw.circle(self.screen, COLOR_ROBOT, bot_pos, 6)
            # Vẽ vòng tròn bán kính tìm kiếm quanh robot
            r_search = self.rrtx.shrinkingBallRadius()
            pygame.draw.circle(self.screen, (100, 100, 100), bot_pos, int(r_search), 1)

            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    viz = Visualizer()
    viz.run()