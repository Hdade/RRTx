# utils/Config.py

# Simulation Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# RRTX Hyperparameters
MAX_ITER = 10000 
DELTA = 20.0            # Tăng bước nhảy một chút để vươn xa nhanh hơn
EPSILON = 5.0           # Tăng lên 5.0. Đừng quá cầu toàn với sai số < 5px.
GAMMA = 3000.0          # Giảm xuống. GAMMA quá lớn làm bán kính tìm kiếm r to, gây chậm.
                        # Công thức tham khảo: gamma approx 50 * (free_space)^(1/d)

# Map Dimensions
X_DIM = SCREEN_WIDTH
Y_DIM = SCREEN_HEIGHT
DIMENSION = 2

# Spatial Grid Settings (Mới)
GRID_SIZE = 40.0        # Kích thước mỗi ô lưới, nên >= DELTA