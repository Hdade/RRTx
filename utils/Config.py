import math
import random
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
WIDTH, HEIGHT = 50, 50
MAX_ITER = 300       # Initial build iterations
REPAIR_ITER = 200     # Repair iterations when path is broken
NORMAL_ITER = 5      # Normal iterations when path is valid
REWIRE_RADIUS = 8.0  # Radius for checking neighbors
STEP_LEN = 2.0       # Growth step size
ROBOT_SPEED = 1.0    # How fast robot moves per frame