from utils.RRTx import *

# --- Main Execution ---
print("Initializing RRTx Simulation...")
# Initial Obstacles
obstacles = [(25, 25, 6), (40, 10, 5), (10, 40, 5)] 
rrtx = RRTx(start=[5, 5], goal=[45, 45], obstacle_list=obstacles)

fig, ax = plt.subplots()

# 1. Initial Build (Static Environment)
print(f"Building initial tree ({MAX_ITER} nodes)...")
for _ in range(MAX_ITER):
    rrtx.grow_tree()

# 2. Dynamic Loop
time_step = 0
while True:
    # A. Move Obstacle to block the path dynamically
    if  0 < time_step < 100:
        obstacles[0] = (25 + (time_step)*0.1, 25 - (time_step)*0.1, 6)
        obstacles[1] = (40 - (time_step)*0.1, 10 + (time_step)*0.1, 5)
        obstacles[2] = (10 + (time_step)*0.1, 40 - (time_step)*0.1, 5)
    
    # B. Check for broken edges & Repair (Heavy sampling when broken)
    is_path_valid = rrtx.check_and_repair()
    
    # If path is broken, we spend more time "thinking" (sampling) to fix it
    repair_iterations = REPAIR_ITER if not is_path_valid else NORMAL_ITER
    rrtx.grow_tree(num_samples=repair_iterations)
    
    # C. Move Robot
    if rrtx.get_dist(rrtx.robot_pos, rrtx.goal) > 2.0:
        rrtx.move_robot()
    else:
        print("Goal Reached!")
        break

    # D. Visualize
    rrtx.draw(ax)
    plt.pause(0.01)
    time_step += 1

