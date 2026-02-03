from utils.Config import *

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTx:
    def __init__(self, start, goal, obstacle_list):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.node_list = [self.goal] # Root the tree at the GOAL
        self.robot_pos = Node(start[0], start[1])

    def get_dist(self, node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def get_nearest_node(self, random_node):
        dists = [self.get_dist(node, random_node) for node in self.node_list]
        min_idx = dists.index(min(dists))
        return self.node_list[min_idx]

    def steer(self, from_node, to_node):
        dist = self.get_dist(from_node, to_node)
        if dist > STEP_LEN:
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node = Node(from_node.x + STEP_LEN * math.cos(theta),
                            from_node.y + STEP_LEN * math.sin(theta))
        else:
            new_node = Node(to_node.x, to_node.y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.get_dist(from_node, new_node)
        return new_node

    def is_collision(self, node):
        for (ox, oy, size) in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            d = math.hypot(dx, dy)
            if d <= size:
                return True # Collision
        return False

    def is_collision_path(self, node1, node2):
        # Check collision along the line between two nodes
        dist = self.get_dist(node1, node2)
        steps = int(dist / 0.5) + 1
        dx = (node2.x - node1.x) / steps
        dy = (node2.y - node1.y) / steps
        
        curr_x, curr_y = node1.x, node1.y
        for _ in range(steps):
            if self.is_collision(Node(curr_x, curr_y)):
                return True
            curr_x += dx
            curr_y += dy
        return False

    def find_near_nodes(self, new_node):
        n_nodes = len(self.node_list) + 1
        r = min(REWIRE_RADIUS, 50.0 * math.sqrt((math.log(n_nodes) / n_nodes)))
        dist_list = [(self.get_dist(node, new_node), node) for node in self.node_list]
        near_nodes = [node for d, node in dist_list if d <= r]
        return near_nodes

    def rewire(self, new_node, near_nodes):
        # 1. Choose best parent for new_node
        for near_node in near_nodes:
            if self.is_collision_path(near_node, new_node):
                continue
            
            # Cost to root (Goal) via near_node
            new_cost = near_node.cost + self.get_dist(near_node, new_node)
            if new_cost < new_node.cost:
                new_node.parent = near_node
                new_node.cost = new_cost
        
        # 2. Rewire near_nodes to use new_node as parent if it's shorter
        for near_node in near_nodes:
            if near_node == new_node.parent: 
                continue
            if self.is_collision_path(new_node, near_node):
                continue
                
            new_cost = new_node.cost + self.get_dist(new_node, near_node)
            if new_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_cost

    def grow_tree(self, num_samples=1):
        """Standard RRT* expansion step"""
        for _ in range(num_samples):
            if random.randint(0, 100) > 10:
                rnd = Node(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
            else:
                rnd = Node(self.robot_pos.x, self.robot_pos.y) # Bias towards robot

            nearest_node = self.get_nearest_node(rnd)
            new_node = self.steer(nearest_node, rnd)

            if not self.is_collision(new_node) and not self.is_collision_path(nearest_node, new_node):
                near_nodes = self.find_near_nodes(new_node)
                self.node_list.append(new_node)
                self.rewire(new_node, near_nodes)

    def check_and_repair(self):
        """
        RRTx Key Feature: Check if current path is broken. 
        If yes, prune and force rapid regrowth.
        """
        # Find path from robot to goal (tracing parents)
        # Note: Since tree is rooted at Goal, we find nearest node to Robot and trace UP.
        nearest = self.get_nearest_node(self.robot_pos)
        path_node = nearest
        
        valid_path = True
        while path_node.parent is not None:
            if self.is_collision_path(path_node, path_node.parent):
                # Cut the edge
                path_node.parent = None 
                path_node.cost = float('inf')
                valid_path = False
                break
            path_node = path_node.parent
            
        return valid_path

    def move_robot(self):
        nearest = self.get_nearest_node(self.robot_pos)
        
        # Move towards the parent of the nearest node (towards Goal)
        if nearest.parent:
            theta = math.atan2(nearest.parent.y - nearest.y, nearest.parent.x - nearest.x)
            self.robot_pos.x += ROBOT_SPEED * math.cos(theta)
            self.robot_pos.y += ROBOT_SPEED * math.sin(theta)
        else:
            # If detached, move towards nearest valid node
            theta = math.atan2(nearest.y - self.robot_pos.x, nearest.x - self.robot_pos.x)
            self.robot_pos.x += ROBOT_SPEED * 0.5 * math.cos(theta)
            self.robot_pos.y += ROBOT_SPEED * 0.5 * math.sin(theta)

    def draw(self, ax):
        ax.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        # Draw Tree
        for node in self.node_list:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", linewidth=0.5, alpha=0.5)

        # Draw Obstacles
        for (ox, oy, size) in self.obstacle_list:
            circle = plt.Circle((ox, oy), size, color='k')
            ax.add_artist(circle)

        # Draw Start/Robot and Goal
        ax.plot(self.robot_pos.x, self.robot_pos.y, "xr", markersize=10, label="Robot")
        ax.plot(self.goal.x, self.goal.y, "xb", markersize=10, label="Goal")
        
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_title("RRTx Simulation: Dynamic Replanning")
        ax.grid(True)