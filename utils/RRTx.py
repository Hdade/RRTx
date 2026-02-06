import heapq, math, random
from utils.node import *
from utils.model import *
from utils.Config import *
from utils.SpatialGrid import SpatialGrid

class RRTx:
    def __init__(self, start: Node, goal: Node, model):
        goal.g = goal.lmc = 0
        start.g = start.lmc = float('inf')

        self.v_start = start
        self.v_goal = goal
        self.v_bot = start

        self.V = [goal]
        self.Q = []
        self.Obstacles = []
        self.Orphans = set()

        self.model = model

        self.spatial_grid = SpatialGrid(GRID_SIZE)
        self.spatial_grid.add(goal)

    def unitBallVolume(self, d=2):
        """zeta_D calculate"""
        if d == 2:
            return math.pi
        
        if d == 3:
            return 4.0 * math.pi / 3.0
        
        return (math.pi ** (d / 2)) / math.gamma(d / 2 + 1)

    def shrinkingBallRadius(self):
        """Algorithm 7"""
        card_V = len(self.V) # |V|
        
        if card_V <= 1:
            return DELTA
        
        zeta_D = self.unitBallVolume(DIMENSION)
        term1 = (GAMMA / zeta_D) * (math.log(card_V) / card_V)
        r = math.pow(term1, 1.0 / DIMENSION)
        return min(r, DELTA)
    
    def verifyOrphan(self, v):
        """Algorithm 10"""
        if v in self.Q:
            self.Q.remove(v)
            heapq.heapify(self.Q)
        
        self.Orphans.add(v)

    def isSegmentInObstacle(self, v_pos, u_pos, o):
        return o.intersectSegment(v_pos, u_pos)
    
    def removeObstacle(self, o):
        """Algorithm 11"""
        if o in self.Obstacles:
            self.Obstacles.remove(o)

        nodes2Update = []

        for v in self.V:
            neighbors = v.N0_out + v.Nr_out
            for u in neighbors:
                if self.isSegmentInObstacle(v.pos, u.pos, o) and not self.isCollision(v, u):
                    nodes2Update.append(v)
                    break
        
        for v in nodes2Update:
            self.updateLMC(v)

            if v.lmc != v.g:
                self.verifyQueue(v)

    def addNewObstacle(self, o):
        """Algorithm 12"""
        self.Obstacles.append(o)

        for v in self.V:
            neighbors = v.N0_out + v.Nr_out
            for u in neighbors:
                if self.isSegmentInObstacle(v.pos, u.pos, o) and v.parent == u:
                    self.verifyOrphan(v)

    def propogateDescendants(self):
        """Algorithm 9"""
        orphanList = list(self.Orphans)
        
        while len(orphanList) > 0:
            v = orphanList.pop()
            for child in v.children:
                if child not in self.Orphans:
                    self.Orphans.add(child)
                    orphanList.append(child)

        for v in self.Orphans:
            neighborsToWarn = set(v.N0_out + v.Nr_out)

            if v.parent:
                neighborsToWarn.add(v.parent)

            for u in neighborsToWarn:
                if u not in self.Orphans:
                    u.g = float('inf')
                    self.verifyQueue(u)

        for v in self.Orphans:
            v.g = float('inf')
            v.lmc = float('inf')

            if v.parent:
                if v in v.parent.children:
                    v.parent.children.remove(v)
                v.parent = None

        self.Orphans.clear()

    def updateObstacles(self, r, newObstacles):
        """Algorithm 8"""
        vanished = [o for o in self.Obstacles if o not in newObstacles]
        if len(vanished) > 0:
            for o in vanished:
                self.removeObstacle(o)
            
            self.Obstacles = [o for o in self.Obstacles if o not in vanished]
            self.reduceInconsistency(r)

        appeared = [o for o in newObstacles if o not in self.Obstacles]
        if len(appeared) > 0:
            for o in appeared:
                self.addNewObstacle(o)

            self.Obstacles.extend(appeared)
            self.propogateDescendants()
            self.verifyQueue(self.v_bot)
            self.reduceInconsistency(r)

    def makeParentOf(self, v, u):
        if v.parent and v in v.parent.children:
            v.parent.children.remove(v)

        v.parent = u

        if u is not None:
            u.children.append(v)

    def updateRobot(self):
        if self.v_bot == self.v_goal or self.v_bot.parent is None:
            return self.v_bot
        
        return self.v_bot.parent

    def randomNode(self):
        # if random.random() <= 0.95:
        #     return self.v_start

        rx = np.random.uniform(0, X_DIM)
        ry = np.random.uniform(0, Y_DIM)
        return Node(rx, ry)

    def nearestNode(self, v):
        if not self.V:
            return None
        
        nearest_v = None
        minDist = float('inf')

        for u in self.V:
            dist = self.d(u, v)
            if dist < minDist:
                minDist = dist
                nearest_v = u

        return nearest_v

    def near(self, v, r):
        V_near = []
        for u in self.V:
            if self.d(u, v) <= r:
                V_near.append(u)

        return V_near

    def d(self, u, v):
        return self.model.distance(u, v)

    def saturate(self, v, v_nearest):
        newPos = self.model.steer(v_nearest, v, DELTA)
        return Node(newPos[0], newPos[1])

    def findParent(self, v, U, r):
        """Algorithm 6"""
        for u in U:
            dist_v_u = self.d(v, u)
            if dist_v_u <= r and v.lmc > dist_v_u + u.lmc and not self.isCollision(v, u):
                v.parent = u
                v.lmc = dist_v_u + u.lmc

    def isCollision(self, u, v):
        for obs in self.Obstacles:
            if obs.intersectSegment(u.pos, v.pos):
                return True
        return False
    
    def isInsideObstacle(self, v):
        for obs in self.Obstacles:
            if obs.isInside(v.pos[0], v.pos[1]):
                return True
        
        return False

    def extend(self, v, r):
        """Algorithm 2"""
        V_near = self.near(v, r)
        self.findParent(v, V_near, r)

        if v.parent is None:
            return
        
        self.V.append(v)
        for u in V_near:
            if not self.isCollision(v, u):
                v.N0_out.append(u)
                u.Nr_in.append(v)

            if not self.isCollision(u, v):
                u.Nr_out.append(v)
                v.N0_in.append(u)

    def cullNeighbors(self, v, r):
        """Algorithm 3"""
        for u in list(v.Nr_out):
            if self.d(v, u) > r and v.parent != u:
                v.Nr_out.remove(u)
                if v in u.Nr_in:
                    u.Nr_in.remove(v)

    def verifyQueue(self, v):
        """Algorithm 13"""
        if v in self.Q:
            self.Q.remove(v)
            heapq.heapify(self.Q)

        v_key = self.calculateKey(v)
        heapq.heappush(self.Q, v)

    def rewireNeighbors(self, v, r):
        """Algorithm 4"""
        if v.g - v.lmc > EPSILON:
            self.cullNeighbors(v, r)
            N_v_in = v.N0_in + v.Nr_in
            for u in N_v_in:
                if u == v.parent:
                    continue

                if u.lmc > self.d(u, v) + v.lmc:
                    u.lmc = self.d(u, v) + v.lmc
                    self.makeParentOf(u, v)
                    if u.g - u.lmc > EPSILON:
                        self.verifyQueue(u)

    def updateLMC(self, v):
        """Algorithm 14"""
        r = self.shrinkingBallRadius()
        self.cullNeighbors(v, r)

        potential_parents = v.N0_out + v.Nr_out
        for u in potential_parents:
            if u in self.Orphans or u.parent == v or self.isCollision(v, u):
                continue

            dist_v_u = self.d(v, u)

            if v.lmc > dist_v_u + u.lmc:
                v.lmc = dist_v_u + u.lmc
                self.makeParentOf(v, u)

    def calculateKey(self, v):
        """Key: (min(v.g, v.lmc), v.g)"""
        return (min(v.g, v.lmc), v.g)

    def reduceInconsistency(self, r):
        """Algorithm 5"""
        while len(self.Q) > 0:
            v_top = self.Q[0]
            key_top = self.calculateKey(v_top)
            key_bot = self.calculateKey(self.v_bot)
            
            isRobotConsistency = not (
                self.v_bot.lmc != self.v_bot.g
                or self.v_bot.g == float('inf')
                or self.v_bot in self.Q
            )

            if key_top >= key_bot and isRobotConsistency:
                break

            v = heapq.heappop(self.Q)
            if v.g - v.lmc > EPSILON:
                self.updateLMC(v)
                self.rewireNeighbors(v, r)

            v.g = v.lmc

    def getSensorData(self):
        return self.model.obstacles

    def obstacleHasChanged(self):
        current = self.getSensorData()
        set_known = set(self.Obstacles)
        set_current = set(current)
        return set_known != set_current
    
    def RRTx(self):
        """Algorithm 1"""
        while self.v_bot != self.v_goal:
            r = self.shrinkingBallRadius()

            currentVisibleObstacle = self.getSensorData()
            if self.obstacleHasChanged():
                self.updateObstacles(r, currentVisibleObstacle)

            if self.v_bot.g < float('inf') and self.v_bot != self.v_goal:
                self.v_bot = self.updateRobot()

            v = self.randomNode()
            v_nearest = self.nearestNode(v)

            if self.d(v, v_nearest) > DELTA:
                v = self.saturate(v, v_nearest)

            if not self.isInsideObstacle(v):
                self.extend(v, r)

            if v in self.V:
                self.rewireNeighbors(v, r)
                self.reduceInconsistency(r)
    
    def step(self):
        if self.v_bot == self.v_goal:
            return True
        
        r = self.shrinkingBallRadius()
        
        currentVisibleObstacle = self.getSensorData()
        if self.obstacleHasChanged():
            self.updateObstacles(r, currentVisibleObstacle)
        
        if self.v_bot.g < float('inf') and self.v_bot != self.v_goal:
            self.v_bot = self.updateRobot()
        
        v = self.randomNode()
        v_nearest = self.nearestNode(v)
        
        if v_nearest is None:
            return False
        
        if self.d(v, v_nearest) > DELTA:
            v = self.saturate(v, v_nearest)
        
        if not self.isInsideObstacle(v):
            self.extend(v, r)
        
        if v in self.V:
            self.rewireNeighbors(v, r)
            self.reduceInconsistency(r)
        
        return False