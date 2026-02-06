import math

class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = float(cell_size)
        self.grid = {}

    def _getKey(self, pos):
        return (
            int(math.floor(pos[0] / self.cell_size)),
            int(math.floor(pos[1] / self.cell_size))
        )

    def add(self, node):
        key = self._getKey(node.pos)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(node)

    def remove(self, node):
        key = self._getKey(node.pos)
        if key in self.grid:
            if node in self.grid[key]:
                self.grid[key].remove(node)
            if not self.grid[key]:
                del self.grid[key]

    def getNeighbors(self, pos, radius):
        center_key = self._getKey(pos)
        range_cells = int(math.ceil(radius / self.cell_size))
        
        candidates = []
        for dx in range(-range_cells, range_cells + 1):
            for dy in range(-range_cells, range_cells + 1):
                key = (center_key[0] + dx, center_key[1] + dy)
                if key in self.grid:
                    candidates.extend(self.grid[key])
                    
        return candidates