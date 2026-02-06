class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {} 

    def _get_key(self, pos):
        return (int(pos[0] / self.cell_size), int(pos[1] / self.cell_size))

    def add(self, node):
        key = self._get_key(node.pos)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(node)

    def remove(self, node):
        key = self._get_key(node.pos)
        if key in self.grid:
            if node in self.grid[key]:
                self.grid[key].remove(node)
            if not self.grid[key]:
                del self.grid[key]

    def get_neighbors(self, pos, radius):
        center_key = self._get_key(pos)
        range_cells = int(radius / self.cell_size) + 1
        
        candidates = []
        for dx in range(-range_cells, range_cells + 1):
            for dy in range(-range_cells, range_cells + 1):
                key = (center_key[0] + dx, center_key[1] + dy)
                if key in self.grid:
                    candidates.extend(self.grid[key])
        return candidates