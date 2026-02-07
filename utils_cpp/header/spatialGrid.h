#ifndef SPATIAL_GRID_H
#define SPATIAL_GRID_H

#include <unordered_map>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>

#include "node.h"

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

class SpatialGrid {
private:
    double cell_size;
    std::unordered_map<std::pair<int, int>, std::vector<Node*>, PairHash> grid;

    std::pair<int, int> _getKey(const Vec2D& pos) const {
        return {
            static_cast<int>(std::floor(pos.x / cell_size)),
            static_cast<int>(std::floor(pos.y / cell_size))
        };
    }

public:
    SpatialGrid(double size) : cell_size(size) {}

    void add(Node* node) {
        auto key = _getKey(node->pos);
        grid[key].push_back(node);
    }

    void remove(Node* node) {
        auto key = _getKey(node->pos);
        auto it = grid.find(key);
        
        if (it != grid.end()) {
            std::vector<Node*>& list = it->second;
            list.erase(std::remove(list.begin(), list.end(), node), list.end());
            if (list.empty()) {
                grid.erase(it);
            }
        }
    }

    std::vector<Node*> getNeighbors(const Vec2D& pos, double radius) {
        std::vector<Node*> candidates;
        auto center_key = _getKey(pos);
        int range_cells = static_cast<int>(std::ceil(radius / cell_size));

        for (int dx = -range_cells; dx <= range_cells; ++dx) {
            for (int dy = -range_cells; dy <= range_cells; ++dy) {
                std::pair<int, int> key = {center_key.first + dx, center_key.second + dy};
                auto it = grid.find(key);
                if (it != grid.end()) {
                    const auto& nodesInCell = it->second;
                    candidates.insert(candidates.end(), nodesInCell.begin(), nodesInCell.end());
                }
            }
        }
        
        return candidates;
    }
    
    void clear() {
        grid.clear();
    }
};

#endif