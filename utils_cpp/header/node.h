#ifndef NODE_H
#define NODE_H

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "geometry.h" 

struct Node {
    Vec2D pos; 

    double g, lmc;
    Node* parent;
    
    std::vector<Node*> children;
    std::vector<Node*> N0_in;
    std::vector<Node*> N0_out;
    std::vector<Node*> Nr_in;
    std::vector<Node*> Nr_out;

    Node(double x, double y) : pos(x, y), parent(NULL) {
        g = std::numeric_limits<double>::infinity();
        lmc = std::numeric_limits<double>::infinity();

        children.reserve(10);
    }

    std::pair<double, double> getKey() const {
        return {std::min(g, lmc), g};
    }
};

struct NodeComparator {
    bool operator()(const Node* a, const Node* b) const {
        auto key1_a = std::min(a->g, a->lmc);
        auto key1_b = std::min(b->g, b->lmc);

        if(std::abs(key1_a - key1_b) > 1e-9) {
            return key1_a > key1_b;
        }

        if(std::abs(a->g - b->g) > 1e-9) {
            return a->g > b->g;
        }

        return a > b;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Node& n) {
    os << "Node(" << std::fixed << std::setprecision(2) << n.pos.x << ", " << n.pos.y << " | g=" << n.g << ", lmc=" << n.lmc << ")";
    return os;
}

#endif