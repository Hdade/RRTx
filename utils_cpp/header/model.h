#ifndef HOLONOMIC_MODEL_H
#define HOLONOMIC_MODEL_H

#include<vector>
#include<cmath>

#include "config.h"
#include "node.h"
#include "geometry.h"

class HolonomicModel {
public:
    std::vector<Obstacle*> obstacles;

    HolonomicModel(const std::vector<Obstacle*>& obs) : obstacles(obs) {}

    double distance(const Node* n1, const Node* n2) const {
        return (n1->pos - n2->pos).norm();
    }

    Vec2D steer(const Node* nFrom, const Node* nTo, double delta = config::DELTA) const {
        double dist = distance(nFrom, nTo);

        if (dist <= delta || dist < 1e-9) {
            return nTo->pos;
        }

        Vec2D diff = nTo->pos - nFrom->pos;
        return nFrom->pos + (diff * (delta / dist));
    }

    bool checkCollision(const Node* n1, const Node* n2) const {
        for (const auto& obs : obstacles) {
            if (obs->intersectSegment(n1->pos, n2->pos)) {
                return true;
            }
        }
        
        return false;
    }
};

#endif