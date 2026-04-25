#ifndef RRTStar_H
#define RRTStar_H

#include <vector>
#include <queue>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

#include "config.h"
#include "node.h"
#include "geometry.h"
#include "model.h"
#include "spatialGrid.h"
#include "IndexedHeap.h"

class RRTStar
{
private:
    Node *v_start;
    Node *v_goal;
    std::vector<double> heuristic_map;
    int map_width = 0;
    int map_height = 0;
    double getHeuristicProbability(double x, double y);

public:
    int total_iterations = 0;
    int max_nodes = 0;
    Node *v_bot;
    std::vector<Node *> V;
    IndexedHeap Q;
    // std::vector<Node*> Q;
    std::vector<Obstacle *> Obstacles;
    std::unordered_set<Node *> Orphans;

    HolonomicModel *model;
    SpatialGrid *spatial_grid;

    std::mt19937 gen;
    std::uniform_real_distribution<> dis_x;
    std::uniform_real_distribution<> dis_y;

    RRTStar(Node *start, Node *goal, HolonomicModel *m) : v_start(start), v_goal(goal), v_bot(start), model(m)
    {
        std::random_device rd;
        gen = std::mt19937(rd());
        dis_x = std::uniform_real_distribution<>(0, config::X_DIM);
        dis_y = std::uniform_real_distribution<>(0, config::Y_DIM);
        spatial_grid = new SpatialGrid(config::GRID_SIZE);

        v_start->g = 0.0;
        v_start->lmc = 0.0;

        v_goal->g = std::numeric_limits<double>::infinity();
        v_goal->lmc = std::numeric_limits<double>::infinity();

        V.push_back(v_start);
        spatial_grid->add(v_start);
        Q.insert(v_start);
    }

    ~RRTStar()
    {
        if (spatial_grid)
        {
            delete spatial_grid;
            spatial_grid = nullptr;
        }

        for (Node *n : V)
        {
            if (n != v_start && n != v_goal)
            {
                delete n;
            }
        }

        V.clear();
        // Q.clear();
        Orphans.clear();
        Obstacles.clear();
    }

    // Math Helpers
    double unitBallVolume(int d = 2);
    double shrinkingBallRadius();
    std::pair<double, double> calculateKey(Node *v);
    double d(Node *u, Node *v);

    // Core RRT Functions
    Node *randomNode();
    Node *nearestNode(Node *v);
    std::vector<Node *> near(Node *v, double r);
    Node *saturate(Node *v, Node *v_nearest);
    void extend(Node *v, double r);
    void findParent(Node *v, const std::vector<Node *> &U, double r);

    // Collision Helpers
    bool isCollision(Node *u, Node *v);
    bool isInsideObstacle(Node *v);
    bool isSegmentInObstacle(const Vec2D &v_pos, const Vec2D &u_pos, Obstacle *o);

    // Dynamic Updates (Queue & Orphans)
    void verifyQueue(Node *v);
    void verifyOrphan(Node *v);
    void propagateDescendants();
    void makeParentOf(Node *v, Node *u);

    // Obstacle Management
    void updateObstacles(double r, const std::vector<Obstacle *> &newObstacles);
    void removeObstacle(Obstacle *o);
    void addNewObstacle(Obstacle *o);
    std::vector<Obstacle *> getSensorData();
    bool obstacleHasChanged();

    // Graph Rewiring
    void updateLMC(Node *v);
    void rewireNeighbors(Node *v, double r);
    void cullNeighbors(Node *v, double r);
    void reduceInconsistency(double r);

    // Main Loop
    Node *updateRobot();
    void run();
    bool step(bool move_robot);
    bool processRRTStar();
    bool isPathBroken();
    void resetTree();

    void updateSamplingDistribution(const std::vector<double> &flat_map, int width, int height);
    void updateNodeHeuristics();
};

#endif