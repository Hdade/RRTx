#include "RRTStar.h"

double RRTStar::unitBallVolume(int d)
{
    if (d == 2)
    {
        return M_PI;
    }

    if (d == 3)
    {
        return 4.0 * M_PI / 3.0;
    }

    return std::pow(M_PI, d / 2.0) / std::tgamma(d / 2.0 + 1);
}

double RRTStar::shrinkingBallRadius()
{
    double card_V = static_cast<double>(V.size());
    if (card_V <= 1)
    {
        return config::DELTA;
    }

    double zeta_D = unitBallVolume(config::DIMENSION);
    double term1 = (config::GAMMA / zeta_D) * (std::log(card_V) / card_V);
    double r = std::pow(term1, 1.0 / config::DIMENSION);

    return std::max(r, config::DELTA * 1.5);
}

std::pair<double, double> RRTStar::calculateKey(Node *v)
{
    // Key: (min(v.g, v.lmc), v.g)
    return {std::min(v->g, v->lmc), v->g};
}

double RRTStar::d(Node *u, Node *v)
{
    return model->distance(u, v);
}

bool RRTStar::isCollision(Node *u, Node *v)
{
    return model->checkCollision(u, v);
}

bool RRTStar::isInsideObstacle(Node *v)
{
    for (const auto &obs : Obstacles)
    {
        if (obs->isInside(v->pos))
        {
            return true;
        }
    }
    return false;
}

bool RRTStar::isSegmentInObstacle(const Vec2D &v_pos, const Vec2D &u_pos, Obstacle *o)
{
    return o->intersectSegment(v_pos, u_pos);
}

void RRTStar::updateSamplingDistribution(const std::vector<double> &flat_map, int width, int height)
{
    this->heuristic_map = flat_map;
    this->map_width = width;
    this->map_height = height;
}

void RRTStar::updateNodeHeuristics()
{
    if (heuristic_map.empty())
        return;

    for (Node *v : V)
    {
        v->heuristic_val = getHeuristicProbability(v->pos.x, v->pos.y);
    }
}

double RRTStar::getHeuristicProbability(double x, double y)
{
    if (heuristic_map.empty())
    {
        return 0.0;
    }

    int px = static_cast<int>(x);
    int py = static_cast<int>(y);
    px = (px < 0) ? 0 : px;
    px = (px >= map_width) ? map_width - 1 : px;
    py = (py < 0) ? 0 : py;
    py = (py >= map_height) ? map_height - 1 : py;
    return heuristic_map[py * map_width + px];
}

Node *RRTStar::randomNode()
{
    std::uniform_real_distribution<double> coin_flip(0.0, 1.0);
    if (!heuristic_map.empty() && coin_flip(gen) < config::HEURISTIC_RATIO)
    {
        for (int i = 0; i < 100; ++i)
        {
            double rx = dis_x(gen);
            double ry = dis_y(gen);
            double prob = getHeuristicProbability(rx, ry);
            if (prob > coin_flip(gen))
            {
                Node *newNode = new Node(rx, ry);
                newNode->heuristic_val = getHeuristicProbability(rx, ry);
                return newNode;
            }
        }
    }

    double rx = dis_x(gen);
    double ry = dis_y(gen);
    Node *newNode = new Node(rx, ry);
    newNode->heuristic_val = getHeuristicProbability(rx, ry);
    return newNode;
}

Node *RRTStar::nearestNode(Node *v)
{
    if (V.empty())
        return nullptr;

    double search_radius = config::DELTA * 2.0;
    std::vector<Node *> candidates = spatial_grid->getNeighbors(v->pos, search_radius);
    if (candidates.empty())
    {
        candidates = V;
    }

    Node *nearest_v = nullptr;
    double minDist = std::numeric_limits<double>::infinity();
    for (Node *u : candidates)
    {
        if (u->lmc >= std::numeric_limits<double>::infinity())
        {
            continue;
        }

        double dist = d(u, v);
        if (dist < minDist)
        {
            minDist = dist;
            nearest_v = u;
        }
    }

    return nearest_v;
}

std::vector<Node *> RRTStar::near(Node *v, double r)
{
    std::vector<Node *> V_near;
    std::vector<Node *> candidates = spatial_grid->getNeighbors(v->pos, r);
    for (Node *u : candidates)
    {
        if (d(u, v) <= r)
        {
            V_near.push_back(u);
        }
    }

    return V_near;
}

Node *RRTStar::saturate(Node *v, Node *v_nearest)
{
    Vec2D newPos = model->steer(v_nearest, v, config::DELTA);
    Node *newNode = new Node(newPos.x, newPos.y);
    newNode->heuristic_val = getHeuristicProbability(newPos.x, newPos.y);
    return newNode;
}

void RRTStar::makeParentOf(Node *v, Node *u)
{
    if (v->parent == u)
        return;

    if (v->parent != nullptr)
    {
        auto &siblings = v->parent->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), v), siblings.end());
    }

    v->parent = u;
    if (u != nullptr)
    {
        u->children.push_back(v);
    }
}

void RRTStar::findParent(Node *v, const std::vector<Node *> &U, double r)
{
    makeParentOf(v, nullptr);
    v->lmc = std::numeric_limits<double>::infinity();
    for (Node *u : U)
    {
        double dist_v_u = d(v, u);
        if (dist_v_u > r || isCollision(v, u))
            continue;
        double potential_lmc = dist_v_u + u->lmc;
        if (v->lmc > potential_lmc)
        {
            v->lmc = potential_lmc;
            makeParentOf(v, u);
        }
    }
}

void RRTStar::verifyQueue(Node *v)
{
    if (Q.contains(v))
    {
        Q.update(v);
    }
    else
    {
        Q.insert(v);
    }
}

void RRTStar::reduceInconsistency(double r)
{
    while (!Q.empty())
    {
        Node *v_top = Q.top();
        auto key_top = calculateKey(v_top);
        auto key_bot = calculateKey(v_bot);

        bool isRobotConsistent = ((v_bot->lmc == v_bot->g) &&
                                  (v_bot->g != std::numeric_limits<double>::infinity()) &&
                                  (!Q.contains(v_bot)));

        if (key_top >= key_bot && isRobotConsistent)
        {
            break;
        }

        Node *v = Q.pop();
        if (v->g - v->lmc > config::EPSILON)
        {
            updateLMC(v);
            rewireNeighbors(v, r);
        }

        v->g = v->lmc;
    }
}
void RRTStar::verifyOrphan(Node *v)
{
    if (Q.contains(v))
    {
        Q.remove(v);
    }
    Orphans.insert(v);
}

void RRTStar::propagateDescendants()
{
    std::vector<Node *> stack(Orphans.begin(), Orphans.end());
    // std::cout << "Propagating... Initial Orphans: " << stack.size() << std::endl;
    while (!stack.empty())
    {
        Node *v = stack.back();
        stack.pop_back();
        // if(v->children.size() > 0) {
        //     std::cout << "Node at (" << v->pos.x << ") propagate to " << v->children.size() << " children." << std::endl;
        // }
        for (Node *child : v->children)
        {
            if (Orphans.find(child) == Orphans.end())
            {
                Orphans.insert(child);
                stack.push_back(child);
            }
        }
    }

    for (Node *v : Orphans)
    {
        std::vector<Node *> neighborsToWarn = v->N0_out;
        neighborsToWarn.insert(neighborsToWarn.end(), v->Nr_out.begin(), v->Nr_out.end());
        if (v->parent != nullptr)
        {
            neighborsToWarn.push_back(v->parent);
        }

        for (Node *u : neighborsToWarn)
        {
            if (Orphans.find(u) == Orphans.end())
            {
                u->g = std::numeric_limits<double>::infinity();
                verifyQueue(u);
            }
        }
    }

    for (Node *v : Orphans)
    {
        v->g = std::numeric_limits<double>::infinity();
        v->lmc = std::numeric_limits<double>::infinity();
        if (v->parent != nullptr)
        {
            auto &siblings = v->parent->children;
            siblings.erase(std::remove(siblings.begin(), siblings.end(), v), siblings.end());
            v->parent = nullptr;
        }
    }

    Orphans.clear();
}

void RRTStar::cullNeighbors(Node *v, double r)
{
    auto it = v->Nr_out.begin();
    while (it != v->Nr_out.end())
    {
        Node *u = *it;
        if (d(v, u) > r && v->parent != u)
        {
            it = v->Nr_out.erase(it);
            auto &u_Nr_in = u->Nr_in;
            u_Nr_in.erase(std::remove(u_Nr_in.begin(), u_Nr_in.end(), v), u_Nr_in.end());
        }
        else
        {
            ++it;
        }
    }
}

void RRTStar::rewireNeighbors(Node *v, double r)
{
    if (v->g - v->lmc > config::EPSILON)
    {
        cullNeighbors(v, r);
        std::vector<Node *> N_v_in = v->N0_in;
        N_v_in.insert(N_v_in.end(), v->Nr_in.begin(), v->Nr_in.end());
        for (Node *u : N_v_in)
        {
            if (u == v->parent || isCollision(u, v))
            {
                continue;
            }

            double dist = d(u, v);
            if (u->lmc > dist + v->lmc)
            {
                u->lmc = dist + v->lmc;
                makeParentOf(u, v);
                if (u->g - u->lmc > config::EPSILON)
                {
                    verifyQueue(u);
                }
            }
        }
    }
}

void RRTStar::updateLMC(Node *v)
{
    double r = shrinkingBallRadius();
    cullNeighbors(v, r);

    v->lmc = std::numeric_limits<double>::infinity();
    Node *best_parent = nullptr;
    std::vector<Node *> potential_parents = v->N0_out;
    potential_parents.insert(potential_parents.end(), v->Nr_out.begin(), v->Nr_out.end());
    for (Node *u : potential_parents)
    {
        if (Orphans.find(u) != Orphans.end() || u->parent == v || isCollision(v, u))
        {
            continue;
        }

        double dist_v_u = d(v, u);
        double new_cost = dist_v_u + u->lmc;
        if (new_cost < v->lmc)
        {
            v->lmc = new_cost;
            best_parent = u;
        }
    }

    makeParentOf(v, best_parent);
}

void RRTStar::removeObstacle(Obstacle *o)
{
    auto it = std::remove(Obstacles.begin(), Obstacles.end(), o);
    if (it != Obstacles.end())
    {
        Obstacles.erase(it, Obstacles.end());
    }

    std::vector<Node *> nodes2Update;
    for (Node *v : V)
    {
        std::vector<Node *> neighbors = v->N0_out;
        neighbors.insert(neighbors.end(), v->Nr_out.begin(), v->Nr_out.end());
        for (Node *u : neighbors)
        {
            if (isSegmentInObstacle(v->pos, u->pos, o) && !isCollision(v, u))
            {
                nodes2Update.push_back(v);
                break;
            }
        }
    }

    for (Node *v : nodes2Update)
    {
        updateLMC(v);
        if (v->lmc != v->g)
        {
            verifyQueue(v);
        }
    }
}

void RRTStar::addNewObstacle(Obstacle *o)
{
    Obstacles.push_back(o);
    for (Node *v : V)
    {
        std::vector<Node *> neighbors = v->N0_out;
        neighbors.insert(neighbors.end(), v->Nr_out.begin(), v->Nr_out.end());
        for (Node *u : neighbors)
        {
            if (isSegmentInObstacle(v->pos, u->pos, o) && v->parent == u)
            {
                verifyOrphan(v);
            }
        }
    }
}

void RRTStar::updateObstacles(double r, const std::vector<Obstacle *> &newObstacles)
{
    std::vector<Obstacle *> vanished;
    for (Obstacle *o : Obstacles)
    {
        bool found = false;
        for (Obstacle *new_o : newObstacles)
        {
            if (o == new_o)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            vanished.push_back(o);
        }
    }

    if (!vanished.empty())
    {
        for (Obstacle *o : vanished)
        {
            removeObstacle(o);
        }

        model->obstacles = this->Obstacles;
        reduceInconsistency(r);
    }

    std::vector<Obstacle *> appeared;
    for (Obstacle *new_o : newObstacles)
    {
        bool found = false;
        if (new_o->active == false)
        {
            continue; // Skip inactive obstacles
        }
        for (Obstacle *o : Obstacles)
        {
            if (o == new_o)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            appeared.push_back(new_o);
        }
    }

    if (!appeared.empty())
    {
        for (Obstacle *o : appeared)
        {
            addNewObstacle(o);
        }
        model->obstacles = this->Obstacles;

        propagateDescendants();
        verifyQueue(v_bot);
        reduceInconsistency(r);
    }
}

std::vector<Obstacle *> RRTStar::getSensorData()
{
    return model->obstacles;
}

bool RRTStar::obstacleHasChanged()
{
    std::vector<Obstacle *> current = getSensorData();
    if (Obstacles.size() != current.size())
    {
        return true;
    }

    for (Obstacle *o : Obstacles)
    {
        bool found = false;
        for (Obstacle *cur_o : current)
        {
            if (o == cur_o)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            return true;
        }
    }

    return false;
}

Node *RRTStar::updateRobot()
{
    if (v_bot == v_goal)
    {
        return v_bot;
    }

    double dist_to_goal = d(v_bot, v_goal);
    if (dist_to_goal <= config::GOAL_RADIUS)
    {
        if (!isCollision(v_bot, v_goal))
        {
            std::cout << ">>> SNAP! Successfully reach to GOAL!" << std::endl;
            return v_goal;
        }
    }

    return (v_bot->parent == nullptr) ? v_bot : v_bot->parent;
}

void RRTStar::run()
{
    bool move_robot = false;
    while (v_bot != v_goal)
    {
        if (obstacleHasChanged())
        {
            std::vector<Obstacle *> currentVisibleObstacle = getSensorData();
            updateObstacles(shrinkingBallRadius(), currentVisibleObstacle);
        }
        if (this->isPathBroken())
        {
            this->resetTree();
            this->processRRTStar();
            move_robot = true;
        }
        if (move_robot && v_bot->lmc < std::numeric_limits<double>::infinity() && v_bot != v_goal)
        {
            v_bot = updateRobot();
        }
        if (V.size() > config::MAX_ITER)
        {
            break;
        }
    }
}

bool RRTStar::processRRTStar()
{
    if (v_bot == v_goal)
        return true;

    int no_improvement_count = 0;
    double best_cost = std::numeric_limits<double>::infinity();

    for (int i = 0; i < config::MAX_ITER; ++i)
    {
        this->total_iterations++;
        double r = shrinkingBallRadius();
        Node *v = randomNode();
        Node *v_nearest = nearestNode(v);

        if (v_nearest == nullptr)
        {
            delete v;
            continue;
        }

        if (d(v, v_nearest) > config::DELTA)
        {
            Node *v_saturated = saturate(v, v_nearest);
            delete v;
            v = v_saturated;
        }

        if (!isInsideObstacle(v))
        {
            extend(v, r);
            if (std::find(V.begin(), V.end(), v) != V.end())
            {
                rewireNeighbors(v, r);
                // reduceInconsistency(r);
            }
        }
        else
        {
            delete v;
        }

        v_goal->parent = nullptr;
        v_goal->lmc = std::numeric_limits<double>::infinity();
        std::vector<Node *> potential_parents = near(v_goal, r);
        if (!potential_parents.empty())
        {
            findParent(v_goal, potential_parents, r);
        }

        if (v_goal->lmc >= std::numeric_limits<double>::infinity())
        {
            potential_parents = near(v_goal, config::DELTA);
            if (!potential_parents.empty())
            {
                findParent(v_goal, potential_parents, config::DELTA);
            }
        }

        if (v_goal->lmc < std::numeric_limits<double>::infinity())
        {
            if (best_cost - v_goal->lmc > config::EPSILON)
            {
                best_cost = v_goal->lmc;
                no_improvement_count = 0;
            }
            else
            {
                no_improvement_count++;
            }

            if (no_improvement_count >= config::STABLE_THRESHOLD)
            {
                break;
            }
        }
    }

    if (V.size() > this->max_nodes)
    {
        this->max_nodes = V.size();
    }

    return v_goal->lmc < std::numeric_limits<double>::infinity();
}

void RRTStar::extend(Node *v, double r)
{
    std::vector<Node *> V_near = near(v, r);
    findParent(v, V_near, r);
    if (v->parent == nullptr)
    {
        return;
    }

    V.push_back(v);
    spatial_grid->add(v);
    for (Node *u : V_near)
    {
        if (!isCollision(v, u))
        {
            v->N0_out.push_back(u);
            u->Nr_in.push_back(v);
        }

        if (!isCollision(u, v))
        {
            u->Nr_out.push_back(v);
            v->N0_in.push_back(u);
        }
    }
}

bool RRTStar::isPathBroken()
{
    if (v_bot == v_goal)
        return false;

    Node *current = v_goal;
    while (current != nullptr && current != v_bot)
    {
        if (current->parent == nullptr || current->lmc >= std::numeric_limits<double>::infinity())
        {
            return true;
        }
        if (isCollision(current, current->parent))
        {
            return true;
        }

        current = current->parent;
    }
    return current != v_bot;
}

void RRTStar::resetTree()
{
    while (!Q.empty())
        Q.pop();
    Orphans.clear();

    // Xóa các node, chỉ giữ lại gốc mới (v_bot) và v_goal
    for (Node *n : V)
    {
        if (n != v_bot && n != v_goal && n != v_start)
        {
            delete n;
        }
    }
    V.clear();

    delete spatial_grid;
    spatial_grid = new SpatialGrid(config::GRID_SIZE);

    // Reset v_bot làm Root
    v_bot->g = 0.0;
    v_bot->lmc = 0.0;
    v_bot->parent = nullptr;
    v_bot->children.clear();
    v_bot->N0_in.clear();
    v_bot->N0_out.clear();
    v_bot->Nr_in.clear();
    v_bot->Nr_out.clear();

    // Reset v_goal
    v_goal->g = std::numeric_limits<double>::infinity();
    v_goal->lmc = std::numeric_limits<double>::infinity();
    v_goal->parent = nullptr;
    v_goal->children.clear();
    v_goal->N0_in.clear();
    v_goal->N0_out.clear();
    v_goal->Nr_in.clear();
    v_goal->Nr_out.clear();

    V.push_back(v_bot);
    spatial_grid->add(v_bot);
    Q.insert(v_bot);
}