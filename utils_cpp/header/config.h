#ifndef CONFIG_H
#define CONFIG_H

namespace config {
    // Simulation Settings
    static constexpr int SCREEN_WIDTH = 800;
    static constexpr int SCREEN_HEIGHT = 600;
    static constexpr int FPS = 10;
    
    // RRTX Hyperparameters
    static constexpr int MAX_ITER = 10000;
    static constexpr double DELTA = 10.0;
    static constexpr double EPSILON = 1.0;
    static constexpr double GAMMA = 5000.0;
    static constexpr double GOAL_RADIUS = 20.0;

    // Map Dimensions
    static constexpr double X_DIM = static_cast<double>(SCREEN_WIDTH);
    static constexpr double Y_DIM = static_cast<double>(SCREEN_HEIGHT);
    static constexpr int DIMENSION = 2;

    // Spatial Grid Settings
    static constexpr double GRID_SIZE = 2.0 * DELTA;
};

#endif