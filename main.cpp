#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "geometry.h"
#include "node.h"
#include "model.h"
#include "RRTx.h"

namespace py = pybind11;

PYBIND11_MODULE(rrtx_cpp, m) {
    m.doc() = "RRTx C++ Backend optimized plugin";
    auto config_m = m.def_submodule("config", "Simulation Constants");
    config_m.attr("SCREEN_WIDTH") = config::SCREEN_WIDTH;
    config_m.attr("SCREEN_HEIGHT") = config::SCREEN_HEIGHT;
    config_m.attr("FPS") = config::FPS;
    config_m.attr("MAX_ITER") = config::MAX_ITER;
    config_m.attr("DELTA") = config::DELTA;
    config_m.attr("EPSILON") = config::EPSILON;
    config_m.attr("GAMMA") = config::GAMMA;
    config_m.attr("GOAL_RADIUS") = config::GOAL_RADIUS;
    config_m.attr("X_DIM") = config::X_DIM;
    config_m.attr("Y_DIM") = config::Y_DIM;
    config_m.attr("GRID_SIZE") = config::GRID_SIZE;

    py::class_<Vec2D>(m, "Vec2D")
        .def(py::init<double, double>())
        .def_readwrite("x", &Vec2D::x)
        .def_readwrite("y", &Vec2D::y)
        .def("norm", &Vec2D::norm)
        .def("__getitem__", [](const Vec2D &v, size_t i) {
            if (i == 0) return v.x;
            if (i == 1) return v.y;
            throw py::index_error();
        });

    py::class_<Obstacle, std::shared_ptr<Obstacle>>(m, "Obstacle");

    py::class_<Circle, Obstacle, std::shared_ptr<Circle>>(m, "Circle")
        .def(py::init<double, double, double>());

    py::class_<Rectangle, Obstacle, std::shared_ptr<Rectangle>>(m, "Rectangle")
        .def(py::init<double, double, double, double, double>())
        .def("get_vertices", [](const Rectangle &r) {
            std::vector<std::vector<double>> verts;
            for(const auto& v : r.vertices) {
                verts.push_back({v.x, v.y});
            }
            return verts;
        });

    py::class_<Node>(m, "Node")
        .def(py::init<double, double>())
        .def_readwrite("pos", &Node::pos)
        .def_readwrite("g", &Node::g)
        .def_readwrite("lmc", &Node::lmc)
        .def_readwrite("parent", &Node::parent, py::return_value_policy::reference)
        .def_readwrite("heuristic_val", &Node::heuristic_val);

    py::class_<HolonomicModel>(m, "HolonomicModel")
        .def(py::init<const std::vector<Obstacle*>&>());

    py::class_<RRTx>(m, "RRTx")
        .def(py::init<Node*, Node*, HolonomicModel*>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def_readwrite("v_bot", &RRTx::v_bot, py::return_value_policy::reference)
        .def_readwrite("V", &RRTx::V, py::return_value_policy::reference)
        .def_readwrite("Orphans", &RRTx::Orphans, py::return_value_policy::reference)
        .def("step", &RRTx::step, py::arg("move_robot") = true)
        .def("shrinking_ball_radius", &RRTx::shrinkingBallRadius)
        .def("update_obstacles", &RRTx::updateObstacles)
        .def("update_sampling_distribution", &RRTx::updateSamplingDistribution)
        .def("obstacleHasChanged", &RRTx::obstacleHasChanged)
        .def("update_sampling_distribution", &RRTx::updateSamplingDistribution);
}