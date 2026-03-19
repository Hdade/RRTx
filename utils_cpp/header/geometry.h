#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vec2D
{
    double x, y;

    Vec2D() : x(0), y(0) {}
    Vec2D(double x, double y) : x(x), y(y) {}

    Vec2D operator+(const Vec2D &other) const
    {
        return {x + other.x, y + other.y};
    }

    Vec2D operator-(const Vec2D &other) const
    {
        return {x - other.x, y - other.y};
    }

    Vec2D operator*(double scalar) const
    {
        return {x * scalar, y * scalar};
    }

    Vec2D operator/(double scalar) const
    {
        return {x / scalar, y / scalar};
    }

    double dot(const Vec2D &other) const
    {
        return x * other.x + y * other.y;
    }

    double norm() const
    {
        return std::sqrt(x * x + y * y);
    }

    double sqrNorm() const
    {
        return x * x + y * y;
    }

    Vec2D rotate(double cos_a, double sin_a) const
    {
        return {x * cos_a - y * sin_a, x * sin_a + y * cos_a};
    }
};

class Obstacle
{
public:
    bool active; // Thêm thuộc tính active để quản lý trạng thái của obstacle
    virtual ~Obstacle() = default;
    virtual bool isInside(const Vec2D &point) const = 0;
    virtual bool intersectSegment(const Vec2D &p1, const Vec2D &p2) const = 0;
};

class Circle : public Obstacle
{
public:
    Vec2D center;
    double r, r_sqr;

    Circle(double x, double y, double r, bool active = true) : center(x, y), r(r), r_sqr(r * r)
    {
        this->active = active;
    }

    bool isInside(const Vec2D &point) const override
    {
        if (this->active == false)
            return false; // Inactive obstacles are considered non-existent
        return (point - center).sqrNorm() <= r_sqr;
    }

    bool intersectSegment(const Vec2D &p1, const Vec2D &p2) const override
    {
        Vec2D d = p2 - p1;
        Vec2D f = p1 - center;

        double a = d.dot(d);
        double b = 2 * f.dot(d);
        double c = f.dot(f) - r_sqr;
        double delta = b * b - 4 * a * c;

        if (delta < 0)
        {
            return false;
        }

        if (std::abs(a) < 1e-9)
        {
            return isInside(p1);
        }

        delta = std::sqrt(delta);
        double t1 = (-b - delta) / (a * 2.0f);
        double t2 = (-b + delta) / (a * 2.0f);

        if ((t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1))
        {
            return true;
        }

        return (t1 < 0 && t2 > 1) || (t2 < 0 && t1 > 1);
    }
};

class Rectangle : public Obstacle
{
public:
    Vec2D center;
    double width, height;
    double angle, cos_a, sin_a;
    std::array<Vec2D, 2> axes;
    std::array<Vec2D, 4> vertices;

    Rectangle(double x, double y, double w, double h, double angle_deg, bool active = true) : center(x, y), width(w), height(h)
    {
        angle = angle_deg * M_PI / 180.0;
        cos_a = std::cos(angle);
        sin_a = std::sin(angle);
        this->active = active;

        double w2 = width / 2.0;
        double h2 = height / 2.0;
        Vec2D local_v[4] = {{-w2, -h2}, {w2, -h2}, {w2, h2}, {-w2, h2}};

        for (int i = 0; i < 4; ++i)
        {
            vertices[i] = local_v[i].rotate(cos_a, sin_a) + center;
        }

        axes[0] = (vertices[1] - vertices[0]);
        axes[1] = (vertices[3] - vertices[0]);
        axes[0] = axes[0] / axes[0].norm();
        axes[1] = axes[1] / axes[1].norm();
    }

    bool isInside(const Vec2D &point) const override
    {
        if (this->active == false)
            return false; // Inactive obstacles are considered non-existent
        Vec2D d = point - center;
        double local_x = d.x * cos_a + d.y * sin_a;
        double local_y = -d.x * sin_a + d.y * cos_a;
        return (std::abs(local_x) <= width / 2.0) && (std::abs(local_y) <= height / 2.0);
    }

    bool intersectSegment(const Vec2D &p1, const Vec2D &p2) const override
    {
        std::vector<Vec2D> check_axes = {axes[0], axes[1]};
        Vec2D edge = p2 - p1;
        Vec2D axis_edge(-edge.y, edge.x);

        if (axis_edge.sqrNorm() > 1e-9)
        {
            check_axes.push_back(axis_edge / axis_edge.norm());
        }

        for (const auto &axis : check_axes)
        {
            double min_r = std::numeric_limits<double>::infinity();
            double max_r = -std::numeric_limits<double>::infinity();

            for (const auto &v : vertices)
            {
                double proj = v.dot(axis);
                min_r = std::min(min_r, proj);
                max_r = std::max(max_r, proj);
            }

            double proj_p1 = p1.dot(axis);
            double proj_p2 = p2.dot(axis);
            double min_s = std::min(proj_p1, proj_p2), max_s = std::max(proj_p1, proj_p2);

            if (max_s < min_r || min_s > max_r)
            {
                return false;
            }
        }

        return true;
    }
};

#endif