#ifndef VEC3_H
#define VEC3_H

#include "common.h"

class Vec3 {
public:
    double e[3];

    __device__ __host__ Vec3() : e{ 0,0,0 } {}
    __device__ __host__ Vec3(double e) : e{ e, e, e } {}
    __device__ __host__ Vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

    __device__ __host__ double x() const { return e[0]; }
    __device__ __host__ double y() const { return e[1]; }
    __device__ __host__ double z() const { return e[2]; }
    __device__ __host__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __device__ __host__ double operator[](int i) const { return e[i]; }
    __device__ __host__ double& operator[](int i) { return e[i]; }
    __device__ __host__ Vec3& operator+=(const Vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    __device__ __host__ Vec3& operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }
    __device__ __host__ Vec3& operator/=(const double t) {
        return *this *= 1 / t;
    }
    __device__ __host__ double length() const {
        return std::sqrt(length_squared());
    }
    __device__ __host__ double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
    __device__ __host__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }
};


// Vec3 Utility Functions
__device__ __host__ Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ __host__ Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ __host__ Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ __host__ Vec3 operator*(double t, const Vec3& v) {
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ __host__ Vec3 operator*(const Vec3& v, double t) {
    return t * v;
}
__device__ __host__ Vec3 operator/(Vec3 v, double t) {
    return (1 / t) * v;
}
__device__ __host__ Vec3 operator+(const Vec3& v, double t) {
    return Vec3(v[0] + t, v[1] + t, v[2] + t);
}

__device__ __host__ double dot(const Vec3& u, const Vec3& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__device__ __host__ Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ __host__ double length_squared(Vec3 u) {
    return u.length_squared();
}

__device__ __host__ double length(Vec3 u) {
    return u.length();
}

__device__ __host__ Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}

std::ostream& operator<<(std::ostream& out, const Vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

void write_color(std::ostream& out, Vec3 pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);


    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

__device__ Vec3 random_Vec3() {
    return Vec3(random_double(), random_double(), random_double());
}

__device__ Vec3 random_Vec3(double min, double max) {
    return Vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

Vec3 random_Vec3_h() {
    return Vec3(random_double_h(), random_double_h(), random_double_h());
}

Vec3 random_Vec3_h(double min, double max) {
    return Vec3(random_double_h(min, max), random_double_h(min, max), random_double_h(min, max));
}

__device__ __host__ Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ __host__ Vec3 refract(const Vec3& uv, const Vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif // #define VEC3_H