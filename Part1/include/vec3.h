#ifndef VEC3_H
#define VEC3_H

#include "common.h"

class vec3 {
public:
    double e[3];

    __device__ __host__ vec3() : e{ 0,0,0 } {}
    __device__ __host__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

    __device__ __host__ double x() const { return e[0]; }
    __device__ __host__ double y() const { return e[1]; }
    __device__ __host__ double z() const { return e[2]; }
    __device__ __host__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __device__ __host__ double operator[](int i) const { return e[i]; }
    __device__ __host__ double& operator[](int i) { return e[i]; }

    __device__ __host__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __device__ __host__ vec3& operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ __host__ vec3& operator/=(const double t) {
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


// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color


// vec3 Utility Functions
__device__ __host__ vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ __host__ vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ __host__ vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ __host__ vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ __host__ vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__device__ __host__ vec3 operator/(vec3 v, double t) {
    return (1 / t) * v;
}

__device__ __host__ double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__device__ __host__ vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ __host__ vec3 unit_vector(vec3 v) {
    return v / v.length();
}

std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

void write_color(std::ostream& out, color pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    r /= samples_per_pixel;
    g /= samples_per_pixel;
    b /= samples_per_pixel;

    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

__device__ vec3 random_vec3(curandState& d_rng_states) {
    return vec3(random_double(d_rng_states), random_double(d_rng_states), random_double(d_rng_states));
}

__device__ vec3 random_vec3(double min, double max, curandState& d_rng_states) {
    return vec3(random_double(min, max, d_rng_states), 
                random_double(min, max, d_rng_states), 
                random_double(min, max,d_rng_states));
}

__device__ vec3 random_in_unit_disk(curandState& d_rng_states) {
    while (true) {
        auto p = vec3(random_double(-1, 1, d_rng_states), random_double(-1, 1, d_rng_states), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__device__ vec3 random_in_unit_sphere(curandState& d_rng_states) {
    while (true) {
        auto p = random_vec3(d_rng_states);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

vec3 random_vec3_h() {
    return vec3(random_double_h(), random_double_h(), random_double_h());
}

vec3 random_vec3_h(double min, double max) {
    return vec3(random_double_h(min, max), random_double_h(min, max), random_double_h(min, max));
}

__device__ vec3 random_unit_vector(curandState& d_rng_states) {
    return unit_vector(random_in_unit_sphere(d_rng_states));
}

__device__ __host__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ __host__ vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif