#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class Ray {
public:
    Vec3 origin, direction;

    __device__ __host__ Ray() {}
    __device__ __host__ Ray(const Vec3 _origin, const Vec3 _direction)
        : origin(_origin), direction(_direction) {}

    __device__ __host__ Vec3 at(double t) {
        return origin + t * direction;
    }
};
#endif // RAY_H