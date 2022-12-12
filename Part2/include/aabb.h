#ifndef AABB_H
#define AABB_H

#include "common.h"
#include "vec3.h"
#include "ray.h"

class Aabb {
public:
    Vec3 minimum;
    Vec3 maximum;

    __device__ __host__ Aabb() {};
    __device__ __host__ Aabb(Vec3 minn, Vec3 maxn) : minimum(minn), maximum(maxn) { }
    __device__ __host__ Aabb(const Aabb& aabb) : minimum(aabb.minimum), maximum(aabb.maximum) { }

    __device__ bool hit(const Ray r) const {
        double t_min = -INF;
        double t_max = INF;
        for (int i = 0; i < 3; i++) {
            double t0 = fmin((minimum[i] - r.origin[i]) / r.direction[i], (maximum[i] - r.origin[i]) / r.direction[i]);
            double t1 = fmax((minimum[i] - r.origin[i]) / r.direction[i], (maximum[i] - r.origin[i]) / r.direction[i]);

            if (t0 > t_min) t_min = t0;
            if (t1 < t_max) t_max = t1;
            if (t_min > t_max) return false;
        }
        return true;
    }
};

Aabb merge_box(Aabb box0, Aabb box1) {
    Vec3 smallAABB(fmin(box0.minimum.x(), box1.minimum.x()),
        fmin(box0.minimum.y(), box1.minimum.y()),
        fmin(box0.minimum.z(), box1.minimum.z()));
    Vec3 bigAABB(fmax(box0.maximum.x(), box1.maximum.x()),
        fmax(box0.maximum.y(), box1.maximum.y()),
        fmax(box0.maximum.z(), box1.maximum.z()));

    return Aabb(smallAABB, bigAABB);
}
#endif