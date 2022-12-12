#ifndef HIT_H
#define HIT_H

#include "common.h"
#include "vec3.h"
#include "ray.h"

class Material;

struct HitPoint {
public:
    Vec3 p, normal;
    double t;
    bool front_face; // true: 在物体外边; false: 在物体内部
    
    Material** mtl;

    __device__ __host__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#endif // HIT_H