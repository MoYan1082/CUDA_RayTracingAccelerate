#ifndef HIT_H
#define HIT_H

#include "common.h"
#include "vec3.h"
#include "ray.h"
#include "aabb.h"

class lambertian;
class metal;
class dielectric;

struct hit_record {
public:
    point3 p;
    vec3 normal;
    double t;
    bool front_face;
    
    // 材质
    lambertian* mat_ptr0;
    metal* mat_ptr1;
    dielectric* mat_ptr2;

    __device__ __host__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#endif