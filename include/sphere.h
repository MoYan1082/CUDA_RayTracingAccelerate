#ifndef SPHERE_H
#define SPHERE_H

#include "common.h"
#include "hit.h"

#include <vector>

class sphere {
public:
    point3 center;
    double radius;
    aabb box;

    // 材质
    lambertian* mat_ptr0;
    metal* mat_ptr1;
    dielectric* mat_ptr2;

    sphere(point3 cen, double r, lambertian* lam, metal* met, dielectric* die)
        :center(cen), radius(r),
        box(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius))
    {
        mat_ptr0 = lam;
        mat_ptr1 = met;
        mat_ptr2 = die;
    }

    __device__ __host__ bool hit(const ray r, double t_min, double t_max, hit_record& rec) {
        vec3 oc = r.origin - center;
        auto a = r.direction.length_squared();
        auto half_b = dot(oc, r.direction);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        rec.mat_ptr0 = mat_ptr0;
        rec.mat_ptr1 = mat_ptr1;
        rec.mat_ptr2 = mat_ptr2;

        return true;
    }
};

#endif