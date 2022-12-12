#ifndef SPHERE_H
#define SPHERE_H

#include "common.h"
#include "vec3.h"
#include "ray.h"
#include "hit.h"
#include "aabb.h"
#include "material.h"


class Shape {
public:
    Aabb box;
    Material** mtl;

    __device__ virtual bool hit(Ray r_in, double t_min, double t_max, HitPoint& res) = 0;
};


class Triangle : public Shape {
public:
    Vec3 p1, p2, p3;

    Triangle() = delete;
    __device__ __host__ Triangle(Vec3 _p1, Vec3 _p2, Vec3 _p3, Material** _mtl)
        : p1(_p1), p2(_p2), p3(_p3) 
    {
        mtl = _mtl;
        Vec3 maxn(max(_p1[0], max(_p2[0], _p3[0])), max(_p1[1], max(_p2[1], _p3[1])), max(_p1[2], max(_p2[2], _p3[2])));
        Vec3 minn(min(_p1[0], min(_p2[0], _p3[0])), min(_p1[1], min(_p2[1], _p3[1])), min(_p1[2], min(_p2[2], _p3[2])));
        box = Aabb(minn, maxn);
    }

    __device__ virtual bool hit(Ray r_in, double t_min, double t_max, HitPoint& res) override {
        Vec3 normal = unit_vector(cross(p2 - p1, p3 - p1));
        if (fabs(dot(normal, r_in.direction)) < EPS) // 射线和三角形平行
            return false;
        if (dot(normal, r_in.direction) > 0)
            normal = -normal;

        double t = (dot(normal, p1) - dot(r_in.origin, normal)) / dot(r_in.direction, normal);
        if (t <= t_min || t >= t_max) return false;


        Vec3 P = r_in.origin + r_in.direction * t; // 交点
        double tmp1 = dot(cross(p2 - p1, P - p1), normal); // 判断交点是否在三角形中
        double tmp2 = dot(cross(p3 - p2, P - p2), normal);
        double tmp3 = dot(cross(p1 - p3, P - p3), normal);
        
        if (tmp1 > 0 && (tmp2 < 0 || tmp3 < 0)) return false;
        if (tmp1 < 0 && (tmp2 > 0 || tmp3 > 0)) return false;

        res.p = P;
        res.t = t;
        res.mtl = mtl;
        res.normal = normal;

        return true;
    };
};

class Sphere : public Shape {
public:
    Vec3 center;
    double radius;
    
    Sphere() = delete;
    __device__ __host__ Sphere(Vec3 cen, double r, Material** _mtl)
        : center(cen), radius(r)
    {
        mtl = _mtl;
        box = Aabb(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
    }

    __device__ virtual bool hit(Ray r_in, double t_min, double t_max, HitPoint& res) override {
        Vec3 oc = r_in.origin - center;
        auto a = r_in.direction.length_squared();
        auto half_b = dot(oc, r_in.direction);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root <= t_min || root >= t_max) {
            root = (-half_b + sqrtd) / a;
            if (root <= t_min || root >= t_max)
                return false;
        }

        res.t = root;
        res.p = r_in.at(root);
        res.mtl = mtl;

        Vec3 outward_normal = (res.p - center) / radius;
        res.set_face_normal(r_in, outward_normal);

        return true;
    }
};

#endif // SPHERE_H