#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"
#include "vec3.h"
#include "ray.h"

class Camera {
public:
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 x, y, z;

    Camera() = delete;
    Camera(
        Vec3   lookfrom,
        Vec3   lookat,
        Vec3   vup,
        double vfov, // vertical field-of-view in degrees
        double aspect_ratio
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        z = unit_vector(lookfrom - lookat);
        x = unit_vector(cross(vup, z));
        y = unit_vector(cross(z, x));

        origin = lookfrom;
        horizontal = viewport_width * x;
        vertical   = viewport_height * y;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - z;
    }

    __device__ Ray get_ray(double u, double v) const {
        return Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }
};

#endif