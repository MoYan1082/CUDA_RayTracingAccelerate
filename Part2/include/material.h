#ifndef MATERIAL_H
#define MATERIAL_H

#include "common.h"
#include "vec3.h"
#include "ray.h"
#include "hit.h"

class Material {
public:
    __device__ virtual bool scatter(const Ray& r_in,
        const HitPoint& rec, Vec3& attenuation, Ray& scattered) {
        return false;
    }
    __device__ virtual bool emit(Vec3& emitted) {
        return false;
    }
};

class Lambertian : public Material {
public:
    Vec3 albedo;

    __device__ __host__ Lambertian(const Vec3& a) : albedo(a) {}

    __device__ virtual bool scatter(const Ray& r_in, 
        const HitPoint& rec, Vec3& attenuation, Ray& scattered) override 
    {
        auto scatter_direction = rec.normal + random_in_unit_sphere() * random_double();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class Metal : public Material {
public:
    Vec3 albedo;

    __device__ __host__ Metal(const Vec3& a) : albedo(a) {}

    __device__ bool virtual scatter(const Ray& r_in, 
        const HitPoint& rec, Vec3& attenuation, Ray& scattered) override 
    {
        Vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
        scattered = Ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction, rec.normal) > 0);
    }
};

class Dielectric : public Material {
public:
    double ir; // Index of Refraction
    
    __device__ __host__ Dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool virtual scatter(const Ray& r_in, 
        const HitPoint& rec, Vec3& attenuation, Ray& scattered) override
    {
        attenuation = Vec3(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;
        Vec3 unit_direction = unit_vector(r_in.direction);
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = Ray(rec.p, direction);
        return true;
    }

private:
    __device__ static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

class Diffuse_light : public Material {
public:
    Vec3 albedo;

    __device__ __host__ Diffuse_light(Vec3 a) : albedo(a) {}

    __device__ virtual bool emit(Vec3& emitted) override { 
        emitted = albedo;
        return true;
    }
};

#endif // MATERIAL_H