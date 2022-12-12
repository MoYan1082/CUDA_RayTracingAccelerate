#ifndef MATERIAL_H
#define MATERIAL_H

#include "common.h"
#include "vec3.h"
#include "ray.h"
#include "hit.h"

class lambertian {
public:
    color albedo;

    lambertian(const color& a) : albedo(a) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& d_rng_states){
        auto scatter_direction = rec.normal + random_unit_vector(d_rng_states);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

    lambertian* toDevice() {
        lambertian* mat_d;
        CALL(cudaMalloc((void**)&mat_d, sizeof(lambertian)));
        CALL(cudaMemcpy(mat_d, this, sizeof(lambertian), cudaMemcpyHostToDevice));
        return mat_d;
    }
};


class metal {
public:
    color albedo;

    metal(const color& a) : albedo(a) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered){
        vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction, rec.normal) > 0);
    }

    metal* toDevice() {
        metal* mat_d;
        CALL(cudaMalloc((void**)&mat_d, sizeof(metal)));
        CALL(cudaMemcpy(mat_d, this, sizeof(metal), cudaMemcpyHostToDevice));
        return mat_d;
    }
};


class dielectric {
public:
    double ir; // Index of Refraction
    
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& d_rng_states){
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction);
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(d_rng_states))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

    dielectric* toDevice() {
        dielectric* mat_d;
        CALL(cudaMalloc((void**)&mat_d, sizeof(dielectric)));
        CALL(cudaMemcpy(mat_d, this, sizeof(dielectric), cudaMemcpyHostToDevice));
        return mat_d;
    }

private:
    __device__ static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};
#endif