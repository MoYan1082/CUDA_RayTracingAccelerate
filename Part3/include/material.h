#ifndef MATERIAL_H
#define MATERIAL_H

#include "common.h"
#include "vec3.h"
#include "ray.h"
#include "hit.h"

#include <cmath>

__device__ Vec3 toNormalHemisphere(Vec3 v, Vec3 N);
__device__ Vec3 SampleCosineHemisphere();

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
        Vec3 scatter_direction = unit_vector(rec.normal) + toNormalHemisphere(SampleCosineHemisphere(), rec.normal);
        if (scatter_direction.near_zero()) scatter_direction = rec.normal;

        double pdf = dot(unit_vector(scatter_direction), rec.normal) / PI;

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

class Disney_brdf : public Material {
public:
    Vec3 baseColor = Vec3(0.82, 0.67, 0.16);
    float metallic = 0; // [0, 1]
    float subsurface = 0; // [0,1]
    float specular = 5; // [0, 1]
    float roughness = 5.; // [0, 1]
    float specularTint = 0; // [0, 1]
    float anisotropic = 0; // [0, 1]
    float sheen = 0; // [0, 1]
    float sheenTint = .5; // [0, 1]
    float clearcoat = 0; // [0, 1]
    float clearcoatGloss = 1; // [0, 1]


    __device__ __host__ Disney_brdf() {
        roughness = 0.2;
        specular = 1;
        subsurface = 1;
        baseColor = Vec3(150. / 255, 200. / 255, 150. / 255);
    }

    __device__ virtual bool scatter(const Ray& r_in,
        const HitPoint& rec, Vec3& attenuation, Ray& scattered) override
    {
        Vec3 V = -r_in.direction;
        Vec3 N = rec.normal;
        Vec3 L = toNormalHemisphere(SampleCosineHemisphere(), rec.normal);   // 随机出射方向 wi
        float pdf = 1.0 / (2.0 * PI);                                   // 半球均匀采样概率密度
        float cosine_o = max(0., dot(V, N));                             // 入射光和法线夹角余弦
        float cosine_i = max(0., dot(L, N));                             // 出射光和法线夹角余弦
        Vec3 tangent, bitangent;
        {
            Vec3 helper = Vec3(1, 0, 0);
            if (abs(N[0]) > 0.999) helper = Vec3(0, 0, 1);
            bitangent = normalize(cross(N, helper));
            tangent = normalize(cross(N, bitangent));
        }
        Vec3 f_r = BRDF(L, V, N, tangent, bitangent);

        scattered = Ray(rec.p, L);
        attenuation = f_r * cosine_i / pdf;

        return true;
    }

private:
    // 来源: https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
    // L 是反弹方向，V 是入射方向的负方向，N 是表面法线，H 是半角向量
    __device__ __host__ Vec3 BRDF(Vec3 L, Vec3 V, Vec3 N, Vec3 X, Vec3 Y) 
    {
        float NdotL = dot(N, L);
        float NdotV = dot(N, V);
        if (NdotL < 0 || NdotV < 0) return Vec3(0);

        Vec3 H = normalize(L + V);
        float NdotH = dot(N, H);
        float LdotH = dot(L, H);

        Vec3 Cdlin = mon2lin(baseColor);
        float Cdlum = .3 * Cdlin[0] + .6 * Cdlin[1] + .1 * Cdlin[2]; // luminance approx.

        Vec3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : Vec3(1); // normalize lum. to isolate hue+sat
        Vec3 Cspec0 = mix(specular * .08 * mix(Vec3(1), Ctint, specularTint), Cdlin, metallic);
        Vec3 Csheen = mix(Vec3(1), Ctint, sheenTint);

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
        // and mix in diffuse retro-reflection based on roughness
        float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
        float Fd90 = 0.5 + 2 * LdotH * LdotH * roughness;
        float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

        // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
        // 1.25 scale is used to (roughly) preserve albedo
        // Fss90 used to "flatten" retroreflection based on roughness
        float Fss90 = LdotH * LdotH * roughness;
        float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
        float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);

        // specular
        float aspect = sqrt(1 - anisotropic * .9);
        float ax = max(.001, sqr(roughness) / aspect);
        float ay = max(.001, sqr(roughness) * aspect);
        float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
        float FH = SchlickFresnel(LdotH);
        Vec3 Fs = mix(Cspec0, Vec3(1), FH);
        float Gs;
        Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
        Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

        // sheen
        Vec3 Fsheen = FH * sheen * Csheen;

        // clearcoat (ior = 1.5 -> F0 = 0.04)
        float Dr = GTR1(NdotH, mix(.1, .001, clearcoatGloss));
        float Fr = mix(.04, 1.0, FH);
        float Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25);

        return ((1 / PI) * mix(Fd, ss, subsurface) * Cdlin + Fsheen)
            * (1 - metallic)
            + Gs * Fs * Ds + .25 * clearcoat * Gr * Fr * Dr;
    }

    __device__ __host__ float sqr(float x) { return x * x; }

    __device__ __host__ float SchlickFresnel(float u)
    {
        float m = clamp(1 - u, 0, 1);
        float m2 = m * m;
        return m2 * m2 * m; // pow(m,5)
    }

    __device__ __host__ float GTR1(float NdotH, float a)
    {
        if (a >= 1) return 1 / PI;
        float a2 = a * a;
        float t = 1 + (a2 - 1) * NdotH * NdotH;
        return (a2 - 1) / (PI * log(a2) * t);
    }

    __device__ __host__ float GTR2(float NdotH, float a)
    {
        float a2 = a * a;
        float t = 1 + (a2 - 1) * NdotH * NdotH;
        return a2 / (PI * t * t);
    }

    __device__ __host__ float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
    {
        return 1 / (PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
    }

    __device__ __host__ float smithG_GGX(float NdotV, float alphaG)
    {
        float a = alphaG * alphaG;
        float b = NdotV * NdotV;
        return 1 / (NdotV + sqrt(a + b - a * b));
    }

    __device__ __host__ float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
    {
        return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
    }

    __device__ __host__ Vec3 mon2lin(Vec3 x)
    {
        return Vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
    }

    __device__ __host__ Vec3 normalize(Vec3 x)
    {
        return unit_vector(x);
    }
    
    __device__ __host__ Vec3 mix(Vec3 x, Vec3 y, float a)
    {
        a = clamp(a, 0, 1);
        return x * (1 - a) + y * a;
    }
    
    __device__ __host__ double mix(double x, double y, float a)
    {
        a = clamp(a, 0, 1);
        return x * (1 - a) + y * a;
    }
};


__device__ Vec3 SampleCosineHemisphere() {
    // Malley’s Method
    double xi_1 = random_double(), xi_2 = random_double();
    double r = std::sqrt(xi_1);
    double theta = xi_2 * 2.0 * PI;
    double x = r * std::cos(theta);
    double y = r * std::sin(theta);
    double z = std::sqrt(1.0 - x * x - y * y);

    return Vec3(x, y, z);
}
__device__ Vec3 toNormalHemisphere(Vec3 v, Vec3 N) {
    // 将向量 v 投影到 N 的法向半球
    Vec3 helper = Vec3(1, 0, 0);
    if (std::abs(N[0]) > 0.999)
        return v;
    Vec3 tangent = unit_vector(cross(N, helper));
    Vec3 bitangent = unit_vector(cross(N, tangent));
    return v[0] * tangent + v[1] * bitangent + v[2] * N;
}

#endif // MATERIAL_H