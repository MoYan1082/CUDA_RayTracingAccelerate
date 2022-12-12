#include "include/common.h"
#include "include/vec3.h"
#include "include/ray.h"
#include "include/camera.h"
#include "include/hit.h"
#include "include/material.h"
#include "include/sphere.h"
#include "include/bvh.h"

#include <cooperative_groups.h>

// Image config
const auto aspect_ratio = 12.0 / 9.0;
const int image_width = 1200;
const int image_height = 900;
const int samples_per_pixel = 20;
const int max_depth = 50;

// Camera config
point3 lookfrom(13, 2, 3);
point3 lookat(0, 0, 0);
vec3 vup(0, 1, 0);
auto dist_to_focus = 10.0;
camera* cam_h = new camera(lookfrom, lookat, vup, 20, aspect_ratio, dist_to_focus);


std::vector<sphere*> world;

void random_scene() {
    // material order: L M D 
    lambertian* ground_material = new lambertian(color(0.5, 0.5, 0.5));
    lambertian* ground_material_d = ground_material->toDevice();
    world.push_back(new sphere(point3(0, -1000, 0), 1000, ground_material_d, nullptr, nullptr));

    dielectric* material1 = new dielectric(1.5);
    dielectric* material1_d = material1->toDevice();
    world.push_back(new sphere(point3(0, 1, 0), 1.0, nullptr, nullptr, material1_d));

    lambertian* material2 = new lambertian(color(0.4, 0.2, 0.1));
    lambertian* material2_d = material2->toDevice();
    world.push_back(new sphere(point3(-4, 1, 0), 1.0, material2_d, nullptr, nullptr));

    metal* material3 = new metal(color(0.7, 0.6, 0.5));
    metal* material3_d = material3->toDevice();
    world.push_back(new sphere(point3(4, 1, 0), 1.0, nullptr, material3_d, nullptr));

    for (int i = -4; i < 4; i++) {
        for (int j = -4; j < 4; j++) {
            double choose_mat = random_double_h();
            point3 center(i + 0.9 * random_double_h(), 0.2, j + 0.9 * random_double_h());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                if (choose_mat < 0.8) {
                    vec3 albedo = random_vec3_h() * random_vec3_h();
                    lambertian* sphere_material = new lambertian(albedo);
                    lambertian* sphere_material_d = sphere_material->toDevice();
                    world.push_back(new sphere(center, 0.2, sphere_material_d, nullptr, nullptr));
                } else if (choose_mat < 0.95) {
                    vec3 albedo = random_vec3_h(0.5, 1);
                    metal* sphere_material = new metal(albedo);
                    metal* sphere_material_d = sphere_material->toDevice();
                    world.push_back(new sphere(center, 0.2, nullptr, sphere_material_d, nullptr));
                } else {
                    dielectric* sphere_material = new dielectric(1.5);
                    dielectric* sphere_material_d = sphere_material->toDevice();
                    world.push_back(new sphere(center, 0.2, nullptr, nullptr, sphere_material_d));
                }
            }
        }
    }
}

__global__ void ray_color(bvh_node* bvh, camera* cam, vec3* red_d, curandState* d_rng_states, int turn) {
    int i = blockIdx.x; int j = blockIdx.y;

    const int seed = (i + j * image_width) * samples_per_pixel + turn;
    curand_init(seed, 0, 0, &(d_rng_states[seed]));
    
    double u = 1.0 * i / (image_width - 1);
    double v = 1.0 * j / (image_height - 1);
    ray r = cam->get_ray(u, v);

    hit_record rec;
    color attenuationAcc(1.0, 1.0, 1.0);

    int depth = max_depth;
    while (depth > 0 && bvh->hit(r, 0.001, INF, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr0 != nullptr) {
            if (!rec.mat_ptr0->scatter(r, rec, attenuation, scattered, d_rng_states[seed])) {
                red_d[i + j * image_width] += vec3(0., 0., 0.);
                return;
            }
        } else if (rec.mat_ptr1 != nullptr) {
            if (!rec.mat_ptr1->scatter(r, rec, attenuation, scattered)) {
                red_d[i + j * image_width] += vec3(0., 0., 0.);
                return;
            }
        } else if (rec.mat_ptr2 != nullptr) {
            if (!rec.mat_ptr2->scatter(r, rec, attenuation, scattered, d_rng_states[seed])) {
                red_d[i + j * image_width] += vec3(0., 0., 0.);
                return;
            }
        } else {
            assert(false);
        }
        attenuationAcc = attenuationAcc * attenuation;
        r = scattered;
        depth--;
    }

    vec3 unit_direction = r.direction / r.direction.length();
    auto t = 0.5 * (unit_direction.y() + 1.0);
    color background = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);

    red_d[i + j * image_width] += background * attenuationAcc;
}

int main() {
    double start = cpuSecond();
    freopen("figure.ppm", "w", stdout);

    int deviceCount;
    CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "There is no device." << std::endl;
        return false;
    }

    // initial device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CALL(cudaGetDeviceProperties(&deviceProp, dev));
    std::cerr << "Using device " << dev << ": " << deviceProp.name << std::endl;
    CALL(cudaSetDevice(dev));

    // initial random
    curandState* d_rng_states = nullptr;
    CALL(cudaMalloc(&d_rng_states, image_width * image_height * samples_per_pixel * sizeof(curandState)));

    // ================================================================================
    // inital data
    std::cerr << "The size of iamge: (" << image_width << "," << image_height << ")" << std::endl;

    // World
    random_scene();
    bvh_node* bvh_world = new bvh_node(world);
    bvh_node* bvh_world_d = ToDevice(bvh_world);

    // Camera
    camera* cam_d;
    CALL(cudaMalloc(&cam_d, sizeof(camera)));
    CALL(cudaMemcpy(cam_d, cam_h, sizeof(camera), cudaMemcpyHostToDevice));

    int nBytes = sizeof(vec3) * image_width * image_height;
    vec3* res_from_gpu = (vec3*)malloc(nBytes);
    vec3* res_d;
    CALL(cudaMalloc((void**)&res_d, nBytes));

    for(int i = 0; i < samples_per_pixel; i++) {
        dim3 blockDim(samples_per_pixel);
        dim3 gridDim(image_width, image_height);
        ray_color << < gridDim, blockDim >> > (bvh_world_d, cam_d, res_d, d_rng_states, i);
        cudaDeviceSynchronize();
    }

    CALL(cudaMemcpy(res_from_gpu, res_d, nBytes, cudaMemcpyDeviceToHost));
    // ================================================================================
    // Render
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            write_color(std::cout, res_from_gpu[i + j * image_width], samples_per_pixel);
        }
    }
    double finish = cpuSecond();
    int duration = (int)(finish - start);
    std::cerr << "Time: " << duration / 60 << "min " << duration << "s" << std::endl;
    return 0;
}