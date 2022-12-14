#include "include/common.h"
#include "include/ray.h"
#include "include/hit.h"
#include "include/bvh.h"
#include "include/vec3.h"
#include "include/camera.h"
#include "include/shape.h"
#include "include/material.h"
#include "include/loader.h"

// Image config
const auto aspect_ratio = 1.0 / 1.0;
const int image_width = 512;
const int image_height = 512;
const int samples_per_pixel = 3000;
const int max_depth = 4;

// Camera config
Vec3 lookfrom(0, 0, 35);
Vec3 lookat(0, 0, 0);
Vec3 vup(0, 1, 0);
Camera* cam_h = new Camera(lookfrom, lookat, vup, 60, aspect_ratio);

std::vector<Shape*> world;

__global__ void init_material(Material** ground_material, Material** bottom_material, Material** left_material, Material** right_material, 
    Material** material1, Material** material2, Material** material3,
    Material** light_material) 
{
    (*ground_material) = new Lambertian(Vec3(.83, .83, .83));
    (*bottom_material) = new Lambertian(Vec3(.83, .83, .83));
    (*left_material)   = new Lambertian(Vec3(.12, .45, .15));
    (*right_material)  = new Lambertian(Vec3(.65, .05, .05));
    (*material1) = new Dielectric(1.5);
    (*material2) = new Lambertian(Vec3(0.4, 0.2, 0.1));
    (*material3) = new Metal(Vec3(0.7, 0.6, 0.5));
    (*light_material) = new Diffuse_light(Vec3(15, 15, 15));
}

void init_scene() {
    Material** ground_material;
    Material** bottom_material;
    Material** left_material;
    Material** right_material;
    Material** material1;
    Material** material2;
    Material** material3;
    Material** light_material;
    cudaMalloc(&ground_material, sizeof(Material*));
    cudaMalloc(&bottom_material, sizeof(Material*));
    cudaMalloc(&left_material, sizeof(Material*));
    cudaMalloc(&right_material, sizeof(Material*));
    cudaMalloc(&material1, sizeof(Material*));
    cudaMalloc(&material2, sizeof(Material*));
    cudaMalloc(&material3, sizeof(Material*));
    cudaMalloc(&light_material, sizeof(Material*));
    init_material<<<1, 1>>> (ground_material, bottom_material, left_material, right_material, 
        material1, material2, material3,
        light_material);

    std::cerr << "loading model.." << std::endl;
    std::vector<Triangle*> triangles = LoadObj("../models/rabbit.obj", material2);
    for (auto it : triangles) world.push_back(it);
    std::cerr << "model face: " << triangles.size() << std::endl;
    
    // sphere
    world.push_back(new Sphere(Vec3(-3.6, -8, 2), 2, material1)); // Dielectric
    world.push_back(new Sphere(Vec3( 3.6, -8, 0), 2, material3)); // metal

    // cornell box
    // light
    // world.push_back(new Triangle(Vec3(2, 9.9, 2), Vec3(2, 9.9, -2), Vec3(-2, 9.9, -2), light_material));
    // world.push_back(new Triangle(Vec3(2, 9.9, 2), Vec3(-2, 9.9, -2), Vec3(-2, 9.9, 2), light_material));
    // top
    // world.push_back(new Triangle(Vec3(10, 10, 10), Vec3(10, 10, -10), Vec3(-10, 10, -10), ground_material));
    // world.push_back(new Triangle(Vec3(10, 10, 10), Vec3(-10, 10, -10), Vec3(-10, 10, 10), ground_material));
    // bottom
    world.push_back(new Triangle(Vec3(10, -10, 10), Vec3(10, -10, -10), Vec3(-10, -10, -10), bottom_material));
    world.push_back(new Triangle(Vec3(10, -10, 10), Vec3(-10, -10, -10), Vec3(-10, -10, 10), bottom_material));
    // front
    // world.push_back(new Triangle(Vec3(10, 10, 10), Vec3(10, -10, 10), Vec3(-10, -10, 10), ground_material));
    // world.push_back(new Triangle(Vec3(10, 10, 10), Vec3(-10, -10, 10), Vec3(-10, 10, 10), ground_material));
    // back
    // world.push_back(new Triangle(Vec3(10, 10, -10), Vec3(10, -10, -10), Vec3(-10, -10, -10), ground_material));
    // world.push_back(new Triangle(Vec3(10, 10, -10), Vec3(-10, -10, -10), Vec3(-10, 10, -10), ground_material));
    // left
    // world.push_back(new Triangle(Vec3(-10, 10, 10), Vec3(-10, 10, -10), Vec3(-10, -10, -10), left_material));
    // world.push_back(new Triangle(Vec3(-10, 10, 10), Vec3(-10, -10, -10), Vec3(-10, -10, 10), left_material));
    // right
    // world.push_back(new Triangle(Vec3(10, 10, 10), Vec3(10, 10, -10), Vec3(10, -10, -10), right_material));
    // world.push_back(new Triangle(Vec3(10, 10, 10), Vec3(10, -10, -10), Vec3(10, -10, 10), right_material));
}

__global__ void init_random() {
    for (int seed = 0; seed < image_height * image_width; seed++) 
        curand_init(seed, 0, 0, &(d_rng_states[seed]));
}

__device__ void get_sphere_uv(const Vec3 p, double& u, double& v) {
    double theta = acos(-p.y());
    double phi = atan2(-p.z(), p.x()) + PI;

    u = phi / (2 * PI);
    v = theta / PI;
}

__global__ void render(Bvh* bvh, Texture* hdr_texture_d, Camera* cam, Vec3* red_d) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x, j = blockIdx.y;
    Vec3 color_res(0, 0, 0);
    for (int turn = 0; turn <= samples_per_pixel; turn++) {
        double u = (i + random_double()) / ((gridDim.x * blockDim.x) - 1);
        double v = (j + random_double()) / (gridDim.y - 1);

        Ray r = cam->get_ray(u, v);
        HitPoint hitPoint;
        Vec3 attenuationAcc(1.0, 1.0, 1.0);
        Vec3 emitted(0, 0, 0);

        int depth = max_depth;
        bool flag = true;
        while (depth > 0) {
            if(!bvh->hit(r, 0.001, INF, hitPoint)) {
                double u, v;
                get_sphere_uv(unit_vector(r.direction), u, v);
                emitted = hdr_texture_d->value(u, v);
                break;
            }
            if ((*(hitPoint.mtl))->emit(emitted)) break;
            Ray scattered;
            Vec3 attenuation;
            if (!(*(hitPoint.mtl))->scatter(r, hitPoint, attenuation, scattered)) {
                color_res += Vec3(1, 1, 1);
                flag = false;
                break;
            }
            attenuationAcc = attenuationAcc * attenuation;
            r = scattered;
            depth--;
        }
        if (flag) color_res += emitted * attenuationAcc;
    }
    red_d[i * gridDim.y + j] = color_res;
}

int main() {
    std::cerr << GRN << "preparing..." << RESET << std::endl;
    double start = cpuSecond();
    freopen("figure.ppm", "w", stdout);

    // initial device
    int deviceCount;
    CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "There is no device." << std::endl;
        return false;
    }
    int dev = 0;
    cudaDeviceProp deviceProp;
    CALL(cudaGetDeviceProperties(&deviceProp, dev));
    std::cerr << "Using device " << dev << ": " << deviceProp.name << std::endl;
    CALL(cudaSetDevice(dev));

    int thread_num = 64;
    dim3 gridDim(image_width/thread_num, image_height);
    dim3 blockDim(1*thread_num);

    cudaMalloc(&d_rng_states, image_height * image_width * sizeof(curandState));
    init_random <<<1, 1>>> ();
    cudaDeviceSynchronize();

    // ================================================================================
    // inital data
    std::cerr << "The size of image: (" << image_width << "," << image_height << ")" << std::endl;

    // World
    init_scene();
    Bvh* bvh_world = new Bvh(world);
    Bvh* bvh_world_d = ToDevice(bvh_world);

    // Camera
    Camera* cam_d;
    cudaMalloc(&cam_d, sizeof(Camera));
    cudaMemcpy(cam_d, cam_h, sizeof(Camera), cudaMemcpyHostToDevice);

    // hdr
    Texture* hdr_texture_h = new Texture("../resources/BG2.jpg");
    Texture* hdr_texture_d = ToDevice(hdr_texture_h);
    
    // image data
    int nBytes = sizeof(Vec3) * image_width * image_height;
    Vec3* res_from_gpu = (Vec3*)malloc(nBytes);
    Vec3* res_d;
    cudaMalloc(&res_d, nBytes);

    // ================================================================================
    std::cerr << GRN << "parallel rendering..." << RESET << std::endl;
    render <<<gridDim, blockDim>>> (bvh_world_d, hdr_texture_d, cam_d, res_d);
    cudaDeviceSynchronize();
    cudaMemcpy(res_from_gpu, res_d, nBytes, cudaMemcpyDeviceToHost);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            write_color(std::cout, res_from_gpu[i * image_height + j], samples_per_pixel);
        }
    }

    double finish = cpuSecond();
    int duration = (int)(finish - start);
    std::cerr << GRN << "Time: " << duration / 60 << "min " << duration % 60 << "s" << RESET << std::endl;
    return 0;
}