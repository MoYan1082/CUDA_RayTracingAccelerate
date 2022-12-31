#ifndef LOADER_H
#define LOADER_H

#include "common.h"
#include "vec3.h"
#include "shape.h"
#include "material.h"

#include <vector>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"

std::vector<Triangle*> LoadObj(std::string filepath, Material** material) {
    std::vector<Triangle*> triangles;
    std::vector<Vec3> vertices;
    std::vector<unsigned int> indices;

    double maxxd = -1e18;
    double maxyd = -1e18;
    double maxzd = -1e18;
    double minxd = 1e18;
    double minyd = 1e18;
    double minzd = 1e18;

    std::ifstream fin(filepath);
    if (!fin.is_open()) {
        std::cerr << filepath << " cannot be opened" << std::endl;
        exit(-1);
    }
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream sin(line);
        std::string type;

        int slashCnt = 0;
        for (int i = 0; i < line.length(); i++)
            if (line[i] == '/') slashCnt++;
        if (slashCnt > 0) assert(false);

        sin >> type;
        if (type == "v") {
            double x, y, z;
            sin >> x >> y >> z;
            vertices.push_back(Vec3(x, y, z));
            maxxd = max(maxxd, x); maxyd = max(maxxd, y); maxzd = max(maxxd, z);
            minxd = min(minxd, x); minyd = min(minxd, y); minzd = min(minxd, z);
        }
        else if (type == "f") {
            unsigned int v0, v1, v2;
            sin >> v0 >> v1 >> v2;
            indices.push_back(v0 - 1);
            indices.push_back(v1 - 1);
            indices.push_back(v2 - 1);
        }
        else if (type == "#") {
            // 
        }
        else {
            assert(false);
        }
    }
    
    // normalization
    double lenx = maxxd - minxd;
    double leny = maxyd - minyd;
    double lenz = maxzd - minzd;
    double maxaxis = max(lenx, max(leny, lenz));

    double scale = 10;
    for (auto& v : vertices) {
        v[0] /= maxaxis;
        v[1] /= maxaxis;
        v[2] /= maxaxis;

        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
        
        v[0] += 1;
        v[1] -= 7;
        v[2] -= 2;
    }

    for (int i = 0; i < indices.size(); i += 3) {
        triangles.push_back(new Triangle(vertices[indices[i]], 
            vertices[indices[i + 1]], vertices[indices[i + 2]], material));
    }
    return triangles;
}

class Texture {
public:
    const static int bytes_per_pixel = 3;
    unsigned char* data;
    int width, height;
    int bytes_per_scanline;

    Texture()
        : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

    Texture(const char* filename) {
        auto components_per_pixel = bytes_per_pixel;

        std::cerr << "loading texture image file '" << filename << "'." << std::endl;
        data = stbi_load(
            filename, &width, &height, &components_per_pixel, components_per_pixel);

        if (!data) {
            std::cerr << RED << "ERROR: Could not load texture image file '" << filename << "'." << RESET << std::endl;
            exit(-1);
        }

        bytes_per_scanline = bytes_per_pixel * width;
    }

    __device__ __host__ Vec3 value(double u, double v) {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (data == nullptr)
            return Vec3(0, 1, 1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = clamp(u, 0.0, 1.0);
        v = 1.0 - clamp(v, 0.0, 1.0);  // Flip V to image coordinates

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);

        // Clamp integer mapping, since actual coordinates should be less than 1.0
        if (i >= width)  i = width - 1;
        if (j >= height) j = height - 1;

        const auto color_scale = 1.0 / 255.0;
        auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

        return Vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

};

Texture* ToDevice(Texture* host) {
    if (host == nullptr) return nullptr;

    Texture* device;

    Texture* texture = new Texture();
    texture->width = host->width;
    texture->height = host->height;
    texture->bytes_per_scanline = host->bytes_per_scanline;

    int nBytes = host->bytes_per_pixel * host->width * host->height * sizeof(unsigned char);
    cudaMalloc(&(texture->data), nBytes);
    cudaMemcpy(texture->data, host->data, nBytes, cudaMemcpyHostToDevice);

    cudaMalloc(&device, sizeof(Texture));
    cudaMemcpy(device, texture, sizeof(Texture), cudaMemcpyHostToDevice);
    return device;
}

#endif