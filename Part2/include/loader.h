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

    double scale = 5;
    for (auto& v : vertices) {
        v[0] /= maxaxis;
        v[1] /= maxaxis;
        v[2] /= maxaxis;

        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
        
        v[1] -= 6;
        v[0] -= 2.;
    }

    for (int i = 0; i < indices.size(); i += 3) {
        triangles.push_back(new Triangle(vertices[indices[i]], 
            vertices[indices[i + 1]], vertices[indices[i + 2]], material));
    }
    return triangles;
}
#endif