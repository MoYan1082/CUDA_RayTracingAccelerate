#ifndef BVH_H
#define BVH_H

#include "common.h"
#include "aabb.h"
#include "shape.h"

#include <algorithm>

__global__ void copy_sphere(Shape** dst, Sphere src) {
    (*dst) = new Sphere(src.center, src.radius, src.mtl);
}

__global__ void copy_triangle(Shape** dst, Triangle src) {
    (*dst) = new Triangle(src.p1, src.p2, src.p3, src.mtl);
}

class Bvh {
public:
    Aabb box;
    Shape** shape;
    Bvh *left, *right;

    Bvh() : box(), shape(nullptr), left(nullptr), right(nullptr) { }
    Bvh(std::vector<Shape*>& objects) : Bvh(objects, 0, objects.size()) { }
    Bvh(std::vector<Shape*>& objects, int l, int r) {
        assert(objects.size() > 0);
        double avx = 0, avy = 0, avz = 0; // average
        double vax = 0, vay = 0, vaz = 0; // variance
        for (int i = l; i < r; i++) {
            Aabb box_i = objects[i]->box;
            double tmp_x = (box_i.minimum[0] + box_i.maximum[0]) / 2;
            double tmp_y = (box_i.minimum[1] + box_i.maximum[1]) / 2;
            double tmp_z = (box_i.minimum[2] + box_i.maximum[2]) / 2;
            avx += tmp_x;
            avy += tmp_y;
            avz += tmp_z;
        }
        avx /= 1.0 * (r - l);
        avy /= 1.0 * (r - l);
        avz /= 1.0 * (r - l);

        for (int i = l; i < r; i++) {
            Aabb box_i = objects[i]->box;
            double tmp_x = (box_i.minimum[0] + box_i.maximum[0]) / 2;
            double tmp_y = (box_i.minimum[1] + box_i.maximum[1]) / 2;
            double tmp_z = (box_i.minimum[2] + box_i.maximum[2]) / 2;
            vax += (tmp_x - avx) * (tmp_x - avx);
            vay += (tmp_y - avy) * (tmp_y - avy);
            vaz += (tmp_z - avz) * (tmp_z - avz);
        }
        vax /= 1.0 * (r - l);
        vay /= 1.0 * (r - l);
        vaz /= 1.0 * (r - l);

        int axis = 0; // choose an axis by variance
        if (vax >= vay && vax >= vaz)
            axis = 0;
        else if (vay >= vax && vay >= vaz)
            axis = 1;
        else
            axis = 2;

        auto comparator = [&](auto a, auto b) {
            Aabb box_a = a->box;
            Aabb box_b = b->box;
            return box_a.minimum.e[axis] < box_b.minimum.e[axis];
        };

        if (r - l == 1) {
            build(objects[l], objects[l]);
        } else if (r - l == 2) {
            if (comparator(objects[l], objects[l + 1])) {
                build(objects[l], objects[l + 1]);
            } else {
                build(objects[l + 1], objects[l]);
            }
        } else {
            int mid = (l + r) / 2;
            std::nth_element(objects.begin() + l, objects.begin() + mid, objects.begin() + r, comparator);
            this->left = new Bvh(objects, l, mid);
            this->right = new Bvh(objects, mid, r);
        }

        this->box = merge_box(this->left->box, this->right->box);
    }

    __device__ bool hit(Ray r, double t_min, double t_max, HitPoint& rec) {
        const int BVH_STACK_LIMIT = 128;
        Bvh* nodes[BVH_STACK_LIMIT];
        int nodes_size = 0, ind = 0;
        nodes[nodes_size++] = this;

        bool hit_flag = false;
        while (ind < nodes_size) {
            Bvh* cur_node = nodes[ind++];
            if (cur_node->left == nullptr && cur_node->right == nullptr) {
                if ((*(cur_node->shape))->hit(r, t_min, t_max, rec)) {
                    t_max = min(t_max, rec.t);
                    hit_flag = true;
                }
                continue;
            }

            if (!cur_node->box.hit(r)) continue;

            assert(cur_node->left != nullptr);
            assert(nodes_size + 1 <= BVH_STACK_LIMIT);
            nodes[nodes_size++] = cur_node->left;
            assert(cur_node->right != nullptr);
            assert(nodes_size + 1 <= BVH_STACK_LIMIT);
            nodes[nodes_size++] = cur_node->right;
        }

        return hit_flag;
    }
private:
    void build(Shape* l_shape, Shape* r_shape) {
        this->left = new Bvh();
        this->right = new Bvh();
        this->left->box = l_shape->box;
        this->right->box = r_shape->box;

        cudaMalloc(&(this->left->shape), sizeof(Shape*));
        cudaMalloc(&(this->right->shape), sizeof(Shape*));

        Sphere* tmp_sphere;
        Triangle* tmp_triangle;
            
        if (tmp_sphere = dynamic_cast<Sphere*>(l_shape))
            copy_sphere <<<1, 1>>> (this->left->shape, *tmp_sphere);
        else if (tmp_triangle = dynamic_cast<Triangle*>(l_shape))
            copy_triangle <<<1, 1>>> (this->left->shape, *tmp_triangle);
        else
            assert(false);

        if (tmp_sphere = dynamic_cast<Sphere*>(r_shape))
            copy_sphere <<<1, 1>>> (this->right->shape, *tmp_sphere);
        else if (tmp_triangle = dynamic_cast<Triangle*>(r_shape))
            copy_triangle <<<1, 1>>> (this->right->shape, *tmp_triangle);
        else
            assert(false);
    }
};

Bvh* ToDevice(Bvh* host) {
    if (host == nullptr) return nullptr;

    Bvh* device;
    Bvh* bvh = new Bvh();
    bvh->box = host->box;

    if (host->left == nullptr && host->right == nullptr) {
        assert(host->shape != nullptr);
        bvh->shape = host->shape;
    }

    bvh->left = ToDevice(host->left);
    bvh->right = ToDevice(host->right);

    cudaMalloc(&device, sizeof(Bvh));
    cudaMemcpy(device, bvh, sizeof(Bvh), cudaMemcpyHostToDevice);
    return device;
}

#endif