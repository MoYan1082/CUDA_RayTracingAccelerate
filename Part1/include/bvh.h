#ifndef BVH_H
#define BVH_H

#include "common.h"
#include "aabb.h"
#include "sphere.h"

#include <algorithm>

class bvh_node {
public:
    aabb box;
    sphere* hit_sphere;
    bvh_node* left;
    bvh_node* right;
    int obj_size;

    bvh_node() = default;
    bvh_node(std::vector<sphere*>& objects) :bvh_node(objects, 0, objects.size()) {
        obj_size = objects.size();
    }
    bvh_node(std::vector<sphere*>& objects, int l, int r) {
        assert(objects.size() > 0);
        double avx = 0, avy = 0, avz = 0; // average
        double vax = 0, vay = 0, vaz = 0; // variance
        for (int i = l; i < r; i++) {
            aabb box_i = objects[i]->box;
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
            aabb box_i = objects[i]->box;
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
            aabb box_a = a->box;
            aabb box_b = b->box;
            return box_a.minimum.e[axis] < box_b.minimum.e[axis];
        };

        if (r - l == 1) {
            this->left = new bvh_node();
            this->left->box = objects[l]->box;
            this->left->hit_sphere = objects[l];
            this->right = new bvh_node();
            this->right->box = objects[l]->box;
            this->right->hit_sphere = objects[l];
        } else if (r - l == 2) {
            if (comparator(objects[l], objects[l + 1])) {
                this->left = new bvh_node();
                this->left->box = objects[l]->box;
                this->left->hit_sphere = objects[l];
                this->right = new bvh_node();
                this->right->box = objects[l + 1]->box;
                this->right->hit_sphere = objects[l + 1];
            } else {
                this->left = new bvh_node();
                this->left->box = objects[l + 1]->box;
                this->left->hit_sphere = objects[l + 1];
                this->right = new bvh_node();
                this->right->box = objects[l]->box;
                this->right->hit_sphere = objects[l];
            }
        } else {
            int mid = (l + r) / 2;
            std::nth_element(objects.begin() + l, objects.begin() + mid, objects.begin() + r, comparator);
            this->left = new bvh_node(objects, l, mid);
            this->right = new bvh_node(objects, mid, r);
        }

        this->box = merge_box(this->left->box, this->right->box);
    }

    __device__ bool hit(ray r, double t_min, double t_max, hit_record& rec) {
        assert(obj_size <= 128);
        bvh_node* nodes[128];
        int nodes_size = 0, ind = 0;
        nodes[nodes_size++] = this;

        bool hit_flag = false;
        while (ind < nodes_size) {
            bvh_node* cur_node = nodes[ind++];
            if (cur_node->left == nullptr && cur_node->right == nullptr) {
                if (cur_node->hit_sphere->hit(r, t_min, t_max, rec)) {
                    t_max = min(t_max, rec.t);
                    hit_flag = true;
                }
                continue;
            }

            if (!cur_node->box.hit(r)) continue;

            assert(cur_node->left != nullptr);
            assert(cur_node->right != nullptr);
            nodes[nodes_size++] = cur_node->left;
            nodes[nodes_size++] = cur_node->right;
        }

        return hit_flag;
    }
};

bvh_node* ToDevice(bvh_node* host) {
    if (host == nullptr) return nullptr;

    bvh_node* device;
    bvh_node* bvh = new bvh_node();
    bvh->box = host->box;
    bvh->obj_size = host->obj_size;

    if (host->left == nullptr && host->right == nullptr) {
        assert(host->hit_sphere != nullptr);
        CALL(cudaMalloc(reinterpret_cast<void**>(&bvh->hit_sphere), sizeof(sphere)));
        CALL(cudaMemcpy(bvh->hit_sphere, host->hit_sphere, sizeof(sphere), cudaMemcpyHostToDevice));
    }
    bvh->left = ToDevice(host->left);
    bvh->right = ToDevice(host->right);

    CALL(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(bvh_node)));
    CALL(cudaMemcpy(device, bvh, sizeof(bvh_node), cudaMemcpyHostToDevice));
    return device;
}

#endif