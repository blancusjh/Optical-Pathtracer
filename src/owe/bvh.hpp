// Bounding volume hierarchy (binned SAH), used both for triangle meshes and
// for the world-level hierarchy of boundaries.
#pragma once

#include <vector>

#include "math.hpp"

namespace owe {

struct BvhNode {
    AABB box;
    uint32_t start = 0;  // leaf: first index into order; inner: right child index
    uint32_t count = 0;  // > 0 for leaves
    uint32_t axis = 0;
};

class Bvh {
public:
    void build(const std::vector<AABB>& primBounds, uint32_t maxLeaf = 4);
    bool empty() const { return nodes_.empty(); }
    AABB bounds() const { return nodes_.empty() ? AABB{} : nodes_[0].box; }
    size_t nodeCount() const { return nodes_.size(); }

    // Calls visit(primIndex, tmax&) for candidate primitives in roughly
    // front-to-back order. visit may shrink tmax to cull further work.
    template <class F>
    void traverse(const Ray& ray, double tmin, double& tmax, F&& visit) const {
        if (nodes_.empty()) return;
        Vec3 invD{1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z};
        bool neg[3] = {invD.x < 0, invD.y < 0, invD.z < 0};
        uint32_t stack[128];
        int sp = 0;
        uint32_t cur = 0;
        double tEnter;
        if (!nodes_[0].box.intersect(ray.o, invD, tmin, tmax, tEnter)) return;
        while (true) {
            const BvhNode& node = nodes_[cur];
            if (node.count > 0) {
                for (uint32_t i = 0; i < node.count; ++i) visit(order_[node.start + i], tmax);
                if (sp == 0) break;
                cur = stack[--sp];
            } else {
                uint32_t first = cur + 1, second = node.start;
                if (neg[node.axis]) std::swap(first, second);
                double t0, t1;
                bool h0 = nodes_[first].box.intersect(ray.o, invD, tmin, tmax, t0);
                bool h1 = nodes_[second].box.intersect(ray.o, invD, tmin, tmax, t1);
                if (h0 && h1) {
                    if (t1 < t0) std::swap(first, second);
                    if (sp < 127) stack[sp++] = second;
                    cur = first;
                } else if (h0) {
                    cur = first;
                } else if (h1) {
                    cur = second;
                } else {
                    if (sp == 0) break;
                    cur = stack[--sp];
                }
            }
        }
    }

private:
    uint32_t buildRecursive(const std::vector<AABB>& b, const std::vector<Vec3>& c, uint32_t begin, uint32_t end,
                            uint32_t maxLeaf, int depth);
    std::vector<BvhNode> nodes_;
    std::vector<uint32_t> order_;
};

}  // namespace owe
