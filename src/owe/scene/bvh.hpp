// Bounding volume hierarchy (binned SAH), used both for triangle meshes and
// for the world-level hierarchy of boundaries.
#pragma once

#include <vector>

#include "owe/core/math.hpp"

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

    // Primitive indices in leaf order: leaf primitives are order()[start .. start+count).
    const std::vector<uint32_t>& order() const { return order_; }
    const std::vector<BvhNode>& nodes() const { return nodes_; }

    // Calls visit(primIndex, tmax&) for candidate primitives in roughly
    // front-to-back order. visit may shrink tmax to cull further work.
    template <class F>
    void traverse(const Ray& ray, double tmin, double& tmax, F&& visit) const {
        traverseSlots(ray, tmin, tmax, [&](uint32_t slot, double& tm) { visit(order_[slot], tm); });
    }

    // As traverse, but visit receives the leaf-order slot (an index into order()), so callers can
    // keep primitive data laid out in traversal order.
    template <class F>
    void traverseSlots(const Ray& ray, double tmin, double& tmax, F&& visit) const {
        if (nodes_.empty()) return;
        const RaySlabs rs(ray);
        struct Entry {
            uint32_t node;
            double tEnter;
        } stack[128];
        int sp = 0;
        uint32_t cur = 0;
        double tEnter;
        if (!rs.hit(nodes_[0].box, tmin, tmax, tEnter)) return;
        while (true) {
            const BvhNode& node = nodes_[cur];
            bool descend = false;
            if (node.count > 0) {
                for (uint32_t i = 0; i < node.count; ++i) visit(node.start + i, tmax);
            } else {
                uint32_t first = cur + 1, second = node.start;
                if (rs.neg[node.axis]) std::swap(first, second);
                double t0, t1;
                bool h0 = rs.hit(nodes_[first].box, tmin, tmax, t0);
                bool h1 = rs.hit(nodes_[second].box, tmin, tmax, t1);
                if (h0 && h1) {
                    if (t1 < t0) { std::swap(first, second); std::swap(t0, t1); }
                    if (sp < 128) stack[sp++] = {second, t1};
                    cur = first;
                    descend = true;
                } else if (h0 || h1) {
                    cur = h0 ? first : second;
                    descend = true;
                }
            }
            if (descend) continue;
            // A deferred node whose entry lies beyond the closest hit found since is culled: this is
            // exactly what re-testing its box against the shrunken tmax would decide.
            while (sp > 0 && stack[sp - 1].tEnter > tmax) --sp;
            if (sp == 0) break;
            cur = stack[--sp].node;
        }
    }

private:
    std::vector<BvhNode> nodes_;
    std::vector<uint32_t> order_;
};

}  // namespace owe
