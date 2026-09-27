#include "owe/scene/bvh.hpp"

#include <algorithm>
#include <numeric>

namespace owe {

void Bvh::build(const std::vector<AABB>& primBounds, uint32_t maxLeaf) {
    nodes_.clear();
    order_.resize(primBounds.size());
    std::iota(order_.begin(), order_.end(), 0u);
    if (primBounds.empty()) return;
    std::vector<Vec3> centroids(primBounds.size());
    for (size_t i = 0; i < primBounds.size(); ++i) centroids[i] = primBounds[i].centroid();
    nodes_.reserve(primBounds.size() * 2);
    buildRecursive(primBounds, centroids, 0, uint32_t(primBounds.size()), maxLeaf, 0);
}

uint32_t Bvh::buildRecursive(const std::vector<AABB>& b, const std::vector<Vec3>& c, uint32_t begin, uint32_t end,
                             uint32_t maxLeaf, int depth) {
    uint32_t idx = uint32_t(nodes_.size());
    nodes_.emplace_back();
    AABB box, cbox;
    for (uint32_t i = begin; i < end; ++i) {
        box.expand(b[order_[i]]);
        cbox.expand(c[order_[i]]);
    }
    nodes_[idx].box = box;
    uint32_t n = end - begin;
    auto makeLeaf = [&] {
        nodes_[idx].start = begin;
        nodes_[idx].count = n;
    };
    if (n <= maxLeaf || depth > 60) { makeLeaf(); return idx; }

    Vec3 ext = cbox.extent();
    int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    if (!(ext[axis] > 0)) { makeLeaf(); return idx; }

    // Binned SAH over the widest centroid axis.
    constexpr int NB = 16;
    AABB binBox[NB];
    uint32_t binCount[NB] = {};
    double lo = cbox.lo[axis], scale = NB / ext[axis];
    auto binOf = [&](uint32_t p) { return std::min(NB - 1, int((c[p][axis] - lo) * scale)); };
    for (uint32_t i = begin; i < end; ++i) {
        int k = binOf(order_[i]);
        binCount[k]++;
        binBox[k].expand(b[order_[i]]);
    }
    double bestCost = Inf;
    int bestSplit = -1;
    for (int s = 1; s < NB; ++s) {
        AABB l, r;
        uint32_t nl = 0, nr = 0;
        for (int k = 0; k < s; ++k) { if (binCount[k]) { l.expand(binBox[k]); nl += binCount[k]; } }
        for (int k = s; k < NB; ++k) { if (binCount[k]) { r.expand(binBox[k]); nr += binCount[k]; } }
        if (nl == 0 || nr == 0) continue;
        double cost = l.surfaceArea() * nl + r.surfaceArea() * nr;
        if (cost < bestCost) { bestCost = cost; bestSplit = s; }
    }
    uint32_t mid;
    if (bestSplit < 0) {
        mid = begin + n / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                         [&](uint32_t a, uint32_t bb) { return c[a][axis] < c[bb][axis]; });
    } else {
        double leafCost = box.surfaceArea() * n;
        if (n <= 16 && bestCost >= leafCost) { makeLeaf(); return idx; }
        auto it = std::partition(order_.begin() + begin, order_.begin() + end,
                                 [&](uint32_t p) { return binOf(p) < bestSplit; });
        mid = uint32_t(it - order_.begin());
        if (mid == begin || mid == end) mid = begin + n / 2;
    }
    buildRecursive(b, c, begin, mid, maxLeaf, depth + 1);
    uint32_t right = buildRecursive(b, c, mid, end, maxLeaf, depth + 1);
    nodes_[idx].start = right;
    nodes_[idx].count = 0;
    nodes_[idx].axis = uint32_t(axis);
    return idx;
}

}  // namespace owe
