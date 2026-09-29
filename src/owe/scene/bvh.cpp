#include "owe/scene/bvh.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>

#include "owe/core/parallel.hpp"

namespace owe {

namespace {

// A primitive as the builder moves it: its box and its index. Partitioning these records in place
// keeps every pass over a node sequential in memory.
struct Prim {
    AABB box;
    uint32_t index;
    Vec3 centroid() const { return box.centroid(); }
};

struct Range {
    AABB box, cbox;  // bounds of the boxes and of their centroids
    uint32_t begin = 0, end = 0;
};

constexpr int kBins = 16;
constexpr uint32_t kSubtree = 32768;        // ranges below this size are built whole by one thread
constexpr uint32_t kParallelSplit = 1 << 20;  // ranges from this size are binned and partitioned in parallel

// Moves the records satisfying `pred` to the front: the k-th record from the left that fails it
// trades places with the k-th record from the right that passes it (the classic two-pointer
// partition). Large ranges compute the same exchanges in parallel. Returns the boundary.
template <class Pred>
uint32_t partitionPrims(std::vector<Prim>& p, uint32_t begin, uint32_t end, Pred pred) {
    const uint32_t n = end - begin;
    if (n < kParallelSplit) {
        uint32_t i = begin, j = end;
        while (true) {
            while (i != j && pred(p[i])) ++i;
            if (i == j) return i;
            --j;
            while (i != j && !pred(p[j])) --j;
            if (i == j) return i;
            std::swap(p[i], p[j]);
            ++i;
        }
    }
    std::vector<uint8_t> pass(n);
    const size_t parts = hardwareThreads();
    std::vector<uint32_t> passing(parts);
    parallelParts(n, parts, [&](size_t k, size_t b, size_t e) {
        uint32_t c = 0;
        for (size_t i = b; i < e; ++i) c += pass[i] = pred(p[begin + i]) ? 1 : 0;
        passing[k] = c;
    });
    uint32_t mid = 0;
    for (uint32_t c : passing) mid += c;
    // Failing records left of the boundary (ascending) and passing ones right of it (ascending).
    std::vector<uint32_t> fail(parts), keep(parts);
    parallelParts(n, parts, [&](size_t k, size_t b, size_t e) {
        uint32_t f = 0, t = 0;
        for (size_t i = b; i < e; ++i) {
            if (i < mid) f += !pass[i];
            else t += pass[i];
        }
        fail[k] = f;
        keep[k] = t;
    });
    std::vector<uint32_t> failAt(parts + 1, 0), keepAt(parts + 1, 0);
    for (size_t k = 0; k < parts; ++k) {
        failAt[k + 1] = failAt[k] + fail[k];
        keepAt[k + 1] = keepAt[k] + keep[k];
    }
    const uint32_t swaps = failAt[parts];
    std::vector<uint32_t> left(swaps), right(swaps);
    parallelParts(n, parts, [&](size_t k, size_t b, size_t e) {
        uint32_t f = failAt[k], t = keepAt[k];
        for (size_t i = b; i < e; ++i) {
            if (i < mid) { if (!pass[i]) left[f++] = uint32_t(i); }
            else if (pass[i]) right[t++] = uint32_t(i);
        }
    });
    parallelFor(swaps, 1 << 16, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) std::swap(p[begin + left[k]], p[begin + right[swaps - 1 - k]]);
    });
    return begin + mid;
}

void bound(const std::vector<Prim>& p, Range& r) {
    r.box = r.cbox = AABB{};
    for (uint32_t i = r.begin; i < r.end; ++i) {
        r.box.expand(p[i].box);
        r.cbox.expand(p[i].centroid());
    }
}

// Binned SAH over the widest centroid axis. Partitions p[r.begin, r.end) and returns true with
// both halves (and the axis), or false for a leaf.
bool split(std::vector<Prim>& p, const Range& r, uint32_t maxLeaf, int depth, Range& left, Range& right,
           uint32_t& axisOut) {
    const uint32_t n = r.end - r.begin;
    if (n <= maxLeaf || depth > 60) return false;
    Vec3 ext = r.cbox.extent();
    int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    if (!(ext[axis] > 0)) return false;

    AABB binBox[kBins], binCentroids[kBins];
    uint32_t binCount[kBins] = {};
    double lo = r.cbox.lo[axis], scale = kBins / ext[axis];
    auto binOf = [&](const Prim& q) { return std::min(kBins - 1, int((q.centroid()[axis] - lo) * scale)); };
    auto binRange = [&](uint32_t b, uint32_t e, AABB* box, AABB* cen, uint32_t* count) {
        for (uint32_t i = b; i < e; ++i) {
            int k = binOf(p[i]);
            count[k]++;
            box[k].expand(p[i].box);
            cen[k].expand(p[i].centroid());
        }
    };
    if (n < kParallelSplit) {
        binRange(r.begin, r.end, binBox, binCentroids, binCount);
    } else {
        // Bin bounds are minima and maxima, and counts sums: merging parts changes nothing.
        struct Bins { AABB box[kBins], cen[kBins]; uint32_t count[kBins] = {}; };
        std::vector<Bins> part(hardwareThreads());
        parallelParts(n, part.size(), [&](size_t k, size_t b, size_t e) {
            binRange(r.begin + uint32_t(b), r.begin + uint32_t(e), part[k].box, part[k].cen, part[k].count);
        });
        for (const Bins& q : part)
            for (int k = 0; k < kBins; ++k) {
                binCount[k] += q.count[k];
                binBox[k].expand(q.box[k]);
                binCentroids[k].expand(q.cen[k]);
            }
    }
    // Bounds of the bins right of each split, then a sweep from the left: unions of minima and
    // maxima, so each side's box is exact whatever the order of accumulation.
    AABB rightBox[kBins];
    uint32_t rightCount[kBins] = {};
    for (int k = kBins - 1; k >= 1; --k) {
        rightBox[k] = k + 1 < kBins ? rightBox[k + 1] : AABB{};
        rightCount[k] = (k + 1 < kBins ? rightCount[k + 1] : 0) + binCount[k];
        if (binCount[k]) rightBox[k].expand(binBox[k]);
    }
    double bestCost = Inf;
    int bestSplit = -1;
    AABB l;
    uint32_t nl = 0;
    for (int s = 1; s < kBins; ++s) {
        if (binCount[s - 1]) { l.expand(binBox[s - 1]); nl += binCount[s - 1]; }
        const uint32_t nr = rightCount[s];
        if (nl == 0 || nr == 0) continue;
        double cost = l.surfaceArea() * nl + rightBox[s].surfaceArea() * nr;
        if (cost < bestCost) { bestCost = cost; bestSplit = s; }
    }
    uint32_t mid;
    bool fromBins = false;
    if (bestSplit < 0) {
        mid = r.begin + n / 2;
        std::nth_element(p.begin() + r.begin, p.begin() + mid, p.begin() + r.end,
                         [&](const Prim& a, const Prim& b) { return a.centroid()[axis] < b.centroid()[axis]; });
    } else {
        double leafCost = r.box.surfaceArea() * n;
        if (n <= 16 && bestCost >= leafCost) return false;
        mid = partitionPrims(p, r.begin, r.end, [&](const Prim& q) { return binOf(q) < bestSplit; });
        fromBins = mid != r.begin && mid != r.end;
        if (!fromBins) mid = r.begin + n / 2;
    }
    left.begin = r.begin;
    left.end = right.begin = mid;
    right.end = r.end;
    if (fromBins) {
        // The halves are exactly the bins on either side of the split.
        left.box = left.cbox = right.box = right.cbox = AABB{};
        for (int k = 0; k < kBins; ++k) {
            if (!binCount[k]) continue;
            Range& h = k < bestSplit ? left : right;
            h.box.expand(binBox[k]);
            h.cbox.expand(binCentroids[k]);
        }
    } else {
        bound(p, left);
        bound(p, right);
    }
    axisOut = uint32_t(axis);
    return true;
}

// Depth-first subtree of r appended to `out`; child links are relative to out's first node.
void buildSubtree(std::vector<Prim>& p, const Range& r, uint32_t maxLeaf, int depth, std::vector<BvhNode>& out) {
    uint32_t idx = uint32_t(out.size());
    out.emplace_back();
    out[idx].box = r.box;
    Range left, right;
    uint32_t axis = 0;
    if (!split(p, r, maxLeaf, depth, left, right, axis)) {
        out[idx].start = r.begin;
        out[idx].count = r.end - r.begin;
        return;
    }
    out[idx].axis = axis;
    buildSubtree(p, left, maxLeaf, depth + 1, out);
    out[idx].start = uint32_t(out.size());
    buildSubtree(p, right, maxLeaf, depth + 1, out);
}

}  // namespace

// Large inputs: the upper levels are split level by level (siblings in parallel), the subtrees
// below them are built on all threads, and everything is laid out depth-first with one copy per
// node. Each split is the same function of the same records, so the hierarchy (and the primitive
// order) is identical to a sequential build.
void Bvh::build(const std::vector<AABB>& primBounds, uint32_t maxLeaf) {
    nodes_.clear();
    order_.resize(primBounds.size());
    if (primBounds.empty()) return;
    const size_t N = primBounds.size();
    std::vector<Prim> p(N);
    parallelFor(N, 1 << 16, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) p[i] = {primBounds[i], uint32_t(i)};
    });
    Range root;
    root.begin = 0;
    root.end = uint32_t(N);
    bound(p, root);

    struct Upper {
        Range range;
        int left = -1, right = -1;  // upper nodes
        int task = -1;              // or: a subtree built whole
        bool leaf = false;
        uint32_t axis = 0;
        int depth = 0;
    };
    std::vector<Upper> upper{Upper{root}};
    std::vector<int> frontier{0};
    std::vector<int> tasks;  // upper nodes whose subtrees are built whole
    while (!frontier.empty()) {
        std::vector<int> open, next;
        for (int u : frontier) {
            if (upper[size_t(u)].range.end - upper[size_t(u)].range.begin < kSubtree) {
                upper[size_t(u)].task = int(tasks.size());
                tasks.push_back(u);
            } else {
                open.push_back(u);
            }
        }
        std::vector<Range> halves(2 * open.size());
        std::vector<char> splitOk(open.size());
        parallelParts(open.size(), open.size(), [&](size_t k, size_t, size_t) {
            Upper& u = upper[size_t(open[k])];
            splitOk[k] = split(p, u.range, maxLeaf, u.depth, halves[2 * k], halves[2 * k + 1], u.axis);
        });
        for (size_t k = 0; k < open.size(); ++k) {
            const int u = open[k];
            if (!splitOk[k]) {
                upper[size_t(u)].leaf = true;
                continue;
            }
            const int depth = upper[size_t(u)].depth + 1;
            for (int side = 0; side < 2; ++side) {
                Upper child;
                child.range = halves[2 * k + size_t(side)];
                child.depth = depth;
                (side == 0 ? upper[size_t(u)].left : upper[size_t(u)].right) = int(upper.size());
                next.push_back(int(upper.size()));
                upper.push_back(child);
            }
        }
        frontier = std::move(next);
    }

    // The subtrees, largest first, on every thread.
    std::vector<std::vector<BvhNode>> built(tasks.size());
    std::vector<size_t> byCost(tasks.size());
    std::iota(byCost.begin(), byCost.end(), size_t(0));
    std::sort(byCost.begin(), byCost.end(), [&](size_t a, size_t b) {
        const Range &ra = upper[size_t(tasks[a])].range, &rb = upper[size_t(tasks[b])].range;
        return ra.end - ra.begin > rb.end - rb.begin;
    });
    std::atomic<size_t> nextTask{0};
    const size_t workers = std::min<size_t>(hardwareThreads(), tasks.size());
    parallelParts(workers, workers, [&](size_t, size_t, size_t) {
        for (size_t k; (k = nextTask.fetch_add(1)) < tasks.size();) {
            const Upper& u = upper[size_t(tasks[byCost[k]])];
            buildSubtree(p, u.range, maxLeaf, u.depth, built[byCost[k]]);
        }
    });

    size_t total = 0;
    for (const Upper& u : upper) total += u.task < 0 ? 1 : 0;
    for (const auto& b : built) total += b.size();
    nodes_.reserve(total);
    auto emit = [&](auto&& self, int ui) -> void {
        const Upper& u = upper[size_t(ui)];
        if (u.task >= 0) {
            const uint32_t base = uint32_t(nodes_.size());
            for (BvhNode node : built[size_t(u.task)]) {
                if (node.count == 0) node.start += base;
                nodes_.push_back(node);
            }
            return;
        }
        const uint32_t idx = uint32_t(nodes_.size());
        nodes_.emplace_back();
        nodes_[idx].box = u.range.box;
        if (u.leaf) {
            nodes_[idx].start = u.range.begin;
            nodes_[idx].count = u.range.end - u.range.begin;
            return;
        }
        nodes_[idx].axis = u.axis;
        self(self, u.left);
        nodes_[idx].start = uint32_t(nodes_.size());
        self(self, u.right);
    };
    emit(emit, 0);
    parallelFor(N, 1 << 16, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) order_[i] = p[i].index;
    });
}

}  // namespace owe
