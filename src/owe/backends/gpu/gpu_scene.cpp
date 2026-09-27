#include "owe/backends/gpu/gpu_scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "owe/scene/geometry.hpp"

namespace owe::gpu {

namespace {

float roundDown(double x) {
    float f = float(x);
    if (double(f) > x) f = std::nextafter(f, -INFINITY);
    return f;
}
float roundUp(double x) {
    float f = float(x);
    if (double(f) < x) f = std::nextafter(f, INFINITY);
    return f;
}
float bitsAsFloat(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Box rounded outward to float, with a margin of a few float ulps of its own magnitude: a float
// ray reaching a primitive near the box face must still be inside the box.
void exportBox(const AABB& b, const Vec3& origin, float lo[3], float hi[3]) {
    for (int a = 0; a < 3; ++a) {
        double l = b.lo[a] - origin[a], h = b.hi[a] - origin[a];
        double pad = 4e-7 * (std::abs(l) + std::abs(h)) + 1e-30;
        lo[a] = roundDown(l - pad);
        hi[a] = roundUp(h + pad);
    }
}

void exportRows(const Mat3& R, const Vec3& t, float out[12]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) out[4 * i + j] = float(R.m[i][j]);
        out[4 * i + 3] = float(t[i]);
    }
}

void exportBvh(const Bvh& bvh, const Vec3& origin, uint32_t slotBase, std::vector<GNode>& out) {
    uint32_t base = uint32_t(out.size());
    for (const BvhNode& n : bvh.nodes()) {
        GNode g{};
        exportBox(n.box, origin, g.lo, g.hi);
        if (n.count > 0) {
            g.a = slotBase + n.start;
            g.b = n.count;
        } else {
            g.a = base + n.start;
            g.b = kNodeInner | n.axis;
        }
        out.push_back(g);
    }
}

void set3(float out[4], const Vec3& v, float w);

// Vertices (not edges) are rounded, so triangles sharing a vertex share its float value exactly.
// v0.w: the triangle's index in its mesh; v1.w: the boundary it belongs to (both as bits).
GTriangle exportTriangle(const Vec3& a, const Vec3& b, const Vec3& c, uint32_t prim, uint32_t owner) {
    GTriangle t{};
    const Vec3* v[3] = {&a, &b, &c};
    float* out[3] = {t.v0, t.v1, t.v2};
    for (int k = 0; k < 3; ++k) {
        out[k][0] = float(v[k]->x);
        out[k][1] = float(v[k]->y);
        out[k][2] = float(v[k]->z);
    }
    t.v0[3] = bitsAsFloat(prim);
    t.v1[3] = bitsAsFloat(owner);
    return t;
}

// A boundary's frames relative to the camera-relative origin, its box, and the float error scale
// of the local coordinates reached from the camera (ray offsets).
void exportFrame(const Boundary& b, const Vec3& origin, GBoundary& r) {
    // x_local = Rᵀ (x_world − t) = Rᵀ x_rel + Rᵀ (origin − t)
    exportRows(b.toLocal.R, b.toLocal.R * (origin - b.toWorld.t), r.toLocal);
    exportRows(b.toWorld.R, b.toWorld.t - origin, r.toWorld);
    exportBox(b.worldBox, origin, r.boxLo, r.boxHi);
    AABB lb = b.shape->bounds();
    double ext = std::max(maxAbsComponent(lb.lo), maxAbsComponent(lb.hi));
    r.scale = float(length(origin - b.toWorld.t) + (std::isfinite(ext) ? ext : 0.0));
}

float groupScale(const World& w, const MeshGroup& g, const Vec3& origin) {
    return float(length(origin - w.boundaries()[g.frame].toWorld.t) + g.ext);
}

bool sameFrame(const Transform& a, const Transform& b) {
    for (int i = 0; i < 3; ++i) {
        if (a.t[i] != b.t[i]) return false;
        for (int j = 0; j < 3; ++j)
            if (a.R.m[i][j] != b.R.m[i][j]) return false;
    }
    return true;
}

// Every mesh in one frame shares a group: one hierarchy over all their triangles. Modelled scenes
// are mostly many meshes spanning the same space (bricks, boards, carpet threads, books ... each a
// mesh "per finish"); separate hierarchies overlap completely, and a ray descended into ~20 of them
// before its hit. The top level cannot separate them; one hierarchy per frame can.
std::shared_ptr<const MeshGroups> buildMeshGroups(const World& w) {
    auto out = std::make_shared<MeshGroups>();
    const auto& bs = w.boundaries();
    for (uint32_t i = 0; i < bs.size(); ++i) {
        if (!dynamic_cast<const MeshShape*>(bs[i].shape.get())) continue;
        const bool null = w.optics()[bs[i].optics].type == SurfaceType::Null;
        MeshGroup* g = nullptr;
        for (MeshGroup& e : out->groups)
            if (e.null == null && sameFrame(bs[e.frame].toWorld, bs[i].toWorld)) g = &e;
        if (!g) {
            out->groups.push_back({});
            g = &out->groups.back();
            g->frame = i;
            g->null = null;
        }
        g->members.push_back(i);
        g->worldBox.expand(bs[i].worldBox);
        AABB lb = bs[i].shape->bounds();
        g->ext = std::max(g->ext, std::max(maxAbsComponent(lb.lo), maxAbsComponent(lb.hi)));
    }
    for (MeshGroup& g : out->groups) {
        std::vector<AABB> boxes;
        std::vector<std::pair<uint32_t, uint32_t>> refs;  // (boundary, triangle)
        for (uint32_t i : g.members) {
            const auto& m = static_cast<const MeshShape&>(*bs[i].shape);
            for (uint32_t prim = 0; prim < m.triangles().size(); ++prim) {
                const auto& t = m.triangles()[prim];
                AABB box;
                for (int k = 0; k < 3; ++k) box.expand(m.positions()[t[k]]);
                boxes.push_back(box);
                refs.push_back({i, prim});
            }
        }
        Bvh bvh;
        bvh.build(boxes, 4);
        g.nodeBase = uint32_t(out->nodes.size());
        g.triBase = uint32_t(out->triangles.size());
        g.triCount = uint32_t(refs.size());
        exportBvh(bvh, Vec3(), g.triBase, out->nodes);
        for (uint32_t slot = 0; slot < refs.size(); ++slot) {
            auto [i, prim] = refs[bvh.order()[slot]];
            const auto& m = static_cast<const MeshShape&>(*bs[i].shape);
            const auto& t = m.triangles()[prim];
            out->triangles.push_back(exportTriangle(m.positions()[t[0]], m.positions()[t[1]], m.positions()[t[2]], prim, i));
        }
    }
    return out;
}

// The groups depend only on the meshes and their frames, not on the camera or on light
// sampling: build them once per scene. An entry holds the meshes weakly, so a scene that is gone
// (or a mesh freed and another allocated in its place) never matches.
std::shared_ptr<const MeshGroups> meshGroups(const World& w) {
    struct Entry {
        std::vector<uint32_t> index;
        std::vector<std::weak_ptr<const Shape>> shapes;
        std::vector<Transform> frames;
        std::shared_ptr<const MeshGroups> groups;
    };
    static std::mutex mutex;
    static std::vector<Entry> cache;  // most recent last
    Entry key;
    for (uint32_t i = 0; i < w.boundaries().size(); ++i) {
        const Boundary& b = w.boundaries()[i];
        if (!dynamic_cast<const MeshShape*>(b.shape.get())) continue;
        key.index.push_back(i);
        key.shapes.push_back(b.shape);
        key.frames.push_back(b.toWorld);
    }
    std::lock_guard<std::mutex> lock(mutex);
    for (const Entry& e : cache) {
        if (e.index != key.index) continue;
        bool same = true;
        for (size_t k = 0; same && k < e.shapes.size(); ++k)
            same = !e.shapes[k].expired() && e.shapes[k].lock() == key.shapes[k].lock() && sameFrame(e.frames[k], key.frames[k]);
        if (same) return e.groups;
    }
    key.groups = buildMeshGroups(w);
    cache.push_back(key);
    if (cache.size() > 2) cache.erase(cache.begin());
    return key.groups;
}

class Exporter {
public:
    Exporter(const Scene& scene, GpuScene& out) : w_(scene.world), g_(out) {}

    uint32_t spectrum(const Spectrum& sp) {
        GSpectrum r{};
        r.kind = uint32_t(sp.kind());
        r.a = float(sp.paramA());
        r.b = float(sp.paramB());
        r.c = float(sp.paramC());
        r.scale = float(sp.paramScale());
        if (sp.kind() == Spectrum::Kind::Tabulated) {
            const std::vector<double>* key = sp.tableLambdas();
            r.table = kGpuNone;
            for (auto& [k, off] : tables_)
                if (k == key) r.table = off;
            if (r.table == kGpuNone) {
                r.table = uint32_t(g_.scalars.size());
                for (double l : *sp.tableLambdas()) g_.scalars.push_back(float(l));
                for (double v : *sp.tableValues()) g_.scalars.push_back(float(v));
                tables_.push_back({key, r.table});
            }
            r.count = uint32_t(key->size());
        }
        for (size_t i = 0; i < g_.spectra.size(); ++i)
            if (std::memcmp(&g_.spectra[i], &r, sizeof r) == 0) return uint32_t(i);
        g_.spectra.push_back(r);
        return uint32_t(g_.spectra.size() - 1);
    }

    void media() {
        for (const Medium& m : w_.media()) {
            GMedium r{};
            const IndexModel& ix = m.index;
            r.indexKind = uint32_t(ix.kind());
            r.relAir = ix.isRelativeToAir() ? 1 : 0;
            r.coeffs = uint32_t(g_.scalars.size());
            switch (ix.kind()) {
                case IndexModel::Kind::Constant:
                case IndexModel::Kind::Cauchy:
                    for (double v : ix.coeffA()) g_.scalars.push_back(float(v));
                    r.nTerms = uint32_t(ix.coeffA().size());
                    break;
                case IndexModel::Kind::Sellmeier:
                    for (double v : ix.coeffA()) g_.scalars.push_back(float(v));
                    for (double v : ix.coeffB()) g_.scalars.push_back(float(v));
                    r.nTerms = uint32_t(ix.coeffA().size());
                    break;
                case IndexModel::Kind::Tabulated: r.coeffs = spectrum(ix.table()); break;
                case IndexModel::Kind::Ciddor: break;
            }
            r.absorption = spectrum(m.absorption);
            r.scattering = spectrum(m.scattering);
            r.flags = (m.absorbs() ? 1u : 0u) | (m.scatters() ? 2u : 0u);
            r.g = float(m.g);
            g_.media.push_back(r);
        }
        for (const Region& r : w_.regions()) g_.regions.push_back(r.medium);
    }

    void optics(const Vec3& origin) {
        for (const SurfaceOptics& o : w_.optics()) {
            GOptics r{};
            r.type = uint32_t(o.type);
            r.texKind = uint32_t(o.texture.kind);
            r.reflectance = spectrum(o.reflectance);
            r.backAbsorbs = o.backAbsorbs ? 1 : 0;
            r.texA = spectrum(o.texture.a);
            r.texB = spectrum(o.texture.b);
            r.texC = spectrum(o.texture.c);
            r.conductorN = spectrum(o.conductor.n);
            r.conductorK = spectrum(o.conductor.k);
            r.roughness = float(o.roughness);
            r.texScale = float(o.texture.scale);
            r.texParam = float(o.texture.param);
            if (o.sampleAimShare > 0) {
                set3(r.sampleAimCenter, o.sampleAimCenter - origin, float(o.sampleAimRadius));
                set3(r.sampleAimNormal, o.sampleAimNormal, float(o.sampleAimShare));
            }
            g_.optics.push_back(r);
        }
        for (const Emission& e : w_.emissions())
            g_.emissions.push_back({spectrum(e.radiance), e.front ? 1u : 0u, e.back ? 1u : 0u, e.nee ? 1u : 0u});
    }

    void push4(double a, double b = 0, double c = 0, double d = 0) {
        g_.shapeData.insert(g_.shapeData.end(), {float(a), float(b), float(c), float(d)});
    }

    void shape(const Shape& sh, GBoundary& r, std::vector<const MeshShape*>& meshes) {
        r.data = uint32_t(g_.shapeData.size() / 4);
        if (auto* s = dynamic_cast<const SagSurface*>(&sh)) {
            r.shape = s->asphere().empty() ? kShapeSag : kShapeAsphere;
            push4(s->curvature(), s->conic(), s->rMax(), s->rMin());
            if (!s->asphere().empty()) {
                const auto& A = s->asphere();
                push4(s->zMin(), s->zMax(), double(A.size()));
                for (size_t i = 0; i < A.size(); i += 4)
                    push4(A[i], i + 1 < A.size() ? A[i + 1] : 0, i + 2 < A.size() ? A[i + 2] : 0,
                          i + 3 < A.size() ? A[i + 3] : 0);
            }
        } else if (auto* p = dynamic_cast<const PlaneShape*>(&sh)) {
            r.shape = kShapePlane;
            push4(double(int(p->aperture())), p->a(), p->b());
            push4(p->holeX(), p->holeY());
        } else if (auto* sp = dynamic_cast<const SphereShape*>(&sh)) {
            r.shape = kShapeSphere;
            push4(sp->radius());
        } else if (auto* c = dynamic_cast<const CylinderShape*>(&sh)) {
            r.shape = kShapeCylinder;
            push4(c->radius(), c->z0(), c->z1());
        } else if (auto* rw = dynamic_cast<const RoundWallShape*>(&sh)) {
            r.shape = kShapeRoundWall;
            push4(rw->radius(), rw->height(), double(rw->openings().size()));
            for (auto& o : rw->openings()) push4(o.azimuth, o.width, o.sill, o.top);
        } else if (auto* d = dynamic_cast<const DomeShape*>(&sh)) {
            r.shape = kShapeDome;
            push4(d->radius(), d->slitAzimuth(), d->slitWidth(), d->slitTop());
        } else if (auto* m = dynamic_cast<const MeshShape*>(&sh)) {
            r.shape = kShapeMesh;
            meshes.push_back(m);
            push4(0);  // x: CDF offset (bits), filled with the mesh
        } else {
            throw std::runtime_error("GPU backend: unsupported shape " + sh.describe());
        }
    }

    // An emitting mesh keeps its own copy of its triangles with an area CDF, for area sampling
    // (next-event estimation and particles); intersection always goes through its group.
    void samplingCopy(const MeshShape& m, uint32_t owner, GBoundary& r) {
        r.meshTris = uint32_t(g_.triangles.size());
        r.meshCount = uint32_t(m.triangles().size());
        const auto& pos = m.positions();
        uint32_t cdf = uint32_t(g_.scalars.size());
        double acc = 0, total = m.area();
        for (uint32_t prim = 0; prim < m.triangles().size(); ++prim) {
            const auto& t = m.triangles()[prim];
            g_.triangles.push_back(exportTriangle(pos[t[0]], pos[t[1]], pos[t[2]], prim, owner));
            acc += 0.5 * length(cross(pos[t[1]] - pos[t[0]], pos[t[2]] - pos[t[0]]));
            g_.scalars.push_back(total > 0 ? float(acc / total) : 1.0f);
        }
        if (!m.triangles().empty()) g_.scalars.back() = 1.0f;
        g_.shapeData[4 * r.data] = bitsAsFloat(cdf);
    }

    // Records: every world boundary at its own index (so lights, sensors and triangle owners name
    // boundaries directly), then one record per mesh group. The top-level hierarchy is built over
    // the non-mesh boundaries and the groups; g_.top maps its leaf entries to records.
    void boundaries(const Vec3& origin) {
        const size_t nb = w_.boundaries().size();
        if (w_.bvh().order().size() != nb) throw std::runtime_error("GPU backend: world is not built");
        groups_ = meshGroups(w_);
        g_.groups = groups_;
        std::vector<const MeshShape*> meshes(nb, nullptr);
        for (size_t i = 0; i < nb; ++i) {
            const Boundary& b = w_.boundaries()[i];
            GBoundary r{};
            exportFrame(b, origin, r);
            std::vector<const MeshShape*> unused;
            shape(*b.shape, r, unused);
            if (!unused.empty()) meshes[i] = unused.front();
            r.front = b.front;
            r.back = b.back;
            r.optics = b.optics;
            r.emission = b.emission >= 0 ? uint32_t(b.emission) : kGpuNone;
            int li = w_.lightOfBoundary(uint32_t(i));
            r.light = li >= 0 ? uint32_t(li) : kGpuNone;
            r.area = float(b.shape->area());
            r.sampleable = b.shape->canSample() ? 1 : 0;
            r.meshNodes = r.meshTris = r.meshCount = 0;
            g_.boundaries.push_back(r);
        }
        // Group triangles first (their BLAS leaves index them directly), then sampling copies.
        g_.triangles = groups_->triangles;
        for (size_t i = 0; i < nb; ++i)
            if (meshes[i] && w_.boundaries()[i].emission >= 0) samplingCopy(*meshes[i], uint32_t(i), g_.boundaries[i]);
        // Group records and the top-level entries.
        std::vector<AABB> boxes;
        g_.topRecord.clear();
        for (size_t i = 0; i < nb; ++i)
            if (!meshes[i]) {
                boxes.push_back(w_.boundaries()[i].worldBox);
                g_.topRecord.push_back(uint32_t(i));
            }
        for (size_t k = 0; k < groups_->groups.size(); ++k) {
            const MeshGroup& grp = groups_->groups[k];
            GBoundary r{};
            exportFrame(w_.boundaries()[grp.frame], origin, r);
            exportBox(grp.worldBox, origin, r.boxLo, r.boxHi);
            r.scale = groupScale(w_, grp, origin);
            r.shape = kShapeMesh;
            r.data = 0;
            r.front = r.back = r.optics = 0;
            r.emission = r.light = kGpuNone;
            r.meshTris = grp.triBase;
            r.meshCount = grp.triCount;
            boxes.push_back(grp.worldBox);
            g_.topRecord.push_back(uint32_t(g_.boundaries.size()));
            g_.boundaries.push_back(r);
        }
        g_.top = std::make_shared<Bvh>();
        g_.top->build(boxes, 2);
        // Top-level hierarchy first (root 0; leaves index g_.top order), then the groups' BLASes.
        g_.nodes.clear();
        exportBvh(*g_.top, origin, 0, g_.nodes);
        std::vector<uint32_t> ordered(g_.topRecord.size());
        for (size_t s2 = 0; s2 < ordered.size(); ++s2) ordered[s2] = g_.topRecord[g_.top->order()[s2]];
        g_.topRecord = ordered;
        const uint32_t base = uint32_t(g_.nodes.size());
        for (GNode n : groups_->nodes) {
            if (n.b & kNodeInner) n.a += base;
            g_.nodes.push_back(n);
        }
        for (size_t k = 0; k < groups_->groups.size(); ++k)
            g_.boundaries[nb + k].meshNodes = base + groups_->groups[k].nodeBase;
    }

    std::shared_ptr<const MeshGroups> groups_;

    void lights() {
        double acc = 0;
        for (size_t i = 0; i < w_.lights().size(); ++i) {
            const LightEntry& L = w_.lights()[i];
            double p = w_.lightSelectPdf(i);
            acc += p;
            GLight r{};
            r.kind = L.kind == LightEntry::Kind::Sun ? 1u : 0u;
            r.boundary = L.kind == LightEntry::Kind::Sun ? kGpuNone : L.boundary;
            r.pdf = float(p);
            r.cdf = float(acc);
            g_.lights.push_back(r);
        }
        if (!g_.lights.empty()) g_.lights.back().cdf = 1.0f;
    }

private:
    const World& w_;
    GpuScene& g_;
    std::vector<std::pair<const std::vector<double>*, uint32_t>> tables_;
};

void set3(float out[4], const Vec3& v, float w = 0) {
    out[0] = float(v.x);
    out[1] = float(v.y);
    out[2] = float(v.z);
    out[3] = w;
}

void setCentre(GGlobals& G, const Vec3& c) {
    G.emit[1] = float(c.x);
    G.emit[2] = float(c.y);
    G.emit[3] = float(c.z);
}

}  // namespace

size_t GpuScene::bytes() const {
    return sizeof(globals) + nodes.size() * sizeof(GNode) + boundaries.size() * sizeof(GBoundary) +
           shapeData.size() * 4 + triangles.size() * sizeof(GTriangle) + regions.size() * 4 +
           media.size() * sizeof(GMedium) + spectra.size() * sizeof(GSpectrum) + scalars.size() * 4 +
           optics.size() * sizeof(GOptics) + emissions.size() * sizeof(GEmission) + lights.size() * sizeof(GLight);
}

GpuScene flattenScene(const Scene& scene, int detectorIndex, const RenderSettings& settings) {
    if (detectorIndex < 0 || detectorIndex >= int(scene.detectors.size())) throw std::runtime_error("no such detector");
    const Detector& det = *scene.detectors[size_t(detectorIndex)];
    const World& w = scene.world;
    GpuScene g;
    GGlobals& G = g.globals;

    // The camera-relative origin: the observer's eye or the sensor's centre.
    if (auto* o = dynamic_cast<const IdealObserver*>(&det)) {
        g.origin = o->position;
        G.detectorKind = kDetObserver;
        set3(G.fwd, o->forward());
        set3(G.right, o->right());
        set3(G.upv, o->upVec());
        G.tanX = float(o->tanX());
        G.tanY = float(o->tanY());
        G.pupilRadius = float(o->pupilRadius);
        G.focusDistance = std::isfinite(o->focusDistance) ? float(o->focusDistance) : -1.0f;
    } else if (auto* s = dynamic_cast<const SurfaceSensor*>(&det)) {
        g.origin = s->toWorld().t;
        G.detectorKind = kDetSensor;
        const Mat3& R = s->toWorld().R;
        set3(G.fwd, Vec3(R.m[0][0], R.m[0][1], R.m[0][2]));
        set3(G.right, Vec3(R.m[1][0], R.m[1][1], R.m[1][2]));
        set3(G.upv, Vec3(R.m[2][0], R.m[2][1], R.m[2][2]));
        set3(G.sensorNormal, s->normal(), float(s->aimShare));
        G.halfX = float(s->halfX);
        G.halfY = float(s->halfY);
        G.flipX = s->flipX ? 1 : 0;
        G.flipY = s->flipY ? 1 : 0;
        G.hasAim = s->hasAim ? 1 : 0;
        set3(G.aimCenter, s->aimCenter - g.origin, float(s->focusShare));
        set3(G.aimNormal, s->aimNormal);
        set3(G.focusCenter, s->focusCenter - g.origin);
        G.aimRadius = float(s->aimRadius);
        G.focusRadius = float(s->focusRadius);
    } else {
        throw std::runtime_error("GPU backend: unsupported detector type");
    }
    set3(G.detPos, Vec3());
    G.detRegion = det.region;
    G.width = det.width;
    G.height = det.height;

    Exporter ex(scene, g);
    ex.media();
    ex.optics(g.origin);
    ex.boundaries(g.origin);
    ex.lights();

    const Environment& env = w.env;
    G.boundaryCount = uint32_t(w.boundaries().size());
    G.lightCount = uint32_t(w.lights().size());
    G.ambientRegion = w.ambientRegion();
    G.skyModel = uint32_t(env.skyModel);
    G.skyZenith = ex.spectrum(env.zenith);
    G.skyHorizon = ex.spectrum(env.horizon);
    G.skyGround = ex.spectrum(env.ground);
    set3(G.up, env.up);
    G.hasSun = env.hasSun ? 1 : 0;
    G.sunRadiance = ex.spectrum(env.sunRadiance);
    G.sunLight = env.hasSun && !w.lights().empty() ? uint32_t(w.lights().size() - 1) : kGpuNone;
    set3(G.sunDir, env.sunDir, float(std::cos(env.sunAngularRadius)));
    G.sunSolidAngle = float(env.sunSolidAngle());
    G.cieYIntegral = float(cieYIntegral());
    G.fresnelFloor = float(clampd(settings.fresnelFloor, 0.0, 0.49));
    G.sceneScale = float(w.sceneRadius());
    G.maxDepth = settings.maxDepth;
    G.rrDepth = settings.rrDepth;
    // Particles (light and hybrid integrators): emitted power shares and the disk the distant
    // sources (sky, sun) shine onto, centred on the scene.
    double sky = w.environmentPower(), lamps = w.lights().empty() ? 0.0 : w.totalLightPower();
    g.emittedPower = sky + lamps;
    G.emit[0] = g.emittedPower > 0 ? float(sky / g.emittedPower) : 0.0f;
    setCentre(G, w.sceneCenter() - g.origin);
    G.targetBoundary = det.boundary() >= 0 ? uint32_t(det.boundary()) : kGpuNone;
    G.hasNull = std::any_of(w.boundaries().begin(), w.boundaries().end(),
                            [&](const Boundary& b) { return w.optics()[b.optics].type == SurfaceType::Null; }) ? 1 : 0;

    // Kernels index these arrays unconditionally; keep every buffer non-empty.
    if (g.triangles.empty()) g.triangles.push_back({});
    if (g.lights.empty()) g.lights.push_back({});
    if (g.emissions.empty()) g.emissions.push_back({});
    if (g.scalars.empty()) g.scalars.push_back(0);
    if (g.shapeData.empty()) g.shapeData.resize(4, 0.0f);
    if (g.nodes.empty()) g.nodes.push_back({});
    if (g.boundaries.empty()) g.boundaries.push_back({});
    return g;
}

void updateObserver(GpuScene& g, const World& w, const IdealObserver& o) {
    GGlobals& G = g.globals;
    G.detectorKind = kDetObserver;
    G.detRegion = o.region;
    G.width = o.width;
    G.height = o.height;
    set3(G.fwd, o.forward());
    set3(G.right, o.right());
    set3(G.upv, o.upVec());
    G.tanX = float(o.tanX());
    G.tanY = float(o.tanY());
    G.pupilRadius = float(o.pupilRadius);
    G.focusDistance = std::isfinite(o.focusDistance) ? float(o.focusDistance) : -1.0f;
    // Rebase from the original double-precision world each time, never from previous float data.
    g.origin = o.position;
    setCentre(G, w.sceneCenter() - g.origin);
    for (size_t i = 0; i < w.optics().size(); ++i) {
        const auto& optics = w.optics()[i];
        if (optics.sampleAimShare > 0)
            set3(g.optics[i].sampleAimCenter, optics.sampleAimCenter - g.origin, float(optics.sampleAimRadius));
    }
    const size_t nb = w.boundaries().size();
    for (size_t i = 0; i < nb; ++i) exportFrame(w.boundaries()[i], g.origin, g.boundaries[i]);
    for (size_t k = 0; k < g.groups->groups.size(); ++k) {
        const MeshGroup& grp = g.groups->groups[k];
        GBoundary& r = g.boundaries[nb + k];
        exportFrame(w.boundaries()[grp.frame], g.origin, r);
        exportBox(grp.worldBox, g.origin, r.boxLo, r.boxHi);
        r.scale = groupScale(w, grp, g.origin);
    }
    for (size_t i = 0; i < g.top->nodes().size(); ++i) exportBox(g.top->nodes()[i].box, g.origin, g.nodes[i].lo, g.nodes[i].hi);
}

}  // namespace owe::gpu
