// The world flattened for GPU kernels: plain arrays of 16-byte-aligned records mirrored exactly by
// shaders/scene.slang. Every struct is built from 4-component vectors of 32-bit scalars, so its
// layout is identical under std430 and scalar block layout, on every Vulkan and Metal target.
//
// Precision. Portable GPUs compute in float (Apple GPUs have no double at all), so the world is
// exported camera-relative: every position is expressed relative to the detector's origin in
// double on the host, then rounded. Rigid transforms are split so a boundary's local frame is
// reached with a single float rotation plus a translation that already includes the camera
// offset. Bounding boxes are rounded outward, so float traversal never misses geometry. The CPU
// tracer (IEEE double) remains the reference against which this backend is validated.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "owe/scene/scene.hpp"

namespace owe::gpu {

// Keep every enum and struct below in sync with shaders/scene.slang.
enum ShapeKind : uint32_t {
    kShapeSag = 0,       // conic sag (quadric solution)
    kShapeAsphere = 1,   // sag with even asphere terms (bracketed root)
    kShapePlane = 2,
    kShapeSphere = 3,
    kShapeCylinder = 4,
    kShapeRoundWall = 5,
    kShapeDome = 6,
    kShapeMesh = 7,
};
constexpr uint32_t kGpuNone = 0xFFFFFFFFu;
constexpr uint32_t kNodeInner = 0x80000000u;

struct alignas(16) GNode {  // BVH node; inner: a = right child (left is next), b = kNodeInner | axis
    float lo[3];            //           leaf:  a = first slot, b = count
    uint32_t a;
    float hi[3];
    uint32_t b;
};

struct alignas(16) GBoundary {
    float toLocal[12];  // rows [Rᵀ | t'] mapping camera-relative world → shape-local
    float toWorld[12];  // rows [R | t − camera] mapping shape-local → camera-relative world
    float boxLo[3];     // world box (camera-relative), rounded outward
    uint32_t shape;     // ShapeKind
    float boxHi[3];
    uint32_t data;      // first float4 of the shape's parameters in `shapeData`
    uint32_t front, back, optics, emission;  // emission: kGpuNone if none
    uint32_t light;     // index into lights, or kGpuNone
    uint32_t meshNodes, meshTris, meshCount;  // mesh: BLAS root in `nodes`, first triangle, triangle count
    float scale;        // size of the local coordinates reached from the camera (float error scale)
    float area;         // surface area (emitter sampling)
    uint32_t sampleable;
    uint32_t pad;
};

// Leaf order. Vertices (not edges) are rounded, so triangles sharing a vertex share its float
// value exactly. w of v0 carries the original triangle index (as bits).
struct alignas(16) GTriangle {
    float v0[4], v1[4], v2[4];
};

enum IndexKind : uint32_t { kIndexConstant = 0, kIndexCauchy = 1, kIndexSellmeier = 2, kIndexTabulated = 3, kIndexCiddor = 4 };

struct alignas(16) GMedium {
    uint32_t indexKind, relAir, coeffs, nTerms;  // coefficients at scalars[coeffs..]; tabulated: spectrum id
    uint32_t absorption, scattering, flags, pad; // spectrum ids; flags: 1 absorbs, 2 scatters
    float g, pad1, pad2, pad3;
};

enum SpectrumKind : uint32_t { kSpecConstant = 0, kSpecBlackbody = 1, kSpecTabulated = 2, kSpecRgbReflectance = 3, kSpecRgbIlluminant = 4 };

struct alignas(16) GSpectrum {
    uint32_t kind, table, count, pad;  // table: lambdas at scalars[table..], values right after
    float a, b, c, scale;
};

struct alignas(16) GOptics {
    uint32_t type, texKind, reflectance, backAbsorbs;
    uint32_t texA, texB, texC, conductorN;
    uint32_t conductorK;
    float roughness, texScale, texParam;
    float sampleAimCenter[4]; // camera-relative centre, radius
    float sampleAimNormal[4]; // normal, mixture share
};

struct alignas(16) GEmission {
    uint32_t radiance, front, back, nee;
};

struct alignas(16) GLight {
    uint32_t kind, boundary;  // kind: 0 boundary, 1 sun
    float pdf, cdf;
};

enum DetectorKind : uint32_t { kDetObserver = 0, kDetSensor = 1 };

struct alignas(16) GGlobals {
    uint32_t boundaryCount, lightCount, ambientRegion, detectorKind;
    uint32_t skyModel, skyZenith, skyHorizon, skyGround;
    float up[4];
    uint32_t hasSun, sunRadiance, sunLight, pad0;
    float sunDir[4];       // w: cos(sun angular radius)
    float sunSolidAngle, cieYIntegral, fresnelFloor, sceneScale;
    int32_t maxDepth, rrDepth, width, height;
    // Detector (camera-relative: an observer sits at the origin).
    uint32_t detRegion, flipX, flipY, hasAim;
    float detPos[4];       // observer position / sensor centre (zero by construction)
    float fwd[4], right[4], upv[4];  // observer frame; sensor: rows of its rotation
    float tanX, tanY, pupilRadius, focusDistance;  // focusDistance < 0 means infinity
    float sensorNormal[4];
    float halfX, halfY, aimRadius, focusRadius;
    float aimCenter[4];    // w: focus share
    float aimNormal[4];
    float focusCenter[4];
    float emit[4];         // particles: x: the sky's share of emitted power; yzw: scene centre (camera-relative)
    uint32_t targetBoundary;  // the sensor's own boundary (particles deposit on it), or kGpuNone
    uint32_t hasNull;         // the scene has null boundaries (shadow rays check for them)
    uint32_t pad4[2];
};

// All meshes sharing a frame, under one hierarchy (built once per scene; see gpu_scene.cpp).
struct MeshGroup {
    std::vector<uint32_t> members;  // world boundary indices
    uint32_t frame = 0;             // a member: every member has its transform
    AABB worldBox;                  // union of the members' world boxes
    double ext = 0;                 // largest local coordinate of any member (float error scale)
    bool null = false;              // null (index-matched) surfaces: kept apart so rays can skip them
    uint32_t nodeBase = 0, triBase = 0, triCount = 0;
};
struct MeshGroups {
    std::vector<MeshGroup> groups;
    std::vector<GNode> nodes;          // each group's BLAS (inner children relative to nodes[0])
    std::vector<GTriangle> triangles;  // in leaf order; v1.w names the owning boundary
};

// A flattened scene plus the facts the host needs to interpret the kernel's output.
struct GpuScene {
    GGlobals globals{};
    std::vector<GNode> nodes;          // top-level BVH (root 0), then each mesh group's BVH
    std::vector<GBoundary> boundaries; // world boundaries at their own indices, then mesh groups
    std::vector<uint32_t> topRecord;   // top-level leaf entries → boundary records
    std::shared_ptr<Bvh> top;          // the top-level hierarchy in double (rebased on camera moves)
    std::shared_ptr<const MeshGroups> groups;
    std::vector<float> shapeData;      // float4 records
    std::vector<GTriangle> triangles;
    std::vector<uint32_t> regions;     // region → medium
    std::vector<GMedium> media;
    std::vector<GSpectrum> spectra;
    std::vector<float> scalars;        // spectrum tables, index coefficients, mesh CDFs
    std::vector<GOptics> optics;
    std::vector<GEmission> emissions;
    std::vector<GLight> lights;
    Vec3 origin;                       // world position of the camera-relative origin
    double emittedPower = 0;           // sky + lights: particles have something to carry when > 0
    std::vector<std::string> notes;    // approximations made (e.g. features not yet on the GPU)
    size_t bytes() const;
};

// Flattens the scene as seen from detector `detectorIndex`. Throws if the scene uses a feature
// the GPU kernels do not implement.
GpuScene flattenScene(const Scene& scene, int detectorIndex, const RenderSettings& settings);
// Refresh only observer globals and camera-relative world bounds/transforms; meshes stay resident.
void updateObserver(GpuScene& gpuScene, const World& world, const IdealObserver& observer);

}  // namespace owe::gpu
