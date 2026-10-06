// Scene buffers. Mirrors src/owe/backends/gpu/gpu_scene.hpp exactly: every struct is built from
// 4-component vectors of 32-bit scalars, so std430 and scalar layouts agree on every target.
#pragma once

static const uint kShapeSag = 0;
static const uint kShapeAsphere = 1;
static const uint kShapePlane = 2;
static const uint kShapeSphere = 3;
static const uint kShapeCylinder = 4;
static const uint kShapeRoundWall = 5;
static const uint kShapeDome = 6;
static const uint kShapeMesh = 7;
static const uint kShapeWaves = 8;
static const uint kShapeCone = 9;
static const uint kShapeTorus = 10;

static const uint kGpuNone = 0xFFFFFFFFu;
// A triangle's corner words (GTriangle.v2.w): three normals, then three texture coordinates.
static const uint kCornerNormals = 0x80000000u, kCornerUvs = 0x40000000u, kCornerIndex = 0x3FFFFFFFu;
static const uint kNodeInner = 0x80000000u;

// SurfaceType
static const uint kSurfNull = 0;
static const uint kSurfDielectric = 1;
static const uint kSurfDiffuse = 2;
static const uint kSurfConductor = 3;
static const uint kSurfMirror = 4;
static const uint kSurfAbsorber = 5;
static const uint kSurfDetector = 6;
static const uint kSurfStainedGlass = 7;

struct GNode {
    float4 lo;  // w: a (bits)
    float4 hi;  // w: b (bits)
};

struct GBoundary {
    float4 toLocal[3];
    float4 toWorld[3];
    float4 boxLo;   // w: shape kind (bits)
    float4 boxHi;   // w: shape data offset (bits)
    uint4 links;    // front, back, optics, emission
    uint4 mesh;     // light, meshNodes, meshTris, meshCount
    float4 misc;    // scale, area, sampleable (bits), pad
};

struct GTriangle {
    float4 v0;  // w: original index (bits)
    float4 v1;  // w: the owning boundary (bits)
    float4 v2;  // w: first corner word in g_normals, with kCorner flags (bits; kGpuNone: flat, untextured)
};

struct GMedium {
    uint4 index;  // kind, relAir, coeffs, nTerms
    uint4 props;  // absorption, scattering, flags, pad
    float4 g;
};

struct GSpectrum {
    uint4 info;    // kind, table, count, pad
    float4 abcs;   // a, b, c, scale
};

struct GOptics {
    uint4 a;  // type, texKind, reflectance, backAbsorbs
    uint4 b;  // texA, texB, texC, conductorN
    uint4 c;  // conductorK, roughness, texScale, texParam (floats as bits)
    float4 sampleAimCenter; // camera-relative centre, radius
    float4 sampleAimNormal; // normal, mixture share
    float4 roughMap;        // material map: roughness of texture components a, b, c; w: 1 if mapped
    float4 relief;          // relief: scale, depth, octaves (depth 0: none)
};

struct GEmission {
    uint radiance, front, back, nee;
};

struct GLight {
    uint kind, boundary;
    float pdf, cdf;
};

struct GGlobals {
    uint4 counts;        // boundaryCount, lightCount, ambientRegion, detectorKind
    uint4 sky;           // model, zenith, horizon, ground
    float4 up;
    uint4 sun;           // hasSun, radiance, sunLight, pad
    float4 sunDir;       // w: cos(angular radius)
    float4 misc;         // sunSolidAngle, cieYIntegral, fresnelFloor, sceneScale
    int4 limits;         // maxDepth, rrDepth, width, height
    uint4 det;           // region, flipX, flipY, hasAim
    float4 detPos;
    float4 fwd, right, upv;
    float4 lens;         // tanX, tanY, pupilRadius, focusDistance (< 0: infinity)
    float4 sensorNormal;
    float4 sensorSize;   // halfX, halfY, aimRadius, focusRadius
    float4 aimCenter;    // w: focus share
    float4 aimNormal;
    float4 focusCenter;
    float4 emit;         // particles: x: the sky's share of emitted power; yzw: scene centre (camera-relative)
    uint4 target;        // x: the sensor's own boundary (particles deposit on it), or kGpuNone;
                         // y: the scene has null boundaries; z: pixel filter sigma (float bits);
                         // w: share of pupil samples from the guide (float bits)
    float4 guide;        // observer: pupil guide centre (camera-relative), radius (0: none)
    uint4 envMap;        // environment map: image record, marginal CDF, conditional CDFs (scalars), light
    float4 envParams;    // rotation, scale, distribution total, RGB illuminant norm
    float4 envE1;        // the map's frame: e1 | width
    float4 envE2;        // e2 | height
    uint4 envMore;       // x: row weights (scalars)
    float4 rgbA[3];      // RGB → basis weights (rows)
};

// Per-pixel state carried across dispatches of one pass.
struct PixelState {
    uint4 rng;    // PCG32 state (lo, hi), increment (lo, hi)
    float4 acc;   // Σ X, Σ Y, Σ Z over this pass; w: hero-wavelength rotation
    float4 comp;  // Kahan compensation of acc
    uint4 stats;  // paths, segments, region inconsistencies, leaks
};

static const uint kFlagPartition = 1u;  // hybrid: light tracing owns eye → D → S⁺ → light paths
// VCM/BDPT with scattering media: the path tracer owns the paths with a medium scattering vertex
// (VCM, which treats media as extinction only, takes all the others).
static const uint kFlagMediumOnly = 2u;

struct PushConstants {
    uint pass, spp, sampleBase, sampleCount;  // particles: sampleBase is the low word of the first index
    uint seedLo, seedHi, width, height;
    uint flags, baseHi, pad0, pad1;
    uint4 extra;  // integrator-specific (SPPM: wavelength rotation, alpha, radius scale, pixel angle)
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<GGlobals> g_globals;
[[vk::binding(1, 0)]] StructuredBuffer<GNode> g_nodes;
[[vk::binding(2, 0)]] StructuredBuffer<GBoundary> g_boundaries;
[[vk::binding(3, 0)]] StructuredBuffer<float4> g_shapeData;
[[vk::binding(4, 0)]] StructuredBuffer<GTriangle> g_triangles;
[[vk::binding(5, 0)]] StructuredBuffer<uint> g_regions;
[[vk::binding(6, 0)]] StructuredBuffer<GMedium> g_media;
[[vk::binding(7, 0)]] StructuredBuffer<GSpectrum> g_spectra;
[[vk::binding(8, 0)]] StructuredBuffer<float> g_scalars;
[[vk::binding(9, 0)]] StructuredBuffer<GOptics> g_optics;
[[vk::binding(10, 0)]] StructuredBuffer<GEmission> g_emissions;
[[vk::binding(11, 0)]] StructuredBuffer<GLight> g_lights;
[[vk::binding(12, 0)]] RWStructuredBuffer<PixelState> g_pixels;
// Particle splats: per pixel and XYZ channel, kSplatBins exact 64-bit fixed-point bins (lo, hi words).
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> g_splat;
// Statistics of a pass: paths, segments, inconsistencies, leaks as 64-bit (lo, hi) words (particles),
// then the first region inconsistency (transport.slang's noteInconsistency).
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> g_counters;
// Top-level leaf entries → boundary records (world boundaries, then mesh groups).
[[vk::binding(15, 0)]] StructuredBuffer<uint> g_top;
// Images (gpu_scene.hpp): a header (offset, width, height, flags) per image, then the texels.
[[vk::binding(23, 0)]] StructuredBuffer<uint> g_texels;
// Smooth meshes' shading normals (gpu_scene.hpp): three octahedral words per smooth triangle.
[[vk::binding(24, 0)]] StructuredBuffer<uint> g_normals;
#ifdef OWE_RAY_QUERY
// The same top-level entries as hardware acceleration structures (gpu_accel.hpp); instance IDs
// are boundary records, masks separate matter from null boundaries. Binding 32, above every
// kernel's storage buffers (integrators add their own from 16).
[[vk::binding(32, 0)]] RaytracingAccelerationStructure g_scene;
static const uint kMaskMatter = 1, kMaskNull = 2;
#endif

GGlobals globals() { return g_globals[0]; }
