// Surface geometry in local coordinates. A shape knows nothing about media or
// optics; it only answers "where does a ray meet me" and "which way do I face".
// Every shape has an oriented normal; its +normal side is the boundary's front.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "owe/scene/bvh.hpp"
#include "owe/core/math.hpp"
#include "owe/core/sampling.hpp"

namespace owe {

struct LocalHit {
    double t = Inf;
    Vec3 p;
    Vec3 n;  // unit geometric normal, oriented to the shape's front side
    uint32_t prim = 0;
};

class Shape {
public:
    virtual ~Shape() = default;
    virtual bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const = 0;
    virtual AABB bounds() const = 0;
    virtual std::string describe() const = 0;

    // Emitter sampling support.
    virtual bool canSample() const { return false; }
    virtual double area() const { return 0; }
    virtual void sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const;
    // Samples a point as seen from `ref` with a solid-angle pdf (0 on failure).
    virtual bool sampleFrom(const Vec3& ref, double u1, double u2, double u3, Vec3& p, Vec3& n, double& pdfW) const;
    virtual double pdfFrom(const Vec3& ref, const Vec3& p, const Vec3& n) const;
};

// Rotationally symmetric optical surface z = sag(r), vertex at the origin, axis +z:
//   sag(r) = c r² / (1 + √(1 − (1+k) c² r²)) + Σ A_{2m} r^{2m}   (m ≥ 2)
// trimmed to the annulus rMin ≤ r ≤ rMax (mechanical extent). The clear aperture
// is kept as a separate annotation: it describes intent, not matter.
// Normal points toward +z.
class SagSurface : public Shape {
public:
    SagSurface(double curvature, double conic, std::vector<double> asphere, double rMax, double rMin = 0);
    double sag(double r) const;
    double dsag(double r) const;  // d sag / d r
    double curvature() const { return c_; }
    double conic() const { return k_; }
    const std::vector<double>& asphere() const { return A_; }
    double rMax() const { return rMax_; }
    double rMin() const { return rMin_; }
    double zMin() const { return zMin_; }  // slab enclosing the surface (numeric intersection)
    double zMax() const { return zMax_; }
    bool isFlat() const { return c_ == 0 && A_.empty(); }
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    Vec3 normalAt(const Vec3& p) const;

    bool canSample() const override { return isFlat(); }
    double area() const override { return Pi * (rMax_ * rMax_ - rMin_ * rMin_); }
    void sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const override;

    double clearSemiDiameter = 0;  // annotation only (0 = unspecified)

private:
    bool intersectQuadric(const Ray& r, double tmin, double tmax, LocalHit& h) const;
    bool intersectNumeric(const Ray& r, double tmin, double tmax, LocalHit& h) const;
    bool accept(const Vec3& p) const;
    double c_, k_;
    std::vector<double> A_;
    double rMax_, rMin_;
    double zMin_, zMax_;
};

// A monochromatic plane wave on a liquid surface: amplitude (m), wave vector (rad/m), phase (rad).
struct PlaneWave {
    double amplitude = 0;
    double kx = 0, ky = 0;
    double phase = 0;
};

// A liquid surface z = h(x, y) over the rectangle |x| ≤ halfX, |y| ≤ halfY: a sum of plane waves,
//   h = w(x) w(y) Σ aᵢ sin(kᵢ·(x, y) + φᵢ),
// tapered by a smoothstep w to zero within `margin` of the edges, so flat walls or a basin meet it at
// z = 0. h and its gradient are analytic and the normal exact, so the caustics a water surface
// forms follow its true curvature (a faceted mesh has none: it breaks fold caustics into polygons).
// Intersection marches in steps bounded by the surface's Lipschitz constant along the ray (no
// crossing can be skipped), then refines by Newton's method. Normal +z (the front: above the liquid).
//
// Many waves make every evaluation expensive, so a surface of more than a few waves is sampled: its
// heights and exact derivatives (h, ∂h/∂x, ∂h/∂y, ∂²h/∂x∂y) at nodes `spacing` apart define a bicubic
// Hermite spline, a C¹ surface whose own exact normals are used (a smooth surface, not a mesh). With
// the default spacing of a twelfth of the shortest wavelength it follows the wave sum to ~10⁻³ of
// its amplitude and slope. spacing 0 keeps the analytic sum.
class WaveSurface : public Shape {
public:
    // spacing < 0: automatic (analytic up to 6 waves, else a twelfth of the shortest wavelength).
    WaveSurface(std::vector<PlaneWave> waves, double halfX, double halfY, double margin, double spacing = -1);
    // Height and its gradient at (x, y) (zero outside the rectangle).
    void height(double x, double y, double& h, double& dhdx, double& dhdy) const;
    double height(double x, double y) const {
        double h, gx, gy;
        height(x, y, h, gx, gy);
        return h;
    }
    Vec3 normalAt(double x, double y) const;
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    const std::vector<PlaneWave>& waves() const { return waves_; }
    double halfX() const { return hx_; }
    double halfY() const { return hy_; }
    double margin() const { return margin_; }
    double amplitudeBound() const { return amp_; }  // |h| ≤ amplitudeBound
    double slopeBound() const { return slope_; }    // |∇h| ≤ slopeBound
    // The sampled spline (empty when analytic): nodes row by row from (−halfX, −halfY), each
    // (h, ∂h/∂x, ∂h/∂y, ∂²h/∂x∂y).
    bool sampled() const { return !nodes_.empty(); }
    int nodesX() const { return nx_; }
    int nodesY() const { return ny_; }
    double spacingX() const { return dx_; }
    double spacingY() const { return dy_; }
    const std::vector<std::array<double, 4>>& nodes() const { return nodes_; }
    // The analytic wave sum (with the taper) and its derivatives, whatever the representation.
    void analytic(double x, double y, double& h, double& dhdx, double& dhdy, double& dhdxy) const;

private:
    std::vector<PlaneWave> waves_;
    double hx_, hy_, margin_;
    double amp_ = 0, slope_ = 0;
    int nx_ = 0, ny_ = 0;
    double dx_ = 0, dy_ = 0;
    std::vector<std::array<double, 4>> nodes_;
};

// Plane z = 0 trimmed by an aperture. Normal +z.
class PlaneShape : public Shape {
public:
    enum class Aperture { Disk, Rect, Ellipse, DiskHole };
    static std::shared_ptr<PlaneShape> disk(double rMax, double rMin = 0);
    // Disk of radius rMax with a circular hole of radius holeR centred at (hx, hy).
    static std::shared_ptr<PlaneShape> diskWithHole(double rMax, double holeR, double hx, double hy);
    static std::shared_ptr<PlaneShape> rect(double halfX, double halfY);
    static std::shared_ptr<PlaneShape> ellipse(double semiX, double semiY);
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    // A holed disk has no exact area sampler here; emitters on it are still found by BSDF sampling.
    bool canSample() const override { return ap_ != Aperture::DiskHole; }
    double area() const override;
    void sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const override;
    Aperture aperture() const { return ap_; }
    double a() const { return a_; }
    double b() const { return b_; }
    double holeX() const { return hx_; }
    double holeY() const { return hy_; }

private:
    Aperture ap_ = Aperture::Disk;
    double a_ = 0, b_ = 0;  // disk: rMax, rMin; rect: halfX, halfY; ellipse: semiX, semiY; hole: rMax, holeR
    double hx_ = 0, hy_ = 0;  // hole centre
};

// Full sphere centred at the origin, outward normal.
class SphereShape : public Shape {
public:
    explicit SphereShape(double radius) : R_(radius) {}
    double radius() const { return R_; }
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    bool canSample() const override { return true; }
    double area() const override { return 4 * Pi * R_ * R_; }
    void sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const override;
    bool sampleFrom(const Vec3& ref, double u1, double u2, double u3, Vec3& p, Vec3& n, double& pdfW) const override;
    double pdfFrom(const Vec3& ref, const Vec3& p, const Vec3& n) const override;

private:
    double R_;
};

// Open cylinder x² + y² = R², z0 ≤ z ≤ z1, outward normal.
class CylinderShape : public Shape {
public:
    CylinderShape(double radius, double z0, double z1) : R_(radius), z0_(std::min(z0, z1)), z1_(std::max(z0, z1)) {}
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    bool canSample() const override { return true; }
    double area() const override { return 2 * Pi * R_ * (z1_ - z0_); }
    void sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const override;
    double radius() const { return R_; }
    double z0() const { return z0_; }
    double z1() const { return z1_; }

private:
    double R_, z0_, z1_;
};

// Cone frustum: the surface of revolution of the segment (r0, z0)–(r1, z1) about the z axis
// (z0 ≠ z1, r0 ≠ r1; flat and cylindrical segments are planes and cylinders). Normal away from the
// axis, tilted by the slope: (x, y, −ρk)/|…| with ρ = r(z), k = dr/dz.
class ConeShape : public Shape {
public:
    ConeShape(double r0, double z0, double r1, double z1);
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    double r0() const { return r0_; }
    double z0() const { return z0_; }
    double r1() const { return r1_; }
    double z1() const { return z1_; }

private:
    double r0_, z0_, r1_, z1_, k_;
};

// A patch of a torus: the surface of revolution of the circular arc of radius rho centred at
// (R, zc) in the (r, z) half-plane, for arc angles a0 ≤ a ≤ a1 (a from the +r direction toward +z;
// |a1 − a0| ≤ π), about the z axis — the fillet of a vessel's rounded corner. The implicit
// (√(x²+y²) − R)² + (z − zc)² = rho² is found by bracketing along the ray and refined to full
// precision. Normal away from the arc's centre.
class TorusPatchShape : public Shape {
public:
    TorusPatchShape(double R, double zc, double rho, double a0, double a1);
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    double majorRadius() const { return R_; }
    double zc() const { return zc_; }
    double rho() const { return rho_; }
    double a0() const { return a0_; }
    double a1() const { return a1_; }
    // The patch's (r, z) extent.
    double rMin() const { return rMin_; }
    double rMax() const { return rMax_; }
    double zMin() const { return zMin_; }
    double zMax() const { return zMax_; }

private:
    bool onArc(const Vec3& p) const;
    double R_, zc_, rho_, a0_, a1_;
    double rMin_, rMax_, zMin_, zMax_;
};

// Round wall: open cylinder of radius R from z = 0 to H with rectangular openings
// (azimuth from +y toward +x, width along the wall, sill and top heights). Outward normal.
class RoundWallShape : public Shape {
public:
    struct Opening {
        double azimuth, width, sill, top;
    };
    RoundWallShape(double R, double H, std::vector<Opening> openings) : R_(R), H_(H), open_(std::move(openings)) {}
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    double radius() const { return R_; }
    double height() const { return H_; }
    const std::vector<Opening>& openings() const { return open_; }

private:
    bool solid(const Vec3& p) const;
    double R_, H_;
    std::vector<Opening> open_;
};

// Observatory dome: hemisphere of radius R (z ≥ 0) with an observing slit of constant linear
// width centred on an azimuth, open from the horizon up to elevation slitTop. Outward normal.
class DomeShape : public Shape {
public:
    DomeShape(double R, double slitAz, double slitWidth, double slitTop)
        : R_(R), az_(slitAz), w_(slitWidth), top_(slitTop) {}
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override;
    std::string describe() const override;
    double radius() const { return R_; }
    double slitAzimuth() const { return az_; }
    double slitWidth() const { return w_; }
    double slitTop() const { return top_; }

private:
    bool solid(const Vec3& p) const;
    double R_, az_, w_, top_;
};

// Triangle mesh; front side follows counter-clockwise winding. A smooth mesh also has shading
// normals, one per triangle corner, each on its triangle's front side: the GPU kernels interpolate
// them and scatter about the result, while the geometric normal still decides sides (regions,
// offsets). The intersection here stays geometric. A textured model also has texture
// coordinates per triangle corner (u across the image, v down), interpolated the same way.
class MeshShape : public Shape {
public:
    using Corners = std::array<Vec3, 3>;
    using UV = std::array<float, 2>;
    using UvCorners = std::array<UV, 3>;
    MeshShape(std::vector<Vec3> positions, std::vector<std::array<uint32_t, 3>> triangles, std::string label = "mesh",
              std::vector<Corners> shadingNormals = {}, std::vector<UvCorners> uvs = {});
    bool intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const override;
    AABB bounds() const override { return bvh_.bounds(); }
    std::string describe() const override;
    bool canSample() const override { return area_ > 0; }
    double area() const override { return area_; }
    void sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const override;
    size_t triangleCount() const { return tris_.size(); }
    const std::vector<Vec3>& positions() const { return pos_; }
    const std::vector<std::array<uint32_t, 3>>& triangles() const { return tris_; }
    const Bvh& bvh() const { return bvh_; }
    const std::vector<double>& areaCdf() const { return cdf_; }
    const std::vector<Corners>& shadingNormals() const { return normals_; }  // empty: flat
    const std::vector<UvCorners>& uvs() const { return uvs_; }                // empty: none

private:
    std::vector<Vec3> pos_;
    std::vector<std::array<uint32_t, 3>> tris_;
    std::vector<Corners> normals_;
    std::vector<UvCorners> uvs_;
    std::vector<double> cdf_;
    double area_ = 0;
    Bvh bvh_;
    std::string label_;
};

// Polygon soup helpers used by builders.
struct MeshData {
    static constexpr uint32_t kNoNormal = 0xFFFFFFFFu;
    std::vector<Vec3> positions;
    std::vector<std::array<uint32_t, 3>> triangles;
    // Normals given by a file (OBJ vn) and, per triangle, each corner's normal (kNoNormal where the
    // face gave none). Both empty when the file has no normals; builders leave them empty.
    std::vector<Vec3> normals;
    std::vector<std::array<uint32_t, 3>> normalIndex;
    // Texture coordinates per triangle corner (glTF TEXCOORD_0); empty when the file has none.
    std::vector<MeshShape::UvCorners> uvs;
    // A model's parts (glTF materials): each triangle's part, and each part's name and base colour
    // image (a path, empty if none). Empty for OBJ files and builders.
    std::vector<uint32_t> part;
    std::vector<std::string> partNames, partImages;
    void addTriangle(const Vec3& a, const Vec3& b, const Vec3& c);
    void addQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d);  // CCW
    void append(const MeshData& o, const Transform& xf);
};

// Load a Wavefront OBJ (positions, faces and vertex normals). Throws on failure.
MeshData loadObj(const std::string& path, double scale = 1.0);
// Load a glTF 2.0 model (.gltf with its buffers, or .glb): every mesh of its default scene placed
// by its node transforms, turned from glTF's y-up to z-up, with normals, texture coordinates and
// its materials as parts. Throws on failure.
MeshData loadGltf(const std::string& path, double scale = 1.0);
// loadGltf for .gltf and .glb files, loadObj otherwise.
MeshData loadModel(const std::string& path, double scale = 1.0);
// The triangles of the named parts only (vertices and normals kept whole).
MeshData selectParts(const MeshData& m, const std::vector<std::string>& names);

// Shading normals of a smooth mesh, per triangle corner: at each vertex, the angle-weighted mean of
// the normals of the triangles around it (vertices at the same position count as one) that meet the
// corner's triangle at less than `creaseAngle`; sharper edges stay sharp.
std::vector<MeshShape::Corners> smoothNormals(const MeshData& m, double creaseAngle);
// The file's normals per corner, each turned to its triangle's front side; a corner without one
// takes its triangle's geometric normal. Throws when the mesh has no normals.
std::vector<MeshShape::Corners> fileNormals(const MeshData& m);

}  // namespace owe
