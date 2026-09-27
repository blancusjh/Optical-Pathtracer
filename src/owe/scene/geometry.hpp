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

// Triangle mesh; front side follows counter-clockwise winding.
class MeshShape : public Shape {
public:
    MeshShape(std::vector<Vec3> positions, std::vector<std::array<uint32_t, 3>> triangles, std::string label = "mesh");
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

private:
    std::vector<Vec3> pos_;
    std::vector<std::array<uint32_t, 3>> tris_;
    std::vector<double> cdf_;
    double area_ = 0;
    Bvh bvh_;
    std::string label_;
};

// Polygon soup helpers used by builders.
struct MeshData {
    std::vector<Vec3> positions;
    std::vector<std::array<uint32_t, 3>> triangles;
    void addTriangle(const Vec3& a, const Vec3& b, const Vec3& c);
    void addQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d);  // CCW
    void append(const MeshData& o, const Transform& xf);
};

// Load a Wavefront OBJ (positions and faces only). Throws on failure.
MeshData loadObj(const std::string& path, double scale = 1.0);

}  // namespace owe
