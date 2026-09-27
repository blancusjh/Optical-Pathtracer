// Interface optics against closed-form results.
#include "check.hpp"
#include "owe/core/medium.hpp"
#include "owe/transport/optics.hpp"
#include "owe/transport/scattering.hpp"
#include "owe/core/spectrum.hpp"

using namespace owe;

TEST(mirror_law_equal_angles_and_coplanar) {
    Rng rng(1, 2);
    for (int i = 0; i < 1000; ++i) {
        Vec3 n = sampleUniformSphere(rng.uniform(), rng.uniform());
        Vec3 d = sampleUniformSphere(rng.uniform(), rng.uniform());
        if (dot(d, n) > 0) d = -d;
        Vec3 r = reflect(d, n);
        CHECK_NEAR(dot(r, n), -dot(d, n), 1e-14);   // θr = θi
        CHECK_NEAR(dot(cross(d, n), r), 0.0, 1e-14);  // plane of incidence
        CHECK_NEAR(length(r), 1.0, 1e-14);
    }
}

TEST(snell_law_holds_for_random_rays) {
    Rng rng(3, 4);
    for (int i = 0; i < 2000; ++i) {
        double n1 = 1 + rng.uniform(), n2 = 1 + rng.uniform();
        Vec3 n = sampleUniformSphere(rng.uniform(), rng.uniform());
        Vec3 d = sampleUniformSphere(rng.uniform(), rng.uniform());
        Vec3 t;
        double sin1 = length(cross(d, n));
        bool ok = refractDirection(d, n, n1, n2, t);
        if (n1 * sin1 > n2) { CHECK(!ok); continue; }
        CHECK(ok);
        double sin2 = length(cross(t, n));
        CHECK_NEAR(n1 * sin1, n2 * sin2, 1e-12);
        CHECK_NEAR(dot(cross(d, n), t), 0.0, 1e-12);
        CHECK(dot(t, n) * dot(d, n) > 0);  // continues through the surface
    }
}

TEST(fresnel_normal_incidence_brewster_and_energy) {
    double n1 = 1.0, n2 = 1.5;
    double Rs, Rp;
    fresnelDielectricSP(1.0, n1, n2, Rs, Rp);
    CHECK_NEAR(Rs, 0.04, 1e-15);
    CHECK_NEAR(Rp, 0.04, 1e-15);
    CHECK_NEAR(fresnelDielectric(1.0, n2 / n1), 0.04, 1e-15);
    // Brewster angle: p-polarised reflectance vanishes.
    double thB = std::atan(n2 / n1);
    fresnelDielectricSP(std::cos(thB), n1, n2, Rs, Rp);
    CHECK_NEAR(Rp, 0.0, 1e-15);
    CHECK(Rs > 0.1);
    // Energy conservation with the transmission coefficients.
    for (double th = 0; th < 1.5; th += 0.05) {
        double ci = std::cos(th), st = n1 / n2 * std::sin(th), ct = std::sqrt(1 - st * st);
        fresnelDielectricSP(ci, n1, n2, Rs, Rp);
        double ts = 2 * n1 * ci / (n1 * ci + n2 * ct), tp = 2 * n1 * ci / (n2 * ci + n1 * ct);
        double Ts = n2 * ct / (n1 * ci) * ts * ts, Tp = n2 * ct / (n1 * ci) * tp * tp;
        CHECK_NEAR(Rs + Ts, 1.0, 1e-13);
        CHECK_NEAR(Rp + Tp, 1.0, 1e-13);
        CHECK_NEAR(fresnelDielectric(ci, n2 / n1), 0.5 * (Rs + Rp), 1e-13);
        // Incidence from the dense side uses the inverted ratio (reciprocity of R).
        double ciInside = ct;
        CHECK_NEAR(fresnelDielectric(-ciInside, n2 / n1), 0.5 * (Rs + Rp), 1e-12);
    }
}

TEST(total_internal_reflection_at_critical_angle) {
    double n1 = 1.5168, n2 = 1.0;
    double thc = std::asin(n2 / n1);
    Vec3 n{0, 0, 1}, t;
    for (double eps : {1e-6, 1e-3}) {
        Vec3 below{std::sin(thc - eps), 0, -std::cos(thc - eps)};
        Vec3 above{std::sin(thc + eps), 0, -std::cos(thc + eps)};
        CHECK(refractDirection(below, n, n1, n2, t));
        CHECK(!refractDirection(above, n, n1, n2, t));
        double Rs, Rp;
        CHECK(!fresnelDielectricSP(std::cos(thc + eps), n1, n2, Rs, Rp));
        CHECK(Rs == 1 && Rp == 1);
        CHECK(fresnelDielectric(std::cos(thc + eps), n2 / n1) == 1.0);
        CHECK(fresnelDielectric(std::cos(thc - eps), n2 / n1) < 1.0);
    }
}

TEST(conductor_fresnel_limits) {
    // k = 0 reduces to the dielectric result; normal incidence closed form.
    for (double th = 0; th < 1.5; th += 0.1)
        CHECK_NEAR(fresnelConductor(std::cos(th), {1.5, 0}), fresnelDielectric(std::cos(th), 1.5), 1e-13);
    double n = 0.96, k = 6.69;
    CHECK_NEAR(fresnelConductor(1.0, {n, k}), ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k), 1e-13);
    CHECK_NEAR(fresnelConductor(1e-9, {n, k}), 1.0, 1e-6);  // grazing
}

TEST(glass_catalog_matches_published_nd_vd) {
    struct Row { const char* name; double nd, vd, tolN, tolV; };
    const Row rows[] = {
        {"N-BK7", 1.51680, 64.17, 2e-5, 0.05},   {"N-SK16", 1.62041, 60.32, 2e-5, 0.05},
        {"N-BAF10", 1.67003, 47.11, 2e-5, 0.05}, {"F2", 1.62004, 36.37, 2e-5, 0.05},
        {"N-F2", 1.62005, 36.43, 2e-5, 0.05},    {"N-SF6", 1.80518, 25.36, 2e-5, 0.05},
        {"N-SF11", 1.78472, 25.68, 2e-5, 0.05},  {"fused-silica", 1.45846, 67.82, 5e-5, 0.1},
        {"CaF2", 1.43385, 94.99, 5e-5, 0.3},     {"water", 1.3333, 55.8, 1e-3, 1.0},
        {"diamond", 2.4175, 55.3, 1e-3, 1.5},    {"sapphire", 1.7682, 72.2, 5e-4, 0.5},
    };
    for (const Row& r : rows) {
        Medium m;
        CHECK(catalogMedium(r.name, m));
        CHECK_NEAR(m.index.nRelative(LambdaD), r.nd, r.tolN);
        CHECK_NEAR(abbeNumber(m.index), r.vd, r.tolV);
    }
    Medium air;
    catalogMedium("air", air);
    CHECK_NEAR(air.n(LambdaD) - 1, 2.772e-4, 2e-6);
    // Relative-to-air glass: absolute index = relative × n_air.
    Medium bk7;
    catalogMedium("N-BK7", bk7);
    CHECK_NEAR(bk7.n(LambdaD), bk7.index.nRelative(LambdaD) * air.n(LambdaD), 1e-15);
    // Normal dispersion: n decreases with wavelength.
    CHECK(bk7.n(LambdaF) > bk7.n(LambdaD) && bk7.n(LambdaD) > bk7.n(LambdaC));
}

TEST(spectral_sampling_and_colorimetry) {
    CHECK_NEAR(cieYIntegral(), 106.86, 0.5);
    CHECK_NEAR(Spectrum::constant(1).luminance(), 1.0, 1e-12);
    CHECK_NEAR(Spectrum::blackbody(3000, 7.5).luminance(), 7.5, 1e-9);
    CHECK_NEAR(Spectrum::rgbIlluminant(1, 1, 1).luminance(), 1.0, 1e-9);
    // The wavelength pdf integrates to one and its sampler is its inverse CDF.
    double integral = 0, h = 0.01;
    for (double l = LambdaMin; l < LambdaMax; l += h) integral += visibleWavelengthPdf(l + h / 2) * h;
    CHECK_NEAR(integral, 1.0, 2e-4);
    for (double u : {0.1, 0.3, 0.5, 0.9}) {
        double l = sampleVisibleWavelength(u), cdf = 0;
        for (double x = LambdaMin; x < l; x += h) cdf += visibleWavelengthPdf(x + h / 2) * h;
        CHECK_NEAR(cdf, u, 5e-4);
    }
    // RGB reflectances stay physical.
    Rng rng(9, 9);
    for (int i = 0; i < 200; ++i) {
        Spectrum s = Spectrum::rgbReflectance(rng.uniform(), rng.uniform(), rng.uniform());
        for (double l = LambdaMin; l <= LambdaMax; l += 7) CHECK(s(l) >= 0 && s(l) <= 1 + 1e-12);
    }
    CHECK_NEAR(Spectrum::rgbReflectance(1, 1, 1)(555), 1.0, 1e-12);
    // The 6504 K Planckian white maps close to neutral sRGB (D65 differs slightly from a blackbody).
    double rgb[3];
    xyzToLinearSRGB(Spectrum::rgbIlluminant(1, 1, 1).toXYZ(), rgb);
    CHECK(std::abs(rgb[0] / rgb[1] - 1) < 0.08 && std::abs(rgb[2] / rgb[1] - 1) < 0.08);
}

TEST(ggx_distribution_normalised) {
    // Integrate in s = tanθ/α (θ = atan(α s)), mapped from [0,1) by s = y/(1−y), so the peak is resolved.
    for (double a : {0.05, 0.2, 0.6}) {
        GGX mf(a);
        auto theta = [&](double y, double& jac) {
            double sv = y / (1 - y), ds = 1 / sqr(1 - y);
            double th = std::atan(a * sv);
            jac = a / (1 + a * a * sv * sv) * ds;
            return th;
        };
        double s = 0;
        int N = 200000;
        for (int i = 0; i < N; ++i) {
            double jac, th = theta((i + 0.5) / N, jac);
            s += mf.D({std::sin(th), 0, std::cos(th)}) * std::cos(th) * std::sin(th) * 2 * Pi * jac / N;
        }
        CHECK_NEAR(s, 1.0, 1e-4);
        // ∫ D_wo(m) dω = 1 over front-facing microfacets (wo·m > 0) for the distribution of visible normals.
        Vec3 wo = normalize(Vec3(0.5, 0.1, 0.6));
        double v = 0;
        int M = 4000, P = 720;
        for (int i = 0; i < M; ++i) {
            double jac, th = theta((i + 0.5) / M, jac);
            for (int j = 0; j < P; ++j) {
                double phi = 2 * Pi * (j + 0.5) / P;
                Vec3 m{std::sin(th) * std::cos(phi), std::sin(th) * std::sin(phi), std::cos(th)};
                if (dot(wo, m) <= 0) continue;
                v += mf.D(wo, m) * std::sin(th) * jac / M * (2 * Pi / P);
            }
        }
        CHECK_NEAR(v, 1.0, 2e-3);
    }
}

static Interface makeIt(const SurfaceOptics& o, double nf, double nb) {
    Interface it;
    it.optics = &o;
    it.n = {0, 0, 1};
    it.nFront = nf;
    it.nBack = nb;
    it.lambda = 550;
    return it;
}

TEST(scatter_sampling_consistent_with_evaluation) {
    std::vector<SurfaceOptics> list(4);
    list[0].type = SurfaceType::Diffuse;
    list[0].reflectance = Spectrum::constant(0.7);
    list[1].type = SurfaceType::Conductor;
    list[1].roughness = 0.3;
    catalogConductor("gold", list[1].conductor);
    list[2].type = SurfaceType::Dielectric;
    list[2].roughness = 0.25;
    list[3].type = SurfaceType::Dielectric;
    list[3].roughness = 0.6;
    Rng rng(5, 6);
    for (const auto& o : list) {
        Interface it = makeIt(o, 1.0, 1.5);
        int checked = 0;
        for (int i = 0; i < 3000; ++i) {
            Vec3 d = sampleUniformSphere(rng.uniform(), rng.uniform());
            for (TransportMode mode : {TransportMode::Radiance, TransportMode::Importance}) {
                ScatterSample s;
                if (!sampleScatter(it, d, rng.uniform(), rng.uniform(), rng.uniform(), mode, s)) continue;
                double pdf;
                double f = evalScatter(it, d, s.wi, mode, pdf);
                if (pdf < 1e-3 || s.pdf < 1e-3) continue;
                CHECK_REL(s.pdf, pdf, 1e-6);
                CHECK_REL(s.weight, f * std::abs(s.wi.z) / pdf, 1e-6);
                ++checked;
            }
        }
        CHECK(checked > 1000);
    }
}

TEST(rough_reflection_is_reciprocal) {
    SurfaceOptics c;
    c.type = SurfaceType::Conductor;
    c.roughness = 0.2;
    catalogConductor("aluminium", c.conductor);
    SurfaceOptics d;
    d.type = SurfaceType::Dielectric;
    d.roughness = 0.3;
    Rng rng(11, 12);
    for (const SurfaceOptics* o : {&c, &d}) {
        Interface it = makeIt(*o, 1.0, 1.5);
        for (int i = 0; i < 500; ++i) {
            Vec3 a = sampleCosineHemisphere(rng.uniform(), rng.uniform());
            Vec3 b = sampleCosineHemisphere(rng.uniform(), rng.uniform());
            double p1, p2;
            double f1 = evalScatter(it, -a, b, TransportMode::Radiance, p1);
            double f2 = evalScatter(it, -b, a, TransportMode::Radiance, p2);
            CHECK_NEAR(f1, f2, 1e-9 * (1 + f1));
        }
    }
}

TEST(smooth_dielectric_branching_probabilities) {
    SurfaceOptics o;
    o.type = SurfaceType::Dielectric;
    Interface it = makeIt(o, 1.0, 1.5);
    Vec3 d = normalize(Vec3(std::sin(0.7), 0, -std::cos(0.7)));
    double R = fresnelDielectric(std::cos(0.7), 1.5);
    int refl = 0, N = 200000;
    Rng rng(1, 1);
    for (int i = 0; i < N; ++i) {
        ScatterSample s;
        CHECK(sampleScatter(it, d, rng.uniform(), 0, 0, TransportMode::Radiance, s));
        if (!s.transmitted) {
            ++refl;
            CHECK_NEAR(s.weight, 1.0, 1e-15);
        } else {
            CHECK_NEAR(s.weight, sqr(1.0 / 1.5), 1e-15);  // radiance scales by (n_i/n_t)²
        }
    }
    CHECK_NEAR(double(refl) / N, R, 4 * std::sqrt(R * (1 - R) / N));
}

TEST(fresnel_floor_changes_variance_not_expectation) {
    SurfaceOptics o;
    o.type = SurfaceType::Dielectric;
    Interface it = makeIt(o, 1.0, 1.5);
    it.fresnelFloor = 0.3;
    Vec3 d = normalize(Vec3(std::sin(0.4), 0, -std::cos(0.4)));
    double R = fresnelDielectric(std::cos(0.4), 1.5);
    double sumR = 0, sumT = 0;
    int N = 400000, nR = 0;
    Rng rng(2, 2);
    for (int i = 0; i < N; ++i) {
        ScatterSample s;
        CHECK(sampleScatter(it, d, rng.uniform(), 0, 0, TransportMode::Importance, s));
        if (s.transmitted) sumT += s.weight; else { sumR += s.weight; ++nR; }
    }
    CHECK_NEAR(double(nR) / N, 0.3, 0.005);               // branch chosen with the floor probability
    CHECK_NEAR(sumR / N, R, 4 * std::sqrt(0.3 * 0.7 / N) * R / 0.3);  // expected reflected energy is still R
    CHECK_NEAR(sumT / N, 1 - R, 4 * std::sqrt(0.3 * 0.7 / N) * (1 - R) / 0.7);
}
