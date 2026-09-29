// Shading normals on the GPU: smooth mesh normals (corner normals from a crease angle or a file's
// vn, interpolated across each triangle), material maps and surface relief. The kernels scatter
// about the shading normal while the geometric normal keeps deciding sides. A tessellated sphere
// with smooth normals must shade like the analytic sphere it approximates, far closer than the same
// mesh flat; camera paths, particles connected to the eye and photon mapping must agree on a smooth
// mesh, which is what Veach's adjoint correction for shading normals is for (without it the
// particle estimators see a different BSDF than the camera paths).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "check.hpp"
#include "owe/backends/registry.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/scene/detector.hpp"
#include "owe/scene/geometry.hpp"
#include "owe/scene/world.hpp"

using namespace owe;
namespace fs = std::filesystem;

namespace {

// A UV sphere of radius r, wound counter-clockwise seen from outside: `rings` bands of latitude,
// `segments` of longitude, the poles as fans. withNormals: the exact normals as vn, faces v//vn.
std::string uvSphereObj(double r, int segments, int rings, bool withNormals) {
    std::ostringstream o;
    o.precision(17);
    std::vector<Vec3> v{{0, 0, r}};
    for (int i = 1; i < rings; ++i)
        for (int j = 0; j < segments; ++j) {
            double th = Pi * i / rings, ph = 2 * Pi * j / segments;
            v.push_back({r * std::sin(th) * std::cos(ph), r * std::sin(th) * std::sin(ph), r * std::cos(th)});
        }
    v.push_back({0, 0, -r});
    for (const Vec3& p : v) o << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
    if (withNormals)
        for (const Vec3& p : v) {
            Vec3 n = normalize(p);
            o << "vn " << n.x << ' ' << n.y << ' ' << n.z << '\n';
        }
    auto at = [&](int i, int j) { return 2 + (i - 1) * segments + (j % segments); };
    const int bottom = int(v.size());
    auto face = [&](int a, int b, int c) {
        if (withNormals) o << "f " << a << "//" << a << ' ' << b << "//" << b << ' ' << c << "//" << c << '\n';
        else o << "f " << a << ' ' << b << ' ' << c << '\n';
    };
    for (int j = 0; j < segments; ++j) {
        face(1, at(1, j), at(1, j + 1));
        for (int i = 1; i + 1 < rings; ++i) {
            face(at(i, j), at(i + 1, j), at(i + 1, j + 1));
            face(at(i, j), at(i + 1, j + 1), at(i, j + 1));
        }
        face(bottom, at(rings - 1, j + 1), at(rings - 1, j));
    }
    return o.str();
}

std::string writeTemp(const std::string& name, const std::string& text) {
    const fs::path p = fs::temp_directory_path() / name;
    std::ofstream(p) << text;
    return p.string();
}

bool gpuAvailable(const char* integrator) {
    const Backend* gpu = findBackend("gpu");
    return gpu && gpu->available() && gpu->supports(integrator);
}

Image renderGpu(const Scene& sc, const std::string& integrator, int spp, TransportStats* stats = nullptr) {
    RenderSettings rs;
    rs.backend = "gpu";
    rs.integrator = integrator;
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(spp);
    if (stats) *stats = R->stats();
    return R->resolve();
}

// A diffuse ball of radius 1 m, 4 m in front of the eye, lit by the sun from behind the eye's
// left shoulder (irradiance ~1.4 against the dim sky's 0.16); `ball` is the body's type and keys.
Scene ballScene(const std::string& ball, const std::string& extra = "") {
    return loadSceneFromString(R"(
        units = m
        world { sky = uniform(rgb(1, 1, 1, luminance = 0.05))  sun { elevation = 35deg  azimuth = 200deg  luminance = 2e4 } }
        material clay { type = diffuse  reflectance = 0.7 }
        body Ball { )" + ball + R"(  position = (0, 0, 0)  material = clay }
        )" + extra + R"(
        observer Eye { position = (0, -4, 0)  look_at = (0, 0, 0)  up = (0, 0, 1)  fov = 34deg  pupil = 1mm  focus = 4
                       resolution = (96, 96) }
        render { detector = Eye }
    )");
}

// Relative L1 difference of a against b over the pixels within `frac` of the ball's image radius.
double discL1(const Image& a, const Image& b, double frac) {
    const double radius = std::tan(std::asin(0.25)) / std::tan(17.0 * Pi / 180) * 48 * frac;
    double num = 0, den = 0;
    for (int y = 0; y < a.height; ++y)
        for (int x = 0; x < a.width; ++x) {
            if (std::hypot(x + 0.5 - 48, y + 0.5 - 48) > radius) continue;
            num += std::abs(a.at(x, y).y - b.at(x, y).y);
            den += b.at(x, y).y;
        }
    return num / den;
}

// The ratio of a's to b's mean in each of 4 × 4 blocks, as the largest deviation from 1 over the
// blocks holding at least `share` of an average block's light in b.
double worstBlock(const Image& a, const Image& b, double share = 0.25) {
    const int n = 4, bw = a.width / n, bh = a.height / n;
    std::vector<double> sa(n * n, 0), sb(n * n, 0);
    for (int y = 0; y < n * bh; ++y)
        for (int x = 0; x < n * bw; ++x) {
            sa[size_t((y / bh) * n + x / bw)] += a.at(x, y).y;
            sb[size_t((y / bh) * n + x / bw)] += b.at(x, y).y;
        }
    double mean = 0;
    for (double v : sb) mean += v / (n * n);
    double worst = 0;
    for (int k = 0; k < n * n; ++k)
        if (sb[size_t(k)] >= share * mean) worst = std::max(worst, std::abs(sa[size_t(k)] / sb[size_t(k)] - 1));
    return worst;
}

}  // namespace

TEST(shading_smooth_normals_keep_creases_and_follow_curvature) {
    // A cube: every edge is 90°, sharper than a 30° crease, so each corner keeps its face's normal.
    MeshData cube;
    const double h = 0.5;
    Vec3 c[8];
    for (int i = 0; i < 8; ++i) c[i] = {(i & 1) ? h : -h, (i & 2) ? h : -h, (i & 4) ? h : -h};
    cube.addQuad(c[0], c[2], c[3], c[1]);
    cube.addQuad(c[4], c[5], c[7], c[6]);
    cube.addQuad(c[0], c[1], c[5], c[4]);
    cube.addQuad(c[2], c[6], c[7], c[3]);
    cube.addQuad(c[0], c[4], c[6], c[2]);
    cube.addQuad(c[1], c[3], c[7], c[5]);
    auto cn = smoothNormals(cube, 30 * Pi / 180);
    double worst = 0;
    for (size_t i = 0; i < cube.triangles.size(); ++i) {
        const auto& t = cube.triangles[i];
        Vec3 fn = normalize(cross(cube.positions[t[1]] - cube.positions[t[0]], cube.positions[t[2]] - cube.positions[t[0]]));
        for (const Vec3& n : cn[i]) worst = std::max(worst, length(n - fn));
    }
    CHECK(worst < 1e-12);

    // A UV sphere (vertices split per face, as files split them at seams: welded by position):
    // every corner normal close to the exact sphere normal there.
    MeshData sphere;
    {
        const std::string path = writeTemp("owe_test_sphere_flat.obj", uvSphereObj(1, 32, 16, false));
        MeshData shared = loadObj(path);
        for (const auto& t : shared.triangles)
            sphere.addTriangle(shared.positions[t[0]], shared.positions[t[1]], shared.positions[t[2]]);
    }
    cn = smoothNormals(sphere, 40 * Pi / 180);
    worst = 0;
    for (size_t i = 0; i < sphere.triangles.size(); ++i)
        for (int k = 0; k < 3; ++k) {
            Vec3 exact = normalize(sphere.positions[sphere.triangles[i][size_t(k)]]);
            worst = std::max(worst, std::acos(std::min(1.0, dot(cn[i][size_t(k)], exact))));
        }
    std::fprintf(stderr, "    UV sphere 32 × 16: largest corner-normal error %.3f°\n", worst * 180 / Pi);
    CHECK(worst < 1.0 * Pi / 180);
}

TEST(shading_obj_vertex_normals_are_read_and_face_their_triangle) {
    // v//vn, v/vt/vn and relative indices; the second face's file normal points behind it and
    // gives way to the face's own normal.
    const std::string path = writeTemp("owe_test_vn.obj",
                                       "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
                                       "vt 0 0\n"
                                       "vn 0.6 0 0.8\nvn 0 0 1\nvn 0 0 -1\n"
                                       "f 1//1 2//2 3//2\n"
                                       "f -3/1/-1 -1/1/-1 -2/1/-1\n");
    MeshData m = loadObj(path);
    CHECK(m.triangles.size() == 2);
    CHECK(m.normals.size() == 3);
    CHECK(m.normalIndex.size() == 2);
    auto cn = fileNormals(m);
    CHECK_NEAR(cn[0][0].x, 0.6, 1e-12);
    CHECK_NEAR(cn[0][0].z, 0.8, 1e-12);
    CHECK_NEAR(cn[0][1].z, 1.0, 1e-12);
    // Face 2 is (2, 4, 3): counter-clockwise from +z, so its file normal (0, 0, −1) is behind it.
    CHECK_NEAR(cn[1][0].z, 1.0, 1e-12);
    // A mesh without vn has none to give.
    bool threw = false;
    try {
        fileNormals(loadObj(writeTemp("owe_test_novn.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")));
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(shading_smooth_sphere_mesh_shades_like_the_sphere) {
    if (!gpuAvailable("path")) return;
    const std::string flat = writeTemp("owe_test_ball.obj", uvSphereObj(1, 32, 16, false));
    const std::string withVn = writeTemp("owe_test_ball_vn.obj", uvSphereObj(1, 32, 16, true));
    const Image exact = renderGpu(ballScene("type = sphere  radius = 1"), "path", 256);
    const Image faceted = renderGpu(ballScene("type = mesh  file = \"" + flat + "\""), "path", 256);
    TransportStats st;
    const Image smooth = renderGpu(ballScene("type = mesh  file = \"" + flat + "\"  normals = smooth"), "path", 256, &st);
    const Image fromFile = renderGpu(ballScene("type = mesh  file = \"" + withVn + "\"  normals = file"), "path", 256);
    const double eFlat = discL1(faceted, exact, 0.85), eSmooth = discL1(smooth, exact, 0.85),
                 eFile = discL1(fromFile, exact, 0.85);
    std::fprintf(stderr, "    against the analytic sphere (L1): flat %.4f, smooth %.4f, file normals %.4f\n", eFlat, eSmooth,
                 eFile);
    CHECK(eSmooth < 0.35 * eFlat);
    CHECK(eSmooth < 0.02);
    CHECK(eFile < 0.02);
    CHECK(st.inconsistencies == 0);
}

TEST(shading_estimators_agree_on_a_smooth_mesh) {
    // A coarse ball (12 × 6: shading and geometric normals differ by up to 15°) on a floor before a
    // wall (particles cannot see the sky directly: every pixel sees matter), so light reaches the
    // eye from the ball directly, from the floor via the ball, and from the ball via the floor.
    // Camera paths, particles connected to the eye and SPPM see the same image only if the
    // particles scatter with the adjoint of the shading-normal BSDF.
    if (!gpuAvailable("sppm")) return;
    const std::string obj = writeTemp("owe_test_coarse_ball.obj", uvSphereObj(1, 12, 6, false));
    const Scene sc = ballScene("type = mesh  file = \"" + obj + "\"  normals = smooth",
                               "material ground { type = diffuse  reflectance = 0.5 }\n"
                               "body Floor { type = sheet  size = (12, 12)  position = (0, 0, -1.02)  material = ground }\n"
                               "body Wall { type = sheet  size = (12, 8)  position = (0, 2.5, 2.98)  axis = (0, -1, 0)  up = (0, 0, 1)  "
                               "material = ground }");
    const Image path = renderGpu(sc, "path", 4096);
    const Image light = renderGpu(sc, "light", 1024);
    const Image sppm = renderGpu(sc, "sppm", 4096);
    const double lightMean = light.meanY() / path.meanY(), sppmMean = sppm.meanY() / path.meanY();
    const double lightBlocks = worstBlock(light, path), sppmBlocks = worstBlock(sppm, path);
    std::fprintf(stderr, "    against camera paths: particles mean %.4f (worst block %.4f), sppm mean %.4f (worst block %.4f)\n",
                 lightMean, lightBlocks, sppmMean, sppmBlocks);
    CHECK_NEAR(lightMean, 1.0, 0.01);
    CHECK_NEAR(sppmMean, 1.0, 0.02);
    CHECK(lightBlocks < 0.03);
    CHECK(sppmBlocks < 0.05);
}

TEST(shading_normals_shape_a_sunbeam_as_their_cosine) {
    // A 1 m square of two triangles whose corner normals lean up to 30° away from its plane, lit
    // only by a sunbeam that a mirror below sends up at it (sky off; the sun itself reaches only the
    // square's back). Diffuse radiance under a collimated beam of direction wi is ρ/π · E · (wi·ns):
    // the image over that of the same square flat is (wi·ns(p)) / (wi·n) at each point p. For SPPM
    // the beam is photon light, gathered at the visible points; particles connected to the eye
    // carry it with the adjoint correction. Both must draw the ratio exactly.
    if (!gpuAvailable("sppm")) return;
    const Vec3 corners[4] = {{0, -0.5, -0.5}, {0, 0.5, -0.5}, {0, 0.5, 0.5}, {0, -0.5, 0.5}};
    const Vec3 leans[4] = {normalize(Vec3(1, -0.5, -0.3)), normalize(Vec3(1, 0.6, -0.2)), normalize(Vec3(1, 0.2, 0.5)),
                           normalize(Vec3(1, -0.4, 0.4))};
    std::ostringstream obj;
    obj.precision(17);
    for (const Vec3& v : corners) obj << "v " << v.x << ' ' << v.y << ' ' << v.z << '\n';
    for (const Vec3& n : leans) obj << "vn " << n.x << ' ' << n.y << ' ' << n.z << '\n';
    obj << "f 1//1 2//2 3//3\nf 1//1 3//3 4//4\n";
    const std::string file = writeTemp("owe_test_leaning_square.obj", obj.str());
    // Sun 85° high from −x (behind the square); the mirror at (2, 0, −3) sends it along r to the
    // square's centre.
    const double el = 85 * Pi / 180;
    const Vec3 toSun{-std::cos(el), 0, std::sin(el)}, r = normalize(Vec3(-2, 0, 3));
    const Vec3 mirrorAxis = normalize(r + toSun);
    const Vec3 wi = -r;
    auto scene = [&](const std::string& normals) {
        char text[1600];
        std::snprintf(text, sizeof text, R"(
            units = m
            world { sky = none  sun { elevation = 85deg  azimuth = 270deg  luminance = 2e4 } }
            material paper { type = diffuse  reflectance = 0.8 }
            material coating { type = mirror  reflectance = 0.9 }
            body Square { type = mesh  file = "%s"  %s  position = (0, 0, 0)  material = paper }
            body Mirror { type = flat_mirror  size = (2.5, 2.5)  position = (2, 0, -3)  axis = (%.17g, %.17g, %.17g)
                          up = (0, 1, 0)  material = coating }
            observer Eye { position = (4, 0, 0)  look_at = (0, 0, 0)  up = (0, 0, 1)  fov = 16deg  pupil = 0.5mm  focus = 4
                           resolution = (64, 64) }
            render { detector = Eye }
        )", file.c_str(), normals.c_str(), mirrorAxis.x, mirrorAxis.y, mirrorAxis.z);
        return loadSceneFromString(text);
    };
    const Scene smooth = scene("normals = file"), flat = scene("");
    const auto& eye = dynamic_cast<const IdealObserver&>(*smooth.detectors[0]);
    // The expected ratio at each pixel's centre: the ray to the plane x = 0, the corner normals
    // interpolated on its triangle.
    const int W = 64, H = 64;
    std::vector<double> expected(size_t(W) * H, -1);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const double sx = (2 * (x + 0.5) / W - 1) * eye.tanX(), sy = (1 - 2 * (y + 0.5) / H) * eye.tanY();
            const Vec3 d = eye.right() * sx + eye.upVec() * sy + eye.forward();
            const Vec3 p = eye.position + d * (-eye.position.x / d.x);
            const double u = p.y + 0.5, v = p.z + 0.5;  // square coordinates in [0, 1]²
            if (u < 0.03 || u > 0.97 || v < 0.03 || v > 0.97 || std::abs(u - v) < 0.03) continue;  // edges and diagonal
            // Triangle (1, 2, 3) is u ≥ v: corners (0,0), (1,0), (1,1); triangle (1, 3, 4) the other half.
            Vec3 n = u >= v ? leans[0] * (1 - u) + leans[1] * (u - v) + leans[2] * v
                            : leans[0] * (1 - v) + leans[2] * u + leans[3] * (v - u);
            expected[size_t(y) * W + size_t(x)] = dot(wi, normalize(n)) / dot(wi, Vec3(1, 0, 0));
        }
    for (const char* integrator : {"sppm", "light"}) {
        const int n = std::string(integrator) == "sppm" ? 2048 : 4096;
        const Image a = renderGpu(smooth, integrator, n), b = renderGpu(flat, integrator, n);
        // 8 × 8 blocks of pixels: measured and expected ratio of the block sums.
        double worst = 0, sumA = 0, sumE = 0;
        for (int by = 0; by < 8; ++by)
            for (int bx = 0; bx < 8; ++bx) {
                double sa = 0, sb = 0, se = 0;
                int count = 0;
                for (int y = by * 8; y < by * 8 + 8; ++y)
                    for (int x = bx * 8; x < bx * 8 + 8; ++x) {
                        const double e = expected[size_t(y) * W + size_t(x)];
                        if (e < 0) continue;
                        sa += a.at(x, y).y;
                        sb += b.at(x, y).y;
                        se += e * b.at(x, y).y;
                        ++count;
                    }
                if (count < 32 || sb <= 0) continue;
                worst = std::max(worst, std::abs(sa - se) / sb);
                sumA += sa;
                sumE += se;
            }
        std::fprintf(stderr, "    %s: shaded over flat against (wi·ns)/(wi·n): total %.4f, worst block off by %.4f\n", integrator,
                     sumA / sumE, worst);
        CHECK_NEAR(sumA / sumE, 1.0, 0.02);
        CHECK(worst < 0.05);
    }
}

TEST(shading_smooth_glass_mesh_refracts_like_the_glass_ball) {
    // Specular surfaces too: a tessellated N-BK7 ball with smooth normals refracts and reflects
    // about its interpolated normal, so its caustic and the view through it are the analytic ball's
    // up to the polygonal silhouette; flat, every facet throws its own caustic. Samples where the
    // shading and the geometric surface disagree are dropped, so no region is ever crossed wrongly.
    if (!gpuAvailable("sppm")) return;
    const std::string obj = writeTemp("owe_test_glass_ball.obj", uvSphereObj(0.2, 32, 16, false));
    auto scene = [&](const std::string& ball) {
        return loadSceneFromString(R"(
            units = m
            world { sky = uniform(rgb(1, 1, 1, luminance = 0.05))  sun { elevation = 60deg  azimuth = 200deg  luminance = 2e4 } }
            material ground { type = diffuse  reflectance = 0.6 }
            body Floor { type = sheet  size = (4, 4)  position = (0, 0, 0)  material = ground }
            body Ball { )" + ball + R"(  medium = N-BK7  position = (0, 0, 0.35) }
            observer Eye { position = (0, -1.4, 1.0)  look_at = (0, 0.1, 0.15)  up = (0, 0, 1)  fov = 30deg  pupil = 1mm
                           focus = 1.6  resolution = (96, 96) }
            render { detector = Eye }
        )");
    };
    const Image exact = renderGpu(scene("type = sphere  radius = 0.2"), "sppm", 2048);
    const Image faceted = renderGpu(scene("type = mesh  file = \"" + obj + "\""), "sppm", 2048);
    TransportStats st;
    const Image smooth = renderGpu(scene("type = mesh  file = \"" + obj + "\"  normals = smooth"), "sppm", 2048, &st);
    // Relative L1 of 4 × 4 pixel block sums (the noise of single pixels averaged down).
    auto blockL1 = [&](const Image& a) {
        double num = 0, den = 0;
        for (int by = 0; by < 24; ++by)
            for (int bx = 0; bx < 24; ++bx) {
                double sa = 0, se = 0;
                for (int y = by * 4; y < by * 4 + 4; ++y)
                    for (int x = bx * 4; x < bx * 4 + 4; ++x) {
                        sa += a.at(x, y).y;
                        se += exact.at(x, y).y;
                    }
                num += std::abs(sa - se);
                den += se;
            }
        return num / den;
    };
    const double eFlat = blockL1(faceted), eSmooth = blockL1(smooth);
    std::fprintf(stderr, "    against the glass ball (block L1): flat %.4f, smooth %.4f; smooth inconsistencies %llu\n", eFlat,
                 eSmooth, (unsigned long long)st.inconsistencies);
    CHECK(eSmooth < 0.5 * eFlat);
    CHECK(st.inconsistencies == 0);
}

TEST(shading_material_map_gives_each_texture_component_its_roughness) {
    // A silver sheet whose checker texture is a material map, polished (α 0.02) on one set of
    // squares and rough (α 0.5) on the other, seen near the sun's reflection: on each set of squares
    // it must look exactly like a sheet that is all polished or all rough. The squares are told
    // apart by a diffuse white and black checker of the same pattern.
    if (!gpuAvailable("path")) return;
    auto scene = [&](const std::string& material) {
        return loadSceneFromString(R"(
            units = m
            world { sky = uniform(rgb(1, 1, 1, luminance = 0.05))  sun { elevation = 40deg  azimuth = 0deg  luminance = 2e4 } }
            material sheet { )" + material + R"( }
            body Sheet { type = sheet  size = (2, 2)  position = (0, 0, 0)  material = sheet }
            observer Eye { position = (0, -1.2, 1.6)  look_at = (0, 0.1, 0)  up = (0, 0, 1)  fov = 30deg  pupil = 1mm  focus = 2
                           resolution = (96, 96) }
            render { detector = Eye }
        )");
    };
    const Image polished = renderGpu(scene("type = conductor  metal = silver  roughness = 0.02"), "path", 1024);
    const Image rough = renderGpu(scene("type = conductor  metal = silver  roughness = 0.5"), "path", 1024);
    const Image mapped = renderGpu(
        scene("type = conductor  metal = silver  texture = checker(a = 1, b = 1, scale = 0.25)  roughness = (0.02, 0.5)"), "path",
        1024);
    const Image cells = renderGpu(scene("type = diffuse  texture = checker(a = 0.9, b = 0.05, scale = 0.25)"), "path", 64);
    // Each pixel's square (a bright, b dark), and only pixels whose 3 × 3 neighbourhood is one square.
    const int W = 96, H = 96;
    double threshold = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) threshold += cells.at(x, y).y / (W * H);
    auto isA = [&](int x, int y) { return cells.at(x, y).y > threshold; };
    double sums[2][3] = {};  // [square][mapped, polished, rough]
    int counts[2] = {};
    for (int y = 1; y + 1 < H; ++y)
        for (int x = 1; x + 1 < W; ++x) {
            const bool a = isA(x, y);
            bool interior = true;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) interior = interior && isA(x + dx, y + dy) == a;
            if (!interior) continue;
            const int k = a ? 0 : 1;
            sums[k][0] += mapped.at(x, y).y;
            sums[k][1] += polished.at(x, y).y;
            sums[k][2] += rough.at(x, y).y;
            ++counts[k];
        }
    const double onA = sums[0][0] / sums[0][1], onB = sums[1][0] / sums[1][2];
    const double contrastA = sums[0][2] / sums[0][1], contrastB = sums[1][1] / sums[1][2];
    std::fprintf(stderr, "    polished squares (%d px): map / polished %.4f (rough would be %.3f); rough squares (%d px): map / rough %.4f "
                 "(polished would be %.3f)\n", counts[0], onA, contrastA, counts[1], onB, contrastB);
    CHECK(counts[0] > 500 && counts[1] > 500);
    CHECK_NEAR(onA, 1.0, 0.02);
    CHECK_NEAR(onB, 1.0, 0.02);
    CHECK(std::abs(contrastA - 1) > 0.2 || std::abs(contrastB - 1) > 0.2);  // the two finishes do differ here
}

TEST(shading_relief_tilts_the_normal_by_its_slope) {
    // A white sheet with a relief (height depth·fbm(p / scale)), lit only by a 30° sun from +x and
    // seen from straight above: at each point its radiance over the flat sheet's is (ns·s) / (n·s),
    // with ns the normal tilted by the relief's gradient. The gradient here is a finite difference
    // of the reference's own noise (the kernel's is analytic): 4 × 4 pixel blocks must agree.
    if (!gpuAvailable("path")) return;
    const double scale = 0.010, depth = 0.002;
    auto scene = [&](const std::string& relief) {
        return loadSceneFromString(R"(
            units = m
            world { sky = none  sun { elevation = 30deg  azimuth = 90deg  luminance = 2e4 } }
            material paper { type = diffuse  reflectance = 0.8  )" + relief + R"( }
            body Sheet { type = sheet  size = (0.4, 0.4)  position = (0, 0, 0)  material = paper }
            observer Eye { position = (0, 0, 1)  look_at = (0, 0, 0)  up = (0, 1, 0)  fov = 10deg  pupil = 0.1mm  focus = 1
                           resolution = (128, 128) }
            render { detector = Eye }
        )");
    };
    const Scene bumpy = scene("relief = noise(scale = 10mm, depth = 2mm, octaves = 2)");
    const Image a = renderGpu(bumpy, "path", 256), b = renderGpu(scene(""), "path", 256);
    const auto& eye = dynamic_cast<const IdealObserver&>(*bumpy.detectors[0]);
    const Vec3 s{std::cos(Pi / 6), 0, std::sin(Pi / 6)};  // toward the sun (azimuth 90°: +x)
    auto height = [&](const Vec3& p) { return depth * fbm3(p / scale, 2); };
    auto expected = [&](double px, double py) {  // at image position (px, py)
        const int W = 128, H = 128;
        const double sx = (2 * px / W - 1) * eye.tanX(), sy = (1 - 2 * py / H) * eye.tanY();
        const Vec3 d = eye.right() * sx + eye.upVec() * sy + eye.forward();
        const Vec3 p = eye.position + d * (-eye.position.z / d.z);
        const double e = 1e-6;
        const double gx = (height(p + Vec3(e, 0, 0)) - height(p - Vec3(e, 0, 0))) / (2 * e);
        const double gy = (height(p + Vec3(0, e, 0)) - height(p - Vec3(0, e, 0))) / (2 * e);
        return dot(normalize(Vec3(-gx, -gy, 1)), s) / s.z;
    };
    double worst = 0, spread = 0, n = 0;
    for (int by = 0; by < 32; ++by)
        for (int bx = 0; bx < 32; ++bx) {
            double sa = 0, sb = 0, se = 0;
            for (int y = by * 4; y < by * 4 + 4; ++y)
                for (int x = bx * 4; x < bx * 4 + 4; ++x) {
                    double ex = 0;
                    for (int k = 0; k < 16; ++k) ex += expected(x + (k % 4 + 0.5) / 4, y + (k / 4 + 0.5) / 4) / 16;
                    sa += a.at(x, y).y;
                    sb += b.at(x, y).y;
                    se += ex * b.at(x, y).y;
                }
            worst = std::max(worst, std::abs(sa - se) / sb);
            spread += std::abs(se / sb - 1);
            ++n;
        }
    std::fprintf(stderr, "    relief shading against (ns·s)/(n·s): worst 4 × 4 block off by %.4f (the relief varies it by %.3f on average)\n",
                 worst, spread / n);
    CHECK(spread / n > 0.05);  // the relief is visible
    CHECK(worst < 0.02);
}
