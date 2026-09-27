// Standalone modelling bridge. Does not participate in rendering or transport.
// Build: cmake --build build --target export_model_preview
#include "owe/loader/scene_loader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
using namespace owe;

MeshData tessellate(const Shape& shape) {
    MeshData mesh;
    if (auto m = dynamic_cast<const MeshShape*>(&shape)) {
        mesh.positions = m->positions(); mesh.triangles = m->triangles(); return mesh;
    }
    auto box = shape.bounds();
    const int N = 96;
    if (auto p = dynamic_cast<const PlaneShape*>(&shape)) {
        if (p->aperture() == PlaneShape::Aperture::Rect) {
            mesh.addQuad({-p->a(),-p->b(),0},{p->a(),-p->b(),0},{p->a(),p->b(),0},{-p->a(),p->b(),0});
        } else {
            double inner = p->aperture() == PlaneShape::Aperture::Disk ? p->b() : 0;
            double ry = p->aperture() == PlaneShape::Aperture::Ellipse ? p->b() : p->a();
            for(int i=0;i<N;i++) {
                double a=2*Pi*i/N,b=2*Pi*(i+1)/N;
                mesh.addQuad({inner*cos(a),inner*sin(a),0},{p->a()*cos(a),ry*sin(a),0},
                             {p->a()*cos(b),ry*sin(b),0},{inner*cos(b),inner*sin(b),0});
            }
        }
        return mesh;
    }
    bool sphere = dynamic_cast<const SphereShape*>(&shape), dome = dynamic_cast<const DomeShape*>(&shape);
    bool cylinder = dynamic_cast<const CylinderShape*>(&shape), wall = dynamic_cast<const RoundWallShape*>(&shape);
    auto sag = dynamic_cast<const SagSurface*>(&shape);
    int rows = (sphere || dome) ? 48 : (wall ? 64 : (sag ? 12 : 1));
    auto point = [&](double u,double v) -> Vec3 {
        double a=2*Pi*u;
        if (sphere || dome) {
            double e=(dome ? 0 : -Pi/2)+v*(dome ? Pi/2 : Pi);
            double r=box.hi.x; return {r*cos(e)*cos(a),r*cos(e)*sin(a),r*sin(e)};
        }
        if (sag) {double r=sag->rMin()+v*(sag->rMax()-sag->rMin()); return {r*cos(a),r*sin(a),sag->sag(r)};}
        return {box.hi.x*cos(a),box.hi.x*sin(a),box.lo.z+v*(box.hi.z-box.lo.z)};
    };
    if (!(sphere||dome||cylinder||wall||sag)) return mesh;
    for(int j=0;j<rows;j++) for(int i=0;i<N;i++) {
        Vec3 a=point(double(i)/N,double(j)/rows), b=point(double(i+1)/N,double(j)/rows);
        Vec3 c=point(double(i+1)/N,double(j+1)/rows), d=point(double(i)/N,double(j+1)/rows);
        if(dome||wall) {
            // Clip each review polygon at the true analytic opening, instead
            // of discarding whole grid cells (which creates stair-stepped slits).
            using UV=std::array<double,2>;
            std::array<UV,4> uv={UV{double(i)/N,double(j)/rows},UV{double(i+1)/N,double(j)/rows},
                                UV{double(i+1)/N,double(j+1)/rows},UV{double(i)/N,double(j+1)/rows}};
            auto solid=[&](UV t) {
                Vec3 p=point(t[0],t[1]);
                Vec3 direction=normalize(dome?p:Vec3{p.x,p.y,0});
                LocalHit hit;return shape.intersect({p+direction,-direction},0,2,hit);
            };
            std::vector<Vec3> polygon;
            for(int edge=0;edge<4;edge++) {
                auto from=uv[edge],to=uv[(edge+1)%4];
                bool inside=solid(from),next=solid(to);
                if(inside) polygon.push_back(point(from[0],from[1]));
                if(inside!=next) {
                    for(int step=0;step<26;step++) {
                        UV mid={(from[0]+to[0])*.5,(from[1]+to[1])*.5};
                        if(solid(mid)==inside) from=mid;else to=mid;
                    }
                    polygon.push_back(point((from[0]+to[0])*.5,(from[1]+to[1])*.5));
                }
            }
            for(size_t k=1;k+1<polygon.size();k++)
                if(length(cross(polygon[k]-polygon[0],polygon[k+1]-polygon[0]))>1e-14)
                    mesh.addTriangle(polygon[0],polygon[k],polygon[k+1]);
        } else mesh.addQuad(a,b,c,d);
    }
    // Shared vertices let the OBJ smoothing flag represent analytic curvature.
    std::map<std::array<long long,3>,uint32_t> shared;
    std::vector<Vec3> positions;
    std::vector<uint32_t> indices;
    for(auto p:mesh.positions) {
        std::array<long long,3> key={std::llround(p.x*1e9),std::llround(p.y*1e9),std::llround(p.z*1e9)};
        auto [it,inserted]=shared.emplace(key,uint32_t(positions.size()));
        if(inserted) positions.push_back(p);
        indices.push_back(it->second);
    }
    for(auto& t:mesh.triangles) for(auto& index:t) index=indices[index];
    mesh.positions=std::move(positions);
    return mesh;
}

int main(int argc,char** argv) try {
    if(argc!=3) {std::cerr<<"usage: export_model_preview scene.owe prefix\n";return 2;}
    Scene scene=loadScene(argv[1]);
    std::filesystem::path prefix(argv[2]);
    std::filesystem::create_directories(prefix.parent_path());
    std::ofstream obj(prefix.string()+".obj.tmp"),mtl(prefix.string()+".mtl.tmp");
    obj<<std::setprecision(9)<<"mtllib "<<prefix.filename().string()<<".mtl\n";
    for(size_t i=0;i<scene.world.optics().size();++i) {
        const auto& o=scene.world.optics()[i];
        double r=o.albedo({}, {0,0,1},650),g=o.albedo({}, {0,0,1},550),b=o.albedo({}, {0,0,1},450);
        if(o.type==SurfaceType::Conductor) {r=.58;g=.38;b=.12;}
        if(o.type==SurfaceType::Absorber) r=g=b=.015;
        mtl<<"newmtl m"<<i<<"\nKd "<<r<<' '<<g<<' '<<b<<"\nPr 0.65\n";
        if(o.type==SurfaceType::Conductor) mtl<<"Pm 0.8\nPr "<<o.roughness<<"\n";
    }
    size_t offset=1,faces=0;
    for(const auto& boundary:scene.world.boundaries()) {
        const auto& body=scene.world.bodies()[boundary.body];
        if(body.name=="Earth" || length(boundary.toWorld.t)>10000 || body.kind=="starfield") continue;
        if(scene.world.optics()[boundary.optics].type==SurfaceType::Null) continue;
        if(body.name=="Ground" && body.kind=="terrain" && scene.world.optics()[boundary.optics].type==SurfaceType::Dielectric) continue;
        auto mesh=tessellate(*boundary.shape);
        if(mesh.triangles.empty()) continue;
        obj<<"o "<<body.name<<'.'<<boundary.name<<"\nusemtl m"<<boundary.optics<<'\n';
        // Analytic cylinders/spheres are smooth in OWE. Preserve that appearance
        // in the tessellated review without changing authored mesh normals.
        obj<<"s "<<(dynamic_cast<const MeshShape*>(boundary.shape.get()) ? "off" : "1")<<'\n';
        for(auto p:mesh.positions) {p=boundary.toWorld.point(p);obj<<"v "<<p.x<<' '<<p.y<<' '<<p.z<<'\n';}
        for(auto t:mesh.triangles) obj<<"f "<<offset+t[0]<<' '<<offset+t[1]<<' '<<offset+t[2]<<'\n';
        offset+=mesh.positions.size();faces+=mesh.triangles.size();
    }
    std::cout<<prefix<<": "<<faces<<" preview triangles\n";
    std::ofstream cameras(prefix.string()+".cameras.tmp");
    for(const auto& d:scene.detectors) if(auto eye=dynamic_cast<const IdealObserver*>(d.get())) {
        if(length(eye->lookAt)>10000) continue;
        cameras<<eye->name<<' '<<eye->position.x<<' '<<eye->position.y<<' '<<eye->position.z<<' '
               <<eye->lookAt.x<<' '<<eye->lookAt.y<<' '<<eye->lookAt.z<<' '<<eye->fovY<<'\n';
    }
    obj.close();mtl.close();cameras.close();
    if(!obj || !mtl || !cameras) throw std::runtime_error("preview output write failed");
    for(const std::string ext:{".obj",".mtl",".cameras"})
        std::filesystem::rename(prefix.string()+ext+".tmp",prefix.string()+ext);
    // Publish only after all companion files are complete, for the live watcher.
    std::ofstream ready(prefix.string()+".ready.tmp");ready<<faces<<'\n';ready.close();
    std::filesystem::rename(prefix.string()+".ready.tmp",prefix.string()+".ready");
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
