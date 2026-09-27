// Model integration checks, independent of the tracing pipeline.
#include "owe/loader/scene_loader.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace owe;

int main() try {
    for(const auto& entry:std::filesystem::directory_iterator("scenes")) {
        if(entry.path().extension()!=".owe") continue;
        auto scene=loadScene(entry.path().string());
        size_t faces=0;
        for(const auto& b:scene.world.boundaries()) {
            const auto& name=scene.world.bodies()[b.body].name;
            if(name.rfind("Crafted_",0)!=0 && name.rfind("Landscape_",0)!=0) continue;
            auto mesh=dynamic_cast<const MeshShape*>(b.shape.get());
            if(!mesh || mesh->triangles().empty()) throw std::runtime_error("empty crafted mesh");
            for(auto t:mesh->triangles()) {
                auto a=mesh->positions().at(t[0]),c=mesh->positions().at(t[1]),d=mesh->positions().at(t[2]);
                double area=length(cross(c-a,d-a));
                if(!std::isfinite(area) || area<1e-15)
                    throw std::runtime_error("degenerate face in "+scene.world.bodies()[b.body].name+" indices "+std::to_string(t[0])+","+std::to_string(t[1])+","+std::to_string(t[2]));
            }
            faces+=mesh->triangleCount();
        }
        if(entry.path().stem()=="the_telescope") {
            auto xf=scene.world.assemblyToWorld(scene.world.findAssembly("Scope"));
            for(int i=0;i<32;i++) {
                double a=2*Pi*i/32;
                Ray ray{xf.point({.020*cos(a),.020*sin(a),-.05}),xf.vector({0,0,-1})};
                SurfaceHit hit;
                if(!scene.world.intersect(ray,800,hit) || scene.world.boundaryLabel(hit.boundary).find("Statue")==std::string::npos)
                    throw std::runtime_error("landscape blocks the telescope's target statue");
            }
            std::cout<<"Mountain telescope: 32 objective-to-statue rays clear\n";
        }
        if(entry.path().stem()=="the_telescope" || entry.path().stem()=="the_temple" || entry.path().stem()=="the_observatory") {
            Vec3 xy=entry.path().stem()=="the_telescope" ? Vec3{0,1,0} :
                    (entry.path().stem()=="the_temple" ? Vec3{-20,-10,0} : Vec3{8,0,0});
            // The coordinate terrain is reference-only (no matter); the relief is the ground.
            if(scene.world.findBody("Ground")>=0)
                throw std::runtime_error("coordinate terrain must be reference_only (no matter)");
            double z=-Inf;
            int body=scene.world.findBody("Landscape_Ground_relief");
            for(const auto& b:scene.world.boundaries()) if(b.body==body) {
                LocalHit h;
                if(b.shape->intersect({b.toLocal.point({xy.x,xy.y,10000}),b.toLocal.vector({0,0,-1})},0,20000,h))
                    z=std::max(z,b.toWorld.point(h.p).z);
            }
            if(!std::isfinite(z)) throw std::runtime_error("visible landscape relief missing");
            SurfaceHit hit;
            if(!scene.world.intersect({{xy.x,xy.y,z+.1},{0,0,-1}},.2,hit))
                throw std::runtime_error("visible landscape floor missing");
            auto type=scene.world.optics()[scene.world.boundaries()[hit.boundary].optics].type;
            if(type==SurfaceType::Null || type==SurfaceType::Dielectric)
                throw std::runtime_error("coordinate reference hides visible landscape ground");
        }
        if(entry.path().stem()=="the_observatory") {
            for(const std::string name:{"GreatRefractor","Refractor2","MoonScope"}) {
                auto xf=scene.world.assemblyToWorld(scene.world.findAssembly(name));
                double radius=name=="MoonScope" ? .023 : .070;
                for(int ring=0;ring<4;ring++) for(int i=0;i<32;i++) {
                    double a=2*Pi*i/32, r=radius*ring/3;
                    Ray ray{xf.point({r*cos(a),r*sin(a),-.05}),xf.vector({0,0,-1})};
                    SurfaceHit hit;
                    if(scene.world.intersect(ray,20,hit))
                        throw std::runtime_error(name+" sky aperture blocked by "+scene.world.boundaryLabel(hit.boundary));
                }
                std::cout<<name<<": 128 aperture rays clear\n";
            }
            // The floorboards must sit above terrain and below the original furniture.
            double base=scene.world.bodies()[scene.world.findBody("Floor")].xf.t.z+.048;
            for(Vec3 p: {Vec3{2,1,base+.01},Vec3{-2,0,base+.01},Vec3{1,-2,base+.01}}) {
                SurfaceHit hit;
                if(!scene.world.intersect({p,{0,0,-1}},.02,hit) || scene.world.boundaryLabel(hit.boundary).find("Oak_boards")==std::string::npos)
                    throw std::runtime_error("floorboard below ground or missing");
            }
        }
        std::cout<<entry.path().filename()<<": loaded, "<<faces<<" crafted triangles validated\n";
    }
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
