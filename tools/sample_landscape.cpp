// Model-authoring helper: sample the existing terrain, without changing its builder.
#include "owe/scene/builders.hpp"
#include "owe/loader/scene_parser.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace owe;
int main(int argc,char** argv) try {
    if(argc!=3) throw std::runtime_error("usage: sample_landscape scene.owe output-prefix");
    std::ifstream input(argv[1]);std::stringstream contents;contents<<input.rdbuf();
    auto doc=parseSceneText(contents.str(),argv[1]);
    ValuePtr ground;
    for(auto b:doc->items) if(b->str=="body" && b->name=="Ground") ground=b;
    if(!ground) throw std::runtime_error("Ground terrain not found");
    auto number=[&](const char* key,double fallback) {auto v=ground->get(key);return v?v->num:fallback;};
    TerrainSpec spec;
    auto size=ground->get("size");spec.sizeX=size->items[0]->num;spec.sizeY=size->items[1]->num;
    spec.resolution=int(number("resolution",256));spec.amplitude=number("amplitude",30);
    spec.featureSize=number("feature",80);spec.ridge=number("ridge",0);spec.seed=uint64_t(number("seed",1));
    spec.flatRadius=number("flat_radius",0);
    auto mesh=makeTerrain(spec);
    std::filesystem::path prefix(argv[2]);std::filesystem::create_directories(prefix.parent_path());
    std::ofstream data(prefix.string()+".heights",std::ios::binary);
    for(auto p:mesh.positions) data.write(reinterpret_cast<const char*>(&p.z),sizeof(double));
    std::ofstream meta(prefix.string()+".json");
    meta<<std::setprecision(17)<<"{\"resolution\":"<<spec.resolution<<",\"size\":["<<spec.sizeX<<","<<spec.sizeY
        <<"],\"origin_height\":"<<terrainHeight(spec,0,0)<<",\"byte_order\":\"native\"}\n";
    data.close();meta.close();
    if(!data || !meta) throw std::runtime_error("height cache write failed");
    std::cout<<"Sampled "<<argv[1]<<": "<<mesh.positions.size()<<" support heights\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
