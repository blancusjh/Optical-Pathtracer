// Builds a Scene from the .owe language, and hosts the instrument builders
// (physical cameras) shared by the loader and the C++ API.
#pragma once

#include <map>
#include <string>

#include "prescription.hpp"
#include "scene.hpp"
#include "scene_parser.hpp"

namespace owe {

Scene loadScene(const std::string& path);
Scene loadSceneFromString(const std::string& text, const std::string& baseDir = ".", const std::string& origin = "<string>");

// A physical camera: lens assembly + housing + stop(s) + sensor surface. Nothing
// about it is special to the renderer; the sensor is matter in the world.
struct CameraParams {
    double sensorWidth = 0.036, sensorHeight = 0.024;
    int width = 480, height = 320;
    double focusDistance = Inf;  // object distance brought to focus (paraxially, at the design wavelength)
    double sensorShift = 0;      // additional sensor displacement along the axis
    bool housing = true;
};
// Local frame of the camera assembly: light travels +z (from the scene to the sensor),
// so the camera looks toward local −z. Returns the detector index.
int addPhysicalCamera(Scene& scene, const std::string& name, const Prescription& lens, const CameraParams& params,
                      const Transform& placement, int parentAssembly = -1);
// Placement whose local −z points from `position` toward `lookAt`, with local +y near `up`.
Transform cameraPlacement(const Vec3& position, const Vec3& lookAt, const Vec3& up);

}  // namespace owe
