# The dependency rules of the source tree, checked on every `#include "owe/..."`:
#
#   core                 math, sampling, spectra, media
#   scene                the world model: geometry, world, detectors, builders, prescriptions
#   transport            the reference transport (IEEE-754 double): optics, scattering, Tracer
#   analysis             lens diagnostics and path inspection, built on reference walks
#   loader               the .owe language (a physical camera's real focus uses analysis)
#   render               the backend-neutral interface: Renderer, Backend, image output, records
#   backends/cpu         runs the reference transport on host threads
#   backends/gpu         a float32 port on Vulkan; it never includes the reference transport
#   backends             the registry (the only code naming concrete backends) and comparison
#   apps                 front ends: they reach a backend only through the registry
#
# Backends never include each other. Usage: cmake -DROOT=<repository> -P CheckLayers.cmake
set(rule_core                 core/)
set(rule_scene                core/ scene/)
set(rule_transport            core/ scene/ transport/)
set(rule_analysis             core/ scene/ transport/ analysis/)
set(rule_loader               core/ scene/ transport/ analysis/ loader/)
set(rule_render               core/ scene/ render/)
set(rule_backends/cpu         core/ scene/ render/ transport/ backends/cpu/)
set(rule_backends/gpu         core/ scene/ render/ backends/gpu/)
set(rule_backends             core/ scene/ render/ backends/registry.hpp backends/compare.hpp
                              backends/cpu/cpu_backend.hpp backends/gpu/gpu_backend.hpp)
set(rule_apps                 core/ scene/ transport/ analysis/ loader/ render/
                              backends/registry.hpp backends/compare.hpp)

set(violations 0)
file(GLOB_RECURSE sources RELATIVE ${ROOT} ${ROOT}/src/owe/*.hpp ${ROOT}/src/owe/*.cpp ${ROOT}/apps/*.hpp ${ROOT}/apps/*.cpp)
foreach(file ${sources})
  get_filename_component(dir ${file} DIRECTORY)
  string(REGEX REPLACE "^src/owe/" "" layer "${dir}")
  if(NOT DEFINED rule_${layer})
    message(SEND_ERROR "${file}: directory '${layer}' has no layering rule (add one to cmake/CheckLayers.cmake)")
    math(EXPR violations "${violations} + 1")
    continue()
  endif()
  file(STRINGS ${ROOT}/${file} includes REGEX "^[ \t]*#[ \t]*include[ \t]*\"owe/")
  foreach(line ${includes})
    string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]*\"owe/([^\"]+)\".*" "\\1" target "${line}")
    set(ok FALSE)
    foreach(prefix ${rule_${layer}})
      string(FIND "${target}" "${prefix}" at)
      if(at EQUAL 0)
        set(ok TRUE)
        break()
      endif()
    endforeach()
    if(NOT ok)
      message(SEND_ERROR "${file}: '${layer}' may not include owe/${target}")
      math(EXPR violations "${violations} + 1")
    endif()
  endforeach()
endforeach()
list(LENGTH sources count)
if(violations GREATER 0)
  message(FATAL_ERROR "${violations} layering violation(s)")
endif()
message(STATUS "layering: ${count} files respect the dependency rules")
