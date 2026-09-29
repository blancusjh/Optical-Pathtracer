cmake_minimum_required(VERSION 3.20)
project(OpticalWorldEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

option(OWE_GPU "Portable GPU backend: Vulkan compute with Slang kernels (MoltenVK on macOS)" OFF)
option(OWE_VIEWER "Interactive scene browser (SDL3 and Dear ImGui)" OFF)

# The reference tracer relies on strict IEEE-754 double semantics: never -ffast-math.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------- the engine
# The world model, the reference transport, analysis, the .owe loader and the backend-neutral
# rendering interface. It names no backend. (Layer rules: cmake/CheckLayers.cmake.)
add_library(owe_engine STATIC
  src/owe/core/spectrum.cpp
  src/owe/core/medium.cpp
  src/owe/scene/bvh.cpp
  src/owe/scene/geometry.cpp
  src/owe/scene/gltf.cpp
  src/owe/scene/world.cpp
  src/owe/scene/detector.cpp
  src/owe/scene/image.cpp
  src/owe/scene/builders.cpp
  src/owe/scene/prescription.cpp
  src/owe/transport/scattering.cpp
  src/owe/transport/transport.cpp
  src/owe/analysis/analysis.cpp
  src/owe/analysis/inspect.cpp
  src/owe/loader/scene_parser.cpp
  src/owe/loader/scene_loader.cpp
  src/owe/render/exposure.cpp
  src/owe/render/output.cpp
  src/owe/render/record.cpp
)
target_include_directories(owe_engine PUBLIC src)
target_link_libraries(owe_engine PUBLIC Threads::Threads)

# Image decoding for textures and environment maps (PNG, JPEG, Radiance HDR): stb_image, pinned.
include(FetchContent)
FetchContent_Declare(owe_stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
  SOURCE_SUBDIR _none)
FetchContent_MakeAvailable(owe_stb)
target_include_directories(owe_engine PRIVATE ${owe_stb_SOURCE_DIR})

# glTF 2.0 models (textured scans; Sketchfab's own format): cgltf, pinned (v1.15).
FetchContent_Declare(owe_cgltf
  GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
  GIT_TAG 360db1a95480fe102ae9c69b27c5d101167ff5ba
  SOURCE_SUBDIR _none)
FetchContent_MakeAvailable(owe_cgltf)
target_include_directories(owe_engine PRIVATE ${owe_cgltf_SOURCE_DIR})

# Optional: compressed PNG output. Without zlib, PNGs use stored deflate blocks.
find_package(ZLIB)
if(ZLIB_FOUND)
  target_link_libraries(owe_engine PRIVATE ZLIB::ZLIB)
  target_compile_definitions(owe_engine PRIVATE OWE_HAVE_ZLIB=1)
endif()

# ---------------------------------------------------------------- backends
# Each backend depends on the engine only, never on another backend.

# cpu: the reference transport on host threads.
add_library(owe_backend_cpu STATIC src/owe/backends/cpu/cpu_backend.cpp)
target_link_libraries(owe_backend_cpu PUBLIC owe_engine)

# gpu: Vulkan compute with Slang kernels. Scene flattening needs no Vulkan and is always built
# (its layout is tested everywhere); without OWE_GPU the backend reports itself unavailable.
add_library(owe_backend_gpu STATIC src/owe/backends/gpu/gpu_scene.cpp)
target_link_libraries(owe_backend_gpu PUBLIC owe_engine)
if(OWE_GPU)
  include(cmake/OweGpu.cmake)
  target_sources(owe_backend_gpu PRIVATE
    src/owe/backends/gpu/gpu_accel.cpp
    src/owe/backends/gpu/gpu_backend.cpp
    src/owe/backends/gpu/gpu_renderer.cpp
    src/owe/backends/gpu/vulkan_context.cpp)
  target_link_libraries(owe_backend_gpu PRIVATE owe_volk)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Vulkan structs are initialised as {VK_STRUCTURE_TYPE_...}: the remaining members are zero.
    set_source_files_properties(src/owe/backends/gpu/vulkan_context.cpp PROPERTIES COMPILE_OPTIONS -Wno-missing-field-initializers)
  endif()
  set(_kernels ${CMAKE_SOURCE_DIR}/src/owe/backends/gpu/shaders)
  owe_add_kernel(owe_backend_gpu ${_kernels}/path_kernel.slang owe_kernel_path)
  owe_add_kernel(owe_backend_gpu ${_kernels}/particle_kernel.slang owe_kernel_particle)
  owe_add_kernel(owe_backend_gpu ${_kernels}/path_kernel.slang owe_kernel_path_rq RAY_QUERY)
  owe_add_kernel(owe_backend_gpu ${_kernels}/particle_kernel.slang owe_kernel_particle_rq RAY_QUERY)
  owe_add_kernel(owe_backend_gpu ${_kernels}/sppm_camera_kernel.slang owe_kernel_sppm_camera)
  owe_add_kernel(owe_backend_gpu ${_kernels}/sppm_camera_kernel.slang owe_kernel_sppm_camera_rq RAY_QUERY)
  owe_add_kernel(owe_backend_gpu ${_kernels}/sppm_photon_kernel.slang owe_kernel_sppm_photon)
  owe_add_kernel(owe_backend_gpu ${_kernels}/sppm_photon_kernel.slang owe_kernel_sppm_photon_rq RAY_QUERY)
  owe_add_kernel(owe_backend_gpu ${_kernels}/sppm_grid_kernel.slang owe_kernel_sppm_grid)
  owe_add_kernel(owe_backend_gpu ${_kernels}/sppm_update_kernel.slang owe_kernel_sppm_update)
  owe_add_kernel(owe_backend_gpu ${_kernels}/sppm_guide_kernel.slang owe_kernel_sppm_guide)
  owe_add_kernel(owe_backend_gpu ${_kernels}/vcm_light_kernel.slang owe_kernel_vcm_light)
  owe_add_kernel(owe_backend_gpu ${_kernels}/vcm_light_kernel.slang owe_kernel_vcm_light_rq RAY_QUERY)
  owe_add_kernel(owe_backend_gpu ${_kernels}/vcm_camera_kernel.slang owe_kernel_vcm_camera)
  owe_add_kernel(owe_backend_gpu ${_kernels}/vcm_camera_kernel.slang owe_kernel_vcm_camera_rq RAY_QUERY)
  owe_add_kernel(owe_backend_gpu ${_kernels}/vcm_grid_kernel.slang owe_kernel_vcm_grid)
else()
  target_sources(owe_backend_gpu PRIVATE src/owe/backends/gpu/gpu_unavailable.cpp)
endif()

# The registry (the only code naming concrete backends) and cross-backend comparison.
add_library(owe STATIC src/owe/backends/registry.cpp src/owe/backends/compare.cpp)
target_link_libraries(owe PUBLIC owe_engine owe_backend_cpu owe_backend_gpu)

# ---------------------------------------------------------------- applications
add_executable(owe_cli apps/owe.cpp)
set_target_properties(owe_cli PROPERTIES OUTPUT_NAME owe)
target_link_libraries(owe_cli PRIVATE owe)
if(OWE_VIEWER)
  include(cmake/OweViewer.cmake)
else()
  target_sources(owe_cli PRIVATE apps/viewer_unavailable.cpp)
endif()

# Model-authoring helpers (tools/), built on demand: cmake --build build --target <name>.
foreach(tool check_scene_models export_model_preview sample_landscape)
  add_executable(${tool} EXCLUDE_FROM_ALL tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE owe)
endforeach()

# ---------------------------------------------------------------- tests
enable_testing()
add_executable(owe_tests
  tests/test_main.cpp
  tests/test_optics.cpp
  tests/test_geometry.cpp
  tests/test_transport.cpp
  tests/test_instruments.cpp
  tests/test_scene.cpp
  tests/test_render.cpp
  tests/test_backends.cpp
  tests/test_gpu.cpp
  tests/test_navigation.cpp
  tests/test_references.cpp
  tests/test_shading.cpp
  tests/test_vcm.cpp
  tests/test_models.cpp
)
target_link_libraries(owe_tests PRIVATE owe)
add_test(NAME owe_tests COMMAND owe_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
if(OWE_GPU)
  # The conformance suite again with the portable software traversal forced: on a device with
  # ray-tracing hardware, owe_tests covers the hardware traversal and this covers the other path.
  add_test(NAME backends_software_traversal COMMAND owe_tests backends_ WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(backends_software_traversal PROPERTIES ENVIRONMENT "OWE_GPU_RAY_QUERY=0")
endif()
add_test(NAME layering COMMAND ${CMAKE_COMMAND} -DROOT=${CMAKE_SOURCE_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/CheckLayers.cmake)
if(OWE_VIEWER)
  if(ZLIB_FOUND)
    add_executable(owe_viewer_display_tests tests/viewer_display.cpp apps/viewer.cpp)
    target_link_libraries(owe_viewer_display_tests PRIVATE owe owe_imgui ZLIB::ZLIB)
    add_test(NAME viewer_display COMMAND owe_viewer_display_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(viewer_display PROPERTIES TIMEOUT 45
      ENVIRONMENT "SDL_VIDEODRIVER=dummy;SDL_RENDER_DRIVER=software")
  endif()
  add_executable(owe_viewer_navigation_tests tests/viewer_navigation.cpp)
  target_link_libraries(owe_viewer_navigation_tests PRIVATE owe owe_imgui)
  add_test(NAME viewer_navigation COMMAND owe_viewer_navigation_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(viewer_navigation PROPERTIES TIMEOUT 120)
  add_test(NAME viewer_smoke COMMAND ${CMAKE_COMMAND}
    -DOWE=$<TARGET_FILE:owe_cli> -DSOURCE=${CMAKE_SOURCE_DIR} -DOUTPUT=${CMAKE_BINARY_DIR}/viewer-smoke
    -P ${CMAKE_SOURCE_DIR}/tests/viewer_smoke.cmake)
  set_tests_properties(viewer_smoke PROPERTIES TIMEOUT 60)
endif()
