cmake_minimum_required(VERSION 3.20)
project(OpticalWorldEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# The reference tracer relies on strict IEEE-754 double semantics: never -ffast-math.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

add_library(owe
  src/owe/spectrum.cpp
  src/owe/medium.cpp
  src/owe/bvh.cpp
  src/owe/geometry.cpp
  src/owe/world.cpp
  src/owe/scattering.cpp
  src/owe/detector.cpp
  src/owe/transport.cpp
  src/owe/render.cpp
  src/owe/builders.cpp
  src/owe/prescription.cpp
  src/owe/analysis.cpp
  src/owe/inspect.cpp
  src/owe/scene_parser.cpp
  src/owe/scene_loader.cpp
)
target_include_directories(owe PUBLIC src)
target_link_libraries(owe PUBLIC Threads::Threads)

# Optional: compressed PNG output. Without zlib, PNGs use stored deflate blocks.
find_package(ZLIB)
if(ZLIB_FOUND)
  target_link_libraries(owe PRIVATE ZLIB::ZLIB)
  target_compile_definitions(owe PRIVATE OWE_HAVE_ZLIB=1)
endif()

add_executable(owe_cli apps/owe.cpp)
set_target_properties(owe_cli PROPERTIES OUTPUT_NAME owe)
target_link_libraries(owe_cli PRIVATE owe)

enable_testing()
add_executable(owe_tests
  tests/test_main.cpp
  tests/test_optics.cpp
  tests/test_geometry.cpp
  tests/test_transport.cpp
  tests/test_instruments.cpp
  tests/test_scene.cpp
)
target_link_libraries(owe_tests PRIVATE owe)
add_test(NAME owe_tests COMMAND owe_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
