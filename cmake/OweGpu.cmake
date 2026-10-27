# Portable GPU backend: Vulkan compute (NVIDIA, AMD, Intel; Apple silicon through MoltenVK)
# with kernels written in Slang and compiled to SPIR-V at build time.
#
# Nothing is required from the system except a Vulkan driver at run time: Vulkan-Headers and
# volk (the function loader, which opens libvulkan / MoltenVK dynamically) are fetched at pinned
# tags, and slangc is taken from the Vulkan SDK / PATH or downloaded as a pinned release.

include(FetchContent)
enable_language(C)  # volk

set(OWE_VULKAN_TAG "vulkan-sdk-1.4.357.0" CACHE STRING "Vulkan-Headers / volk tag")
set(OWE_SLANG_VERSION "2026.18.3" CACHE STRING "Slang release used when slangc is not found")

FetchContent_Declare(owe_vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG ${OWE_VULKAN_TAG}
  GIT_SHALLOW TRUE)
FetchContent_Declare(owe_volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG ${OWE_VULKAN_TAG}
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _none)  # only volk.c/volk.h are used
FetchContent_MakeAvailable(owe_vulkan_headers owe_volk)

add_library(owe_volk STATIC ${owe_volk_SOURCE_DIR}/volk.c)
target_include_directories(owe_volk PUBLIC ${owe_volk_SOURCE_DIR})
target_link_libraries(owe_volk PUBLIC Vulkan::Headers ${CMAKE_DL_LIBS})
if(APPLE)
  target_compile_definitions(owe_volk PUBLIC VK_ENABLE_BETA_EXTENSIONS)  # VK_KHR_portability_subset
endif()

# --- slangc
find_program(OWE_SLANGC slangc HINTS $ENV{VULKAN_SDK}/bin)
if(NOT OWE_SLANGC)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(_arch aarch64)
  else()
    set(_arch x86_64)
  endif()
  if(CMAKE_HOST_APPLE)
    set(_asset "slang-${OWE_SLANG_VERSION}-macos-${_arch}.tar.gz")
  elseif(CMAKE_HOST_WIN32)
    set(_asset "slang-${OWE_SLANG_VERSION}-windows-${_arch}.zip")
  else()
    set(_asset "slang-${OWE_SLANG_VERSION}-linux-${_arch}-glibc-2.28.tar.gz")
  endif()
  message(STATUS "slangc not found; fetching ${_asset}")
  FetchContent_Declare(owe_slang
    URL https://github.com/shader-slang/slang/releases/download/v${OWE_SLANG_VERSION}/${_asset}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _none)
  FetchContent_MakeAvailable(owe_slang)
  find_program(OWE_SLANGC slangc PATHS ${owe_slang_SOURCE_DIR}/bin NO_DEFAULT_PATH REQUIRED)
endif()
message(STATUS "slangc: ${OWE_SLANGC}")

# owe_add_kernel(<target> <entry .slang> <symbol> [RAY_QUERY]): compiles a compute kernel to SPIR-V
# and embeds it in the target as `const unsigned char <symbol>[]` / `<symbol>_size` (in bytes).
# Modules are looked up next to the entry point (the backend's shaders/ directory) and, for the
# formulas shared with the C++ reference, under src/ (owe/core/shared). RAY_QUERY builds
# the variant that traverses with VK_KHR_ray_query (used where the device has ray-tracing hardware).
function(owe_add_kernel target source symbol)
  get_filename_component(_dir ${source} DIRECTORY)
  set(_spv ${CMAKE_BINARY_DIR}/shaders/${symbol}.spv)
  set(_cpp ${CMAKE_BINARY_DIR}/shaders/${symbol}.cpp)
  set(_deps ${source} ${OWE_SLANGC})
  set(_depfile_args)
  if(CMAKE_GENERATOR MATCHES "Ninja|Makefiles" OR
     (CMAKE_VERSION VERSION_GREATER_EQUAL "3.21" AND CMAKE_GENERATOR MATCHES "Visual Studio|Xcode"))
    set(_depfile_args DEPFILE ${_spv}.d)
  else()
    # Preserve the CMake 3.20 IDE-generator build; newer generators use actual includes.
    file(GLOB _includes CONFIGURE_DEPENDS ${_dir}/*.slang ${_dir}/../*.inc ${CMAKE_SOURCE_DIR}/src/owe/core/shared/*.hpp)
    list(APPEND _deps ${_includes})
  endif()
  set(_flags -profile spirv_1_3)
  if("RAY_QUERY" IN_LIST ARGN)
    set(_flags -profile spirv_1_4 -capability spvRayQueryKHR -DOWE_RAY_QUERY=1)
  endif()
  add_custom_command(
    OUTPUT ${_spv}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${OWE_SLANGC} ${source} -I ${_dir} -I ${CMAKE_SOURCE_DIR}/src -target spirv ${_flags}
            -entry main -stage compute -O3 -warnings-disable 41035 -o ${_spv} -depfile ${_spv}.d
    DEPENDS ${_deps}
    ${_depfile_args}
    COMMENT "slangc ${symbol}"
    VERBATIM)
  add_custom_command(
    OUTPUT ${_cpp}
    COMMAND ${CMAKE_COMMAND} -DIN=${_spv} -DOUT=${_cpp} -DSYMBOL=${symbol} -P ${CMAKE_SOURCE_DIR}/cmake/EmbedSpirv.cmake
    DEPENDS ${_spv} ${CMAKE_SOURCE_DIR}/cmake/EmbedSpirv.cmake
    VERBATIM)
  target_sources(${target} PRIVATE ${_cpp})
endfunction()
