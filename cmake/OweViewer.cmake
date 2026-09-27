# SDL3 is supplied by the platform; Dear ImGui is pinned and compiled into the viewer.
find_package(SDL3 3.2 CONFIG REQUIRED)
include(FetchContent)
FetchContent_Declare(owe_imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.92.9b
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR _none)
FetchContent_MakeAvailable(owe_imgui)
add_library(owe_imgui STATIC
  ${owe_imgui_SOURCE_DIR}/imgui.cpp
  ${owe_imgui_SOURCE_DIR}/imgui_draw.cpp
  ${owe_imgui_SOURCE_DIR}/imgui_tables.cpp
  ${owe_imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${owe_imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
  ${owe_imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp)
target_include_directories(owe_imgui PUBLIC ${owe_imgui_SOURCE_DIR} ${owe_imgui_SOURCE_DIR}/backends)
target_link_libraries(owe_imgui PUBLIC SDL3::SDL3)
target_sources(owe_cli PRIVATE apps/viewer.cpp)
target_link_libraries(owe_cli PRIVATE owe_imgui)
