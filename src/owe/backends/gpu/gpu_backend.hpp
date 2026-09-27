// The portable GPU backend: Vulkan compute with kernels written in Slang (shaders/), running on
// NVIDIA, AMD and Intel GPUs and on Apple silicon through MoltenVK. It is a float32 port of the
// reference transport, working camera-relative so that precision holds from an eyepiece to Saturn,
// and depends only on the scene model: the world is flattened for the kernels (gpu_scene.hpp).
// The public interface carries no Vulkan types; builds without -DOWE_GPU=ON contain a backend that
// reports itself unavailable, and why.
#pragma once

#include "owe/render/backend.hpp"

namespace owe {

const Backend& gpuBackend();

}  // namespace owe
