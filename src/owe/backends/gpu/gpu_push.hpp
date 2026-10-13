#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace owe::gpu {
#define OWE_PUSH_U uint32_t
#define OWE_PUSH_F float
#define OWE_PUSH_FIELD(kind, name, offset) OWE_PUSH_##kind name;
struct PushConstants {
#include "push_fields.inc"
};
#undef OWE_PUSH_FIELD
#define OWE_PUSH_FIELD(kind, name, offset) static_assert(offsetof(PushConstants, name) == offset);
#include "push_fields.inc"
#undef OWE_PUSH_FIELD
#undef OWE_PUSH_U
#undef OWE_PUSH_F
static_assert(std::is_standard_layout_v<PushConstants> && sizeof(PushConstants) == 80);
} // namespace owe::gpu
