#pragma once
#include <cstddef>
#include <type_traits>

namespace owe::gpu {
#define OWE_FILM_FIELD(name, offset) float name[4];
struct alignas(16) FilmPixel {
#include "film_fields.inc"
};
#undef OWE_FILM_FIELD
#define OWE_FILM_FIELD(name, offset) static_assert(offsetof(FilmPixel, name) == offset);
#include "film_fields.inc"
#undef OWE_FILM_FIELD
static_assert(std::is_standard_layout_v<FilmPixel> && sizeof(FilmPixel) == 64);
} // namespace owe::gpu
