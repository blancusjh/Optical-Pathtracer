// Equirectangular environment maps, shared by the reference and the GPU kernels (see prelude.hpp):
// a direction's map coordinates u (azimuth from e1 toward e2, turned by `rotation`, over 2π) and
// v (angle from the zenith `up`, over π), and back.
#pragma once

#include "owe/core/shared/prelude.hpp"

OWE_SHARED_BEGIN

// The map's frame about its unit zenith: e1 the horizontal axis nearest x (else y), e2 = up × e1.
OWE_FN void envMapFrame(Real3 up, OWE_OUT(Real3) e1, OWE_OUT(Real3) e2) {
    e1 = abs(up.x) < 0.9 ? normalize(Real3(1, 0, 0) - up * up.x) : normalize(Real3(0, 1, 0) - up * up.y);
    e2 = cross(up, e1);
}

OWE_FN Real2 envMapUV(Real3 dir, Real3 up, Real3 e1, Real3 e2, Real rotation) {
    Real c = clamp(dot(dir, up), Real(-1), Real(1));
    Real u = (atan2(dot(dir, e2), dot(dir, e1)) - rotation) / (2 * Pi);
    return Real2(u - floor(u), acos(c) / Pi);
}

// The direction at map coordinates (u, v); sinTheta: the sine of its zenith angle.
OWE_FN Real3 envMapDirection(Real u, Real v, Real3 up, Real3 e1, Real3 e2, Real rotation, OWE_OUT(Real) sinTheta) {
    Real theta = Pi * v, phi = 2 * Pi * u + rotation;
    sinTheta = sin(theta);
    return (e1 * cos(phi) + e2 * sin(phi)) * sinTheta + up * cos(theta);
}

// The solid-angle density of directions drawn uniformly over a texel of a W × H map that is chosen
// with probability pCell, at zenith-angle sine sinTheta.
OWE_FN Real envMapPdf(Real pCell, Real W, Real H, Real sinTheta) { return pCell * W * H / (2 * Pi * Pi * sinTheta); }

OWE_SHARED_END
