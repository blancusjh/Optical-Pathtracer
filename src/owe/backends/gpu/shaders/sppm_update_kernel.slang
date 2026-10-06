// SPPM: the progressive update of each pixel after an iteration's photons (Hachisuka & Jensen 2009):
// N' = N + αM, r'² = r²·N'/(N + M), τ' = (τ + Φ)·N'/(N + M); then the iteration's flux is cleared.
#include "sppm.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint pixel = tid.x;
    if (pixel >= pc.width * pc.height) return;
    const uint base = pixel * kFluxWords;
    const uint M = g_flux[base + kFluxCount];
    if (M == 0) return;  // no photon reached this pixel's point: nothing changes (Φ is zero too)
    SppmPixel s = g_sppm[pixel];
    float3 phi = float3(fluxRead(base), fluxRead(base + 2 * kSplatBins), fluxRead(base + 4 * kSplatBins));
    const float N = s.tau.w, Nn = N + sppmAlpha() * float(M);
    const float ratio = Nn / (N + float(M));
    s.tau.xyz = (s.tau.xyz + phi) * ratio;
    s.tau.w = Nn;
    s.misc.x *= sqrt(ratio);
    g_sppm[pixel] = s;
    for (uint k = 0; k < kFluxWords; ++k) g_flux[base + k] = 0;
}
