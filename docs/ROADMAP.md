# Roadmap

State on 2026-10-05, first version. The GPU renders everything below; new work goes on the GPU
first, and the CPU reference is not ported.

## Done in this version

- **Fast loading.** Kernels compile through a pipeline cache kept on disk (`~/.cache/owe/`, per
  device and driver), and a renderer compiles only its integrator's kernels, in parallel. With
  every cache cold, a VCM start dropped from 131 s to 61 s, which is now the largest single
  kernel. Any later start takes 0.4 s.
- **`owe model <file>`:** a model's parts, sizes and textures, with the paths a scene writes to
  load them.
- **Stained glass** (`type = stained_glass`): a thin slab with Fresnel at both faces and
  internal transmittance from a picture.
  - Covers opalescent `haze`, diffuse reflection, and glass masks that hand masked points to an
    opaque stone material.
  - Shadow rays cross it, so sunlight through a window is direct lighting.
  - Specification: `docs/scenes/peacock_lamp.txt`.
- **Mirrors in lens prescriptions:** coaxial `reflect` rows with a bore and spider struts.
  Paraxial data come from the unfolded system, and `afocal` adjusts the gap marked `solve`.
  Specification: `docs/scenes/gregorian.txt`.
- **VCM and BDPT with scattering media,** by partition:
  - the path tracer takes the paths that scatter in a medium;
  - VCM or BDPT takes the rest, through media as extinction.
- **`surface = null`** for index-matched media: haze or smoke in air whose boundary only changes
  the region.
- **Scenes:**
  - the Kunstkammer, the Gregorian, the Peacock lamp and the invisible window, all
    built on museum scans;
  - an 85 mm physical camera in the study.

## Engine: what remains

1. **Loading:**
   - Smaller kernels. Find what inflates them (unrolled hero-wavelength loops, BSDF and texture
     switches inlined at every call site), then try `[noinline]` and specialization constants.
   - Scene-side caching: write generated meshes and mesh-group BVHs to disk keyed by content
     hash. The Kunstkammer's scans take ~14 s to load.
2. **Stained glass, later parts:**
   - roughness of the reflection;
   - a forward-scattering `haze_angle` lobe;
   - reciprocal diffuse transmission (the opalescent lobe uses the incident angle's T).
3. **Mirrors, later parts:**
   - folded systems (`out_axis`, the Newtonian);
   - `tube = true` for reflectors;
   - refraction on a leg travelling −z;
   - annular-pupil guiding.
4. **Volume vertices in VCM** (merging in media, the UPBP family), in place of the partition,
   where caustics are seen through haze.
5. **Sun from photographed skies:** move the sun's disc out of an HDR sky into an analytic sun.
6. **OpenEXR and a celestial frame,** for NASA's Deep Star Maps.
7. **Heterogeneous media** (density grids, delta tracking): smoke.
8. **Manifold next-event estimation:** glints (eye → specular chain → sun) and telescope stars.
9. **BDPT and VCM with physical cameras.**
10. **Smaller items:**
    - glTF roughness and normal maps;
    - OBJ `vt` coordinates;
    - efficiency-aware MIS and an adaptive photon radius;
    - VCM with more than one wavelength group;
    - region inconsistencies where scans and blocks touch.

## Scenes: what remains

- **Final renders and the gallery:** re-render `docs/gallery` and add the new scenes to the README
  with their credits. The CC BY scans include the globe and the tellurium.
- **The invisible window:**
  - the optional colour version: three panels behind red, green and blue filter glasses;
  - the provenance and licence of the Sacred Heart photograph.
- **Now possible with media in VCM:**
  - forest sunbeams in fog;
  - a crystal with rainbow beams in haze;
  - the observatory with the Jagiellonian globe under a photographed night sky.

## Known limitations

- The CPU reference ignores texture coordinates (it projects `uv` images planar), smooth normals
  and material maps, and refuses stained glass.
