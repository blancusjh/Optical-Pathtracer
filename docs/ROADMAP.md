# Roadmap

State on 2026-09-29: exposure and free-view image formation, SPPM with photon guiding, photoreal
shading (smooth normals, material maps, relief, AgX, image textures, HDR skies), BDPT and VCM are
done on the GPU; glTF models with texture coordinates load (`tests/test_models.cpp`). New work
goes on the GPU first. The order below is the priority.

## 1. Fast loading (next)

Measured on an RTX 4060 laptop GPU:

| run | time |
|---|---|
| `the_study` (vcm), driver cache warm | 0.4 s (`owe render`), 3.5 s for four views (`owe view`) |
| `the_temple`, `the_telescope`, warm | ~4 s each (20 s of CPU on host threads: generated meshes, BVHs) |
| `the_study` right after a kernel change | 70 s |
| a two-triangle quad (path) right after a kernel change | 34 s |

Slow loading is the driver compiling pipelines from cold. It happens after every kernel change
and whenever NVIDIA's own cache (`~/.cache/nvidia/GLCache`, 441 MB now) evicts them. It runs on
one thread. Scene loading itself is not the bottleneck. There are twelve large kernels (every
integrator × hardware/software traversal) of 1.1–2.4 MB of SPIR-V each. The engine passes no
pipeline cache (`vkCreateComputePipelines(dev, VK_NULL_HANDLE, …)` in `vulkan_context.cpp`).

1. **Persistent `VkPipelineCache`** in `~/.cache/owe/`, keyed by device, driver version and SPIR-V
   hash. Pipelines then survive driver-cache eviction.
2. **Compile only what the integrator needs**, in parallel on host threads. The viewer should
   compile in the background with a progress note, showing the path tracer while VCM compiles.
3. **Smaller kernels.** First find what inflates them: unrolled hero-wavelength loops, BSDF and
   texture switches inlined at every call site, variants built with `#ifdef`. Then try Slang
   `[noinline]` functions, specialization constants and optimization levels, measuring the
   compile time per kernel.
4. **Then scene-side caching:** write the generated meshes and the mesh groups' BVHs to disk as
   binary files keyed by content hash, and memory-map them. The temple and the telescope would
   then load in well under a second.

## 2. Engine: what remains

In priority order after fast loading. GPU first; the CPU reference is not ported.

1. **`owe model <file>`:** list a model's parts, triangle counts, images and bounds, so scene
   authors know the `part` names and texture files. Small. Today the loader lists the parts only
   when a `part` name is wrong.
2. **Stained glass:** a thin coloured-glass surface.
   - Optics: thin-slab Fresnel with incoherent inter-reflections; transmission goes straight
     through.
   - Colour: the internal transmittance at normal incidence comes from a picture mapped with
     `mapping = uv`, applied as τ^(1/cos θₜ) at oblique incidence. The lead lines are its dark
     texels.
   - Scans whose glass is its own glTF material use it through `part =`.
3. **Sun from photographed skies:** move the sun's disc out of an HDR sky into an analytic sun,
   for next-event estimation and photon guiding.
4. **OpenEXR and a celestial frame:** place NASA's Deep Star Maps for the observer's latitude and
   time.
5. **Participating media in VCM and SPPM** (photons stored in the medium): visible beams from a
   crystal, shafts under a rose window. Today BDPT and VCM refuse scattering media.
6. **Heterogeneous media** (density grids, delta tracking): smoke.
7. **Manifold next-event estimation** (Phase 5): glints (eye → specular chain → sun) and
   telescope stars. The benchmarks' glint masks measure it.
8. **BDPT and VCM with physical cameras:** today they need an ideal observer.
9. **Smaller items:**
   - glTF roughness and normal maps (`metallicRoughnessTexture`, `normalTexture`);
   - OBJ `vt` texture coordinates;
   - efficiency-aware MIS and an adaptive photon radius, from the method plan;
   - VCM with wavelength groups (more than 1) is noisier than with 1, so the default is 1.
   - region inconsistencies around 1e-5 of paths in a few scenes.

## 3. Scene building: what remains

Delegate each scene to an agent. Brief it: GPU only, no edits under `src/`, use the built
`build/owe`, short renders, no commits.

**Assets.**
- Sketchfab models are downloaded through the user's logged-in Chromium
  (the model page's Download → glTF), into the git-ignored `assets/sketchfab/<slug>/`.
- `assets/sketchfab/MANIFEST.json` records for each model:
  - its licence, author and triangle count;
  - its `.gltf` path;
  - its materials, with their base-colour textures.
- CC BY models need credit in the scene's comments and in the README. These are the globe, the
  Beauvais vault, the Valencia and Southwark windows, and the tellurium.
- The night skies `assets/hdri/qwantani_night_puresky_4k.hdr` and `kloppenheim_02_puresky_4k.hdr`
  (Poly Haven, CC0) work today with `sky = map(...)`.

| scene | uses | needs from the engine |
|---|---|---|
| Observatory: the Jagiellonian Globe on the desk, gilt (gold conductor tinted by the scan's photograph, `mapping = uv`); a photographed night sky | globe, night skies | nothing (star maps later: §2.4) |
| Cathedral chapel under the Beauvais vault, sun through stained glass onto the floor | vault, Valencia or Southwark window | stained glass (§2.2) |
| Tiffany lamp: the Peacock lamp lit from inside, its colours thrown on a room | Peacock lamp | stained glass (§2.2) |
| Armoury or study still life: polished armour, helmet, rapier, astrolabe, book | Cleveland armour, helmet, rapier, astrolabe, Coptic book | nothing |
| Instrument room: Gregorian telescope (its mirrors could become real optics), sextant, tellurium | those scans | nothing |
| Caustic panel (Mitsuba's caustic-design tutorial) | solver and script in `assets/` | nothing |
| Forest sunbeams: the telescope's forest in fog | existing forest | path tracing works today; VCM needs §2.5 |
| Crystal and glass sphere with rainbow spots and beams | built from lathes and facets | beams in haze need §2.5 |

**Caustic panel: state and open problems.**
- In `assets/` (not in git):
  - the solver [poisson_caustic_design](https://github.com/dylanmsu/poisson_caustic_design)
    (MIT), built in `assets/tools/`;
  - `assets/caustic_panel/build_caustic_panel.py`;
  - the target, Dürer's *Rhinoceros* (NGA 1964.8.697, public domain).
- The predicted caustic is not recognisable yet. Suspects: the solver's hard-coded index of 1.49
  and its negated axes. Test it on the solver's own `img/siggraph.png` first.
- The solver's side walls are not watertight; rebuild them in the script.
- Then run at 240–300 cells. The fallback solver is
  [causticsEngineering](https://github.com/MattFerraro/causticsEngineering) (MIT, Julia).
- Scene plan:
  - a PMMA panel;
  - a sun of 0.1° angular diameter (1.7 mm of blur at 1 m);
  - a black baffle around the panel;
  - a screen at 1 m;
  - `vcm`.
- Under VCM, sun photons spread over the whole scene's disc, so keep the layout compact. Check
  the panel first with a `sensor` screen and light tracing.

**Existing scenes.** The status page's observatory image is out of date. After the new scenes,
re-render the gallery (`tools/render_gallery.sh`).

## Known limitations

- The CPU reference ignores texture coordinates (it projects `uv` images planar), smooth normals
  and material maps.
