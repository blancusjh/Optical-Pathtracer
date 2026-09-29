# Roadmap

State on 2026-09-28: exposure and free-view image formation, SPPM with photon guiding, photoreal
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

## 2. Assets

- **Sketchfab downloads:** a free account's API token in `~/.config/owe/sketchfab_token`, read by
  `tools/fetch_assets.sh`. The download API returns a signed URL to a glTF zip, which goes into the
  git-ignored `assets/`.
- **Models to fetch** (see the references page):

  | model | Sketchfab id | licence | size | for |
  |---|---|---|---|---|
  | Jagiellonian Globe | `55a10e6015af4447a66f34b272d4bf1a` | CC BY | 1.0M triangles | observatory |
  | Vault, Beauvais Cathedral | `939ab33fef8c421888dbe7eba15ff1fc` | CC BY | 0.73M triangles | cathedral scene |

  Also the Valencia rose window, the Southwark window, the Cleveland armour, the astrolabe, the
  Gregorian telescope and the Coptic book. CC BY models need credit in the scene comments and the
  README.
- **Night skies usable today** with `sky = map(...)`: Poly Haven's `qwantani_night_puresky` and
  `kloppenheim_02_puresky` (CC0).
- **`owe model <file>`:** a command that lists a model's parts, triangle counts, images and
  bounds. Today the loader lists the parts only when a `part` name is wrong.

## 3. Scenes

Delegate each scene to an agent. Brief it: GPU only, no edits under `src/`, short renders, no
commits.

- **Observatory:** the Jagiellonian Globe on the desk (gilt: gold conductor tinted by the scan's
  photograph through `mapping = uv`).
- **Cathedral chapel** under the Beauvais vault: the sun through stained glass onto the floor.
  Needs §4.1.
- **Caustic panel,** reproducing Mitsuba's caustic-design tutorial:
  - Materials so far, in `assets/` (not in git):
    - the solver [poisson_caustic_design](https://github.com/dylanmsu/poisson_caustic_design)
      (MIT), built in `assets/tools/`;
    - `assets/caustic_panel/build_caustic_panel.py`;
    - the target, Dürer's *Rhinoceros* (NGA 1964.8.697, public domain).
  - Open problems:
    - the predicted caustic is not recognisable yet. Suspects: the solver's hard-coded index of
      1.49 and its negated axes. Test it on the solver's own `img/siggraph.png` first;
    - the solver's side walls are not watertight; rebuild them in the script.
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
- **Forest sunbeams:** the telescope's forest in fog, with path tracing (VCM has no media yet).

## 4. Engine

1. **Stained glass:** a thin coloured-glass surface.
   - Optics: thin-slab Fresnel with incoherent inter-reflections; transmission goes straight
     through.
   - Colour: the internal transmittance at normal incidence comes from a picture mapped with
     `mapping = uv`, applied as τ^(1/cos θₜ) at oblique incidence. The lead lines are its dark
     texels.
   - Scans whose glass is its own glTF material use it through `part =`.
2. **Sun from photographed skies:** move the sun's disc out of an HDR sky into an analytic sun,
   for next-event estimation and photon guiding.
3. **OpenEXR and a celestial frame** for NASA's Deep Star Maps: the true stars behind the
   observatory's planets.
4. **Participating media in VCM and SPPM** (photons stored in the medium): visible beams from a
   crystal, shafts under a rose window.
5. **Heterogeneous media** (density grids, delta tracking): smoke.
6. **Manifold next-event estimation** (Phase 5): glints (eye → specular chain → sun) and
   telescope stars. The benchmarks' glint masks measure it.
7. **glTF roughness and normal maps** (`metallicRoughnessTexture`, `normalTexture`), and OBJ `vt`
   texture coordinates.

## Known issues

- A few scenes show region inconsistencies around 1e-5 of paths.
- VCM with wavelength groups (more than 1) is noisier than with 1, so the default is 1.
- BDPT and VCM need an ideal observer and no scattering media.
- The CPU reference ignores texture coordinates (it projects `uv` images planar), smooth normals
  and material maps.
