# The `.owe` scene language

A scene file is a list of settings and blocks:

```
units = mm                          # default length unit for unit-less numbers (m, cm, mm, um, km)
keyword Name { key = value ... }    # blocks may nest (assemblies contain bodies)
```

Comments start with `#` or `//`. Separators (`;`, newlines) are optional.

## Values

| Form | Example | Notes |
|---|---|---|
| number | `6.2`, `-120`, `1e-3`, `inf` | unit-less lengths use `units`; unit-less angles are degrees |
| number with unit | `48 mm`, `0.5 m`, `12deg`, `0.2 rad`, `5000K`, `550 nm` | |
| tuple | `(0, 1, 2)`, `(0, 1, 2) m` | a trailing unit applies to all elements |
| list | `[N-BK7, F2]` | |
| name / string | `N-BK7`, `"lenses/x.lens"` | |
| call | `sphere(R = 48)`, `rgb(0.8, 0.1, 0.1)` | positional and keyword arguments |

Wavelengths without a unit are nanometres.

### Spectra

`0.7` (constant) · `rgb(r, g, b [, luminance = Y, scale = s])` (a smooth, energy-conserving
reflectance for materials; an emission spectrum on a 6504 K Planckian base for lights) ·
`blackbody(T, Y)` (Planck spectrum normalised to luminance-like Y) ·
`tabulated(lambda = [...], values = [...])` · `d65(Y)`.

### Placement (bodies, assemblies, sensors)

`position = (x, y, z)`; orientation by one of `axis = (dx, dy, dz)` (local +z → axis),
`look_at = point` (local +z toward the point), `point_at = point` (local −z toward the point —
for instruments, whose light travels along +z), or `rotate = (rx, ry, rz)` (degrees, applied
x, then y, then z); `up = (…)` hints local +y. `decenter = (dx, dy)` and `tilt = (ax, ay)` then
act in the local frame — the natural tolerancing operations.

Three functions compute positions: `exit_pupil("Instrument" [, offset])` returns the world
position of an instrument's paraxial exit pupil; `on_terrain("Terrain", x, y [, height])`
returns a point on a terrain body; `sky(azimuth, elevation, distance)` returns a point in a
given direction (azimuth from +y toward +x, as for the sun), used to place and aim at the Moon
and planets at their real distances.

`repeat = (n, (dx, dy, dz))` on a body makes `n` identical copies `Name_0 … Name_{n−1}`, each
offset by the step in the parent frame (a row of columns).

## Blocks

### `medium Name { … }`

`base = N-BK7` (start from a catalog entry) · `index = sellmeier(B = (…), C = (…) [, relative_to_air = true])`
| `cauchy(A =, B =, C =)` | `constant(1.5)` | `tabulated(lambda = […], n = […])` ·
`absorption = <spectrum>` (σₐ in 1/m) · `internal_transmittance = (τ, thickness)` ·
`scattering = <spectrum>` (σₛ in 1/m) · `g = 0.3`.

Catalog media: `air` (Ciddor), `vacuum`, `water`, `N-BK7`, `N-SK16`, `N-BAF10`, `F2`,
`N-F2`, `N-SF6`, `N-SF11`, `fused-silica`, `CaF2`, `sapphire`, `diamond`, `PMMA`, `crown`,
`flint`, `opaque`. List them with `owe glass`.

### `material Name { … }` — surface optics of opaque bodies

`type = diffuse | mirror | conductor | absorber | null | dielectric` ·
`reflectance = <spectrum>` · `texture = checker(a =, b =, scale =) | noise(…) | wood(…, turbulence =) | terrain(a =, b =, c =, scale =, snow_line =) |
marble(a =, b =, scale =, turbulence =) | bands(a =, b =, c =, scale =, turbulence =) | radial(a =, b =, scale =) |
image("file.jpg", scale =, mapping = planar | triplanar | uv)` ·
`metal = aluminium | silver | gold | copper` · `roughness = α | (α_a, α_b[, α_c])` (GGX) · `tint = <spectrum>` ·
`back = black` (thin sheets whose back absorbs).

A texture mixes materials, not only colours: with `roughness = (α_a, α_b, α_c)` each texture
component has its own roughness, mixed by the same weights as the colour (a material map: marble
polished in the stone and honed in the veins, brushed and polished metal). A conductor's texture
tints it per component (`tint` otherwise). A transparent body takes `texture` and a roughness pair
too: `medium = N-BK7  texture = checker(a = 1, b = 1, scale = 5mm)  roughness = (0, 0.3)` is clear
glass etched in squares. Material maps are GPU-only; the CPU reference uses the first roughness.

`relief = noise(scale =, depth =, octaves = 3)` gives any surface a relief finer than its geometry:
a height `depth · fbm(p / scale)` in the body's frame whose slopes (about depth / scale) tilt the
shading normal, so a low sun rakes a paper's grain or a concrete floor's pores. Transport treats it
as it treats a smooth mesh's normals (GPU-only; the geometry itself stays exact).

Textures are solid (3-D) patterns in the body's local frame: `marble` has thin dark veins,
`bands` are latitude bands along local z (gas giants), `radial` varies with the distance from
local z (planetary rings). `image` is a photograph (PNG, JPEG, HDR; 8-bit files are sRGB) used
as reflectance: its linear RGB weights three smooth reflectance spectra, so a white pixel is
spectrally flat. It repeats every `scale` (one image per `scale` in each direction), projected
along local z (`planar`) or blended from the three axis projections by the surface's facing
(`triplanar`, for bodies without a natural up), or placed by a textured model's own texture
coordinates (`uv`: a scan's photograph on the scan; GPU-only, and a surface without texture
coordinates takes the planar projection).

Built-in materials: `white`, `grey`, `black`, `ideal_mirror`, `aluminium`, `silver`, `gold`,
`copper`.

### `world { … }`

`medium = air` · `up = (0, 0, 1)` · `sky = none | uniform(<spectrum>) | gradient(zenith =, horizon =, ground =) |
map("sky.hdr", rotation = 0deg, scale = 1)` ·
nested `sun { elevation =, azimuth = | direction = (…); angular_diameter = 0.533deg;
luminance = Y | radiance = <spectrum>; temperature = 5778; nee_share = s }`.

`map` lights the world with an equirectangular HDR photograph of a real sky (zenith at the top row,
azimuth from +x toward +y, turned by `rotation`, its radiance times `scale`). It is a light like
any other: next-event estimation draws directions in proportion to its brightness (a bright sun
in the photograph is found, not waited for), with MIS against the surfaces' own sampling.

The sun is a directional source at infinity with its irradiance at Earth. `nee_share` is a
sampling choice (the fraction of light samples given to the sun; the rest go to emitting
bodies in proportion to their power); it changes noise, never the expected value. It matters
at night, when the sun lights only the Moon and planets and the candles need the samples.

### `body Name { type = … }`

A transparent body takes `medium = …` (its surfaces are Fresnel interfaces, `roughness = α`
for frosted glass; `surface = null` for a medium matched to its surroundings, such as haze or smoke
in air, whose boundary only changes the region, so light and shadow rays cross it); an opaque one
takes `material = …`. Any solid can emit:
`emission = <spectrum>`, `emission_sides = front | back | both`. `in = Name` places a body inside
another body's transparent medium (stones under water), defined earlier.

| type | parameters |
|---|---|
| `lens` | `medium`, `front`, `back` (surfaces), `thickness`, `diameter` / `front_diameter` / `back_diameter` / `edge_diameter`, `rim = ground \| black \| polished`; or `surfaces = […]`, `media = […]`, `thicknesses = […]` for cemented groups; or `focal =`, `form = bi \| plano` |
| `prescription` | `file`, `afocal = true` (solve the tube length; its exit pupil then guides observers' pupil samples), `tube = true` or `tube_radius`, `tube_material` (outer finish: brass, paint; the inside stays black), `eyecup = true` (physical black light shield around the eye-side opening), `rim` |
| `mirror` | `surface`, `diameter`, `thickness`, `hole_diameter`, `material` (default aluminium) |
| `flat_mirror` | `size = (w, h)` or `ellipse = (a, b)`, `material` |
| `stop`, `iris`, `aperture` | `aperture` (diameter), `outer_diameter` |
| `tube` | `diameter`/`radius`, `length`, `material` |
| `sphere`, `cylinder`, `box`, `prism` | `radius`; `radius, height`; `size`; `apex, side, length` |
| `sheet`, `screen`, `disk` | `size = (w, h)` or `radius`/`diameter`, `inner_radius` (annulus); two-sided |
| `mesh` | `file` (OBJ, or glTF 2.0 `.gltf`/`.glb`), `part = "Name" \| ["A", "B"]` (glTF: only these materials' triangles), `scale`, `model_up = y \| -y \| x \| … \| (x, y, z)` (the model's up axis or a measured up vector), `fit_height` (rescale to a height, standing on the origin), `ground = true` (stand on the origin without rescaling), `optional = true` (skip when the file is missing), `closed = true` (treat as a watertight solid; by default an opaque imported mesh is a shell with air on both sides, which tolerates scans with holes and mixed winding), `normals = flat \| smooth \| smooth(crease = 30deg) \| file` (shading normals, below) |
| `lathe` | `profile = [(r, z), …]` (radius, height; closed onto the axis automatically), `segments` |
| `torus`, `ring` | `radius` (to the tube centre), `tube_radius`, `segments`, `tube_segments` |
| `column` | Doric column: `height`, `radius` (lower shaft), `taper`, `entasis`, `flutes`, `flute_depth`, `capital_height` |
| `round_wall` | cylindrical wall: `radius`, `height`, `openings = [(azimuth, width, sill, top), …]` |
| `dome` | hemisphere: `radius`, `slit_azimuth`, `slit_width`, `slit_top` (elevation where the slit ends) |
| `starfield` | `count`, `seed`, `distance`, `angular_radius`, `brightest` (luminance of the brightest star), `min_elevation`: small emitting spheres on the celestial sphere, with a steep magnitude distribution and a range of colour temperatures; directly sampled in a separate group so they do not starve nearby lamps |
| `cup` | `outer_diameter`, `wall`, `base`, `height`, `level`, `glass`, `liquid`, `rod = rod(diameter =, at = (x, y), top =, medium = \| material =)` |
| `vessel` | `outer = [(r, z), (r, z, fillet), …]`, `inner = […]` (profiles from the axis at the bottom up to the rim; a third value rounds that corner with a tangent arc), `rim = round \| flat`, `glass`, `liquid`, `level`: glassware turned on a lathe, every piece an analytic disc, cylinder, cone or torus patch |
| `waves` | `size = (w, l)`, `depth`, `margin` (taper to calm water at the edges), `medium`, `bottom`, `walls` (opaque materials: the liquid then fills a hole in a solid block with a `rim` of ground and a `base`), `waves = [wave(amplitude, wavelength, direction, phase), …]` or `ripples(count, seed, wavelength = (min, max), slope, direction, spread)` |
| `terrain` | `size`, `resolution`, `amplitude`, `feature`, `ridge`, `seed`, `flat_radius`, `reference_only = true` (the height field that `on_terrain` and forests use, without matter: for scenes whose visible ground is a detailed mesh, so no second surface overlaps it) |
| `forest` | `terrain`, `count`, `inner_radius`, `outer_radius`, `seed`, `foliage`, `trunk` |
| `fractal_statue` | `radius`, `depth`, `ratio`, `material`, `pedestal` |

glTF models: every mesh of the file's scene, placed by its node transforms and turned from
glTF's y-up to the engine's z-up (so `model_up` is rarely needed), with its normals (for
`normals = file`) and texture coordinates. Its materials are parts: load a model's stone and its
glass as two bodies with `part =`, each with its own material; `model_up`, `fit_height` and `ground`
measure the whole model, so the parts stay together. The engine does not read glTF materials:
give the body a material, and for a scan's photograph use
`texture = image("textures/…_baseColor.jpeg", mapping = uv)` (the file beside the model).
`owe model <file>` lists a model's parts with their triangles, extents and images, and the paths a
scene writes to load them; `owe` prints the parts too when a `part` name is wrong.

Mesh shading normals: a mesh is flat by default, each triangle its own plane. `normals = smooth`
gives every corner the angle-weighted mean normal of the triangles around its vertex that meet
its own at less than the crease angle (30° unless `smooth(crease = …)` says otherwise), so curved
surfaces shade smoothly while sharper edges stay sharp; vertices at the same position count as
one, so files split at texture seams still smooth across them. A scan's facets are noisy: give it
a larger crease (60–70°). `normals = file` takes the file's own `vn`.
Glass meshes refract about the smooth normal too (a 32 × 16 glass ball's caustic is the analytic
ball's). Only the GPU backend uses shading normals; the CPU reference shades every mesh flat.

Surfaces: `plane()` · `sphere(R =)` · `conic(R =, k =)` · `asphere(R =, k =, A4 =, A6 =, …)`
(coefficients in scene units: z = c r²/(1+√(1−(1+k)c²r²)) + Σ A₂ₘ r²ᵐ).

### `assembly Name { placement …  body … { } assembly … { } }`

Groups bodies mechanically; child placements are relative to the assembly.

### Detectors

* `observer Name { position, look_at | direction, up, fov, pupil (diameter), focus, resolution = (w, h) }`
  — an ideal eye: pupil plus a perfect angular retina. Measures radiance. When it looks into an
  afocal instrument whose exit pupil is much smaller than its own pupil (a 5 mm eye at a
  telescope's 1 mm Ramsden disc), most pupil samples are drawn where the instrument's light
  arrives; the samples are weighted by their density, so only the noise changes.
* `camera Name { lens = "file.lens", sensor = (w, h), resolution, position, look_at, up, focus, f_number, real_focus, sensor_shift, housing }`
  — a physical camera built as lens + housing + mounts + stop + sensor. Measures irradiance.
  `f_number` resizes the lens's physical aperture stop so that the paraxial entrance pupil is
  EFL / (2N) (lenses without a stop get an iris in front); `focus` is the object distance
  brought to focus, with the sensor at the real plane of least blur for the chosen aperture
  (`real_focus = false` uses the paraxial image). The render log prints EFL, f-number, focus and
  the thin-lens depth of field (circle of confusion = diagonal / 1500). Sensor samples are aimed
  mostly at the exit pupil and partly at the whole rear opening, which keeps ghost light.

Every detector accepts `pixel_filter = box | gaussian | gaussian(σ)`: the pixel's response is
its square convolved with a Gaussian of σ pixels (`gaussian` is σ = 0.4). Camera samples and
particle splats are displaced by the same Gaussian, so path, light and hybrid estimates measure
the same filtered image. With the square alone a source smaller than a pixel (a star, a focused
caustic) images as a pixel-aligned square; with the Gaussian it images as a round spot centred
where it falls. Observers default to `gaussian`, sensors and cameras to `box` (a physical
pixel's integral).

Any detector may carry its own display and sampling settings, which apply when it is rendered:
`exposure = EV` (a fixed, manual exposure replacing the meter: for views whose subject is small
and bright, like an eyepiece on a planet), `white_balance = K | none`, `tone = agx | standard`, and
`sun_share = s` (the sun's `nee_share` for this view: an eyepiece on Saturn wants nearly all
light samples on the sun; the candlelit room around it does not).
* `sensor Name { size = (w, h), resolution, placement, material, aim_center, aim_axis, aim_diameter }`
  — a surface irradiance measurement (front = local +z). Without `material` it is an absorbing
  sensor. With a diffuse `material` it is a measurement screen: it records incident irradiance
  and scatters light with the specified reflectance, so an observer can see a projected image.
  The readout is incident irradiance, not the screen's reflected radiance. An opaque screen's
  image is visible from its illuminated side, not transmitted through its back.
  The optional aim is an importance-sampling hint for an opening. Diffuse screens mix 95%
  opening-directed samples with 5% cosine-hemisphere samples, for both readout rays and
  camera-path continuation at the screen; this retains support for indirect illumination.
  Absorbing sensors use the aim exclusively and require it to cover all incident paths.

### `render { … }`

`detector`, `integrator = path | light | hybrid | sppm | bdpt | vcm`, `spp` (sppm, bdpt, vcm: iterations), `seed`, `max_depth`, `exposure`
(EV, display only), `auto_exposure`, `white_balance = K | none` (display white point: a
Bradford adaptation from a Planckian white at K to D65, so candlelight at 2000 K reads as warm
white; the raw PFM is untouched), `tone = agx | standard` (the display's view transform: `agx`,
the default, is a filmic curve over 16.5 stops whose brightest colours go gradually to white, as a
caustic's core or a sun glint does on film; `standard` is linear up to white with a short
roll-off; middle grey shows the same in both), `fresnel_floor` (sampling knob, 0–0.49); for `sppm`:
`photon_radius` (initial radius in pixel footprints, 2), `photon_alpha` (radius reduction, 2/3),
`photons` (per pixel and iteration, 1) — estimator choices that change noise and bias, never the
limit. `vcm` and `bdpt` trace `photons` light subpaths per pixel and iteration, split into
`vcm_groups` wavelength groups (1; more settle a dispersed colour sooner at more noise elsewhere),
and `vcm` merges with `photon_radius` and `photon_alpha` too (its radius a few pixel footprints at
each point's distance from the eye). `bdpt` and `vcm` (GPU) need an observer; `vcm` is the robust
choice for glossy surfaces together with caustics. With scattering media they split the paths:
those that scatter in a medium are path traced in a pass of their own (next-event estimation
reaches the sun through stained glass and haze), and VCM or BDPT takes all the others, through
media as extinction; the two add up to the whole image.

`indirect_guide = BodyName` optionally names a spherical reflector (for example, the Moon).
Diffuse surfaces without their own guide sample a 50/50 mixture of its projected disk and
the full cosine hemisphere. This reduces noise from small indirect light sources; it does
not make the body emissive or replace its reflection, occlusion, or illumination calculation.
The reflector's own material is excluded. Use `indirect_guide = ""` to disable the hint.

## Edits without rewriting the file

Every command accepts `--set Block.key=value` (repeatable), and the studio accepts
`set Block.key=value`. The value is in scene syntax and is appended as the last assignment of
that key in the named block, so it overrides the file: `--set Cam.f_number=2.8`,
`--set "Cam.focus=3.5 m"`, `--set "Lens.position=(0, 0, 0.12)"`, `--set sun.nee_share=0.9`
(unnamed blocks such as `sun` are addressed by keyword), `--set render.spp=64`. Edits are
recorded in the render's JSON record.

## Lens files (`*.lens`)

```
name Achromatic doublet f=100 mm
units mm
wavelength 587.5618          # design wavelength for paraxial data
object_distance inf
# label   R         t      medium  semi-diameter  [k=…] [A4=…] [stop]
S1        44.78     6.0    N-BK7   12.5
S2       -44.78     2.5    F2      12.5
S3      -810.8     95.0    air     12.5
```

Each row is a surface; `medium` is the medium *after* it. Consecutive non-air rows form one
cemented body; `stop` on an air–air row creates an iris. The table is a view: loading it
produces regions, boundaries, rims and mounts.

Mirrors (coaxial): `reflect material=Name substrate=length [hole=diameter]
[support=spider(count=N,width=w,angle=deg,outer_radius=r)]` makes a row a mirror of that scene
material (a mirror or conductor) on an opaque backing, with an optional central bore and opaque
struts behind it. R and t are signed in a fixed chart whose +z is the incoming light's direction;
reflection reverses the travel, so t is negative on a leg travelling −z (a Gregorian:
`P -900 -600 air 54 k=-1 reflect material=Speculum substrate=4 hole=26 stop`, then a concave
ellipsoidal secondary with a positive t back through the bore). With mirrors, `afocal = true`
adjusts the one gap marked `solve`; the paraxial data (focal length, magnification, pupils,
`exit_pupil`) follow the unfolded system, each mirror a lens of power −2 d n / R for travel d.
Refraction on a −z leg, folded mirrors (`out_axis`) and `tube = true` are not supported yet.
