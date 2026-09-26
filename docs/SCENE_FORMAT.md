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
marble(a =, b =, scale =, turbulence =) | bands(a =, b =, c =, scale =, turbulence =) | radial(a =, b =, scale =)` ·
`metal = aluminium | silver | gold | copper` · `roughness = α` (GGX) · `tint = <spectrum>` ·
`back = black` (thin sheets whose back absorbs).

Textures are solid (3-D) patterns in the body's local frame: `marble` has thin dark veins,
`bands` are latitude bands along local z (gas giants), `radial` varies with the distance from
local z (planetary rings).

Built-in materials: `white`, `grey`, `black`, `ideal_mirror`, `aluminium`, `silver`, `gold`,
`copper`.

### `world { … }`

`medium = air` · `up = (0, 0, 1)` · `sky = none | uniform(<spectrum>) | gradient(zenith =, horizon =, ground =)` ·
nested `sun { elevation =, azimuth = | direction = (…); angular_diameter = 0.533deg;
luminance = Y | radiance = <spectrum>; temperature = 5778; nee_share = s }`.

The sun is a directional source at infinity with its irradiance at Earth. `nee_share` is a
sampling choice (the fraction of light samples given to the sun; the rest go to emitting
bodies in proportion to their power); it changes noise, never the expected value. It matters
at night, when the sun lights only the Moon and planets and the candles need the samples.

### `body Name { type = … }`

A transparent body takes `medium = …` (its surfaces are Fresnel interfaces, `roughness = α`
for frosted glass); an opaque one takes `material = …`. Any solid can emit:
`emission = <spectrum>`, `emission_sides = front | back | both`.

| type | parameters |
|---|---|
| `lens` | `medium`, `front`, `back` (surfaces), `thickness`, `diameter` / `front_diameter` / `back_diameter` / `edge_diameter`, `rim = ground \| black \| polished`; or `surfaces = […]`, `media = […]`, `thicknesses = […]` for cemented groups; or `focal =`, `form = bi \| plano` |
| `prescription` | `file`, `afocal = true` (solve the tube length), `tube = true` or `tube_radius`, `tube_material` (outer finish: brass, paint; the inside stays black), `rim` |
| `mirror` | `surface`, `diameter`, `thickness`, `hole_diameter`, `material` (default aluminium) |
| `flat_mirror` | `size = (w, h)` or `ellipse = (a, b)`, `material` |
| `stop`, `iris`, `aperture` | `aperture` (diameter), `outer_diameter` |
| `tube` | `diameter`/`radius`, `length`, `material` |
| `sphere`, `cylinder`, `box`, `prism` | `radius`; `radius, height`; `size`; `apex, side, length` |
| `sheet`, `screen`, `disk` | `size = (w, h)` or `radius`/`diameter`, `inner_radius` (annulus); two-sided |
| `mesh` | `file` (OBJ), `scale`, `model_up = y \| -y \| x \| … \| (x, y, z)` (the model's up axis or a measured up vector), `fit_height` (rescale to a height, standing on the origin), `ground = true` (stand on the origin without rescaling), `optional = true` (skip when the file is missing), `closed = true` (treat as a watertight solid; by default an opaque imported mesh is a shell with air on both sides, which tolerates scans with holes and mixed winding) |
| `lathe` | `profile = [(r, z), …]` (radius, height; closed onto the axis automatically), `segments` |
| `torus`, `ring` | `radius` (to the tube centre), `tube_radius`, `segments`, `tube_segments` |
| `column` | Doric column: `height`, `radius` (lower shaft), `taper`, `entasis`, `flutes`, `flute_depth`, `capital_height` |
| `round_wall` | cylindrical wall: `radius`, `height`, `openings = [(azimuth, width, sill, top), …]` |
| `dome` | hemisphere: `radius`, `slit_azimuth`, `slit_width`, `slit_top` (elevation where the slit ends) |
| `starfield` | `count`, `seed`, `distance`, `angular_radius`, `brightest` (luminance of the brightest star), `min_elevation`: small emitting spheres on the celestial sphere, with a steep magnitude distribution and a range of colour temperatures; not sampled by next-event estimation |
| `cup` | `outer_diameter`, `wall`, `base`, `height`, `level`, `glass`, `liquid`, `rod = rod(diameter =, at = (x, y), top =, medium = \| material =)` |
| `terrain` | `size`, `resolution`, `amplitude`, `feature`, `ridge`, `seed`, `flat_radius` |
| `forest` | `terrain`, `count`, `inner_radius`, `outer_radius`, `seed`, `foliage`, `trunk` |
| `fractal_statue` | `radius`, `depth`, `ratio`, `material`, `pedestal` |

Surfaces: `plane()` · `sphere(R =)` · `conic(R =, k =)` · `asphere(R =, k =, A4 =, A6 =, …)`
(coefficients in scene units: z = c r²/(1+√(1−(1+k)c²r²)) + Σ A₂ₘ r²ᵐ).

### `assembly Name { placement …  body … { } assembly … { } }`

Groups bodies mechanically; child placements are relative to the assembly.

### Detectors

* `observer Name { position, look_at | direction, up, fov, pupil (diameter), focus, resolution = (w, h) }`
  — an ideal eye: pupil plus a perfect angular retina. Measures radiance.
* `camera Name { lens = "file.lens", sensor = (w, h), resolution, position, look_at, up, focus, f_number, real_focus, sensor_shift, housing }`
  — a physical camera built as lens + housing + mounts + stop + sensor. Measures irradiance.
  `f_number` resizes the lens's physical aperture stop so that the paraxial entrance pupil is
  EFL / (2N) (lenses without a stop get an iris in front); `focus` is the object distance
  brought to focus, with the sensor at the real plane of least blur for the chosen aperture
  (`real_focus = false` uses the paraxial image). The render log prints EFL, f-number, focus and
  the thin-lens depth of field (circle of confusion = diagonal / 1500). Sensor samples are aimed
  mostly at the exit pupil and partly at the whole rear opening, which keeps ghost light.

Any detector may carry its own display and sampling settings, which apply when it is rendered:
`exposure = EV` (a fixed exposure replacing auto exposure), `white_balance = K | none`, and
`sun_share = s` (the sun's `nee_share` for this view: an eyepiece on Saturn wants nearly all
light samples on the sun; the candlelit room around it does not).
* `sensor Name { size = (w, h), resolution, placement, aim_center, aim_axis, aim_diameter }`
  — a sensitive surface in the world (front = local +z). The optional aim is an
  importance-sampling hint for an opening through which all light must pass.

### `render { … }`

`detector`, `integrator = path | light | hybrid`, `spp`, `seed`, `max_depth`, `exposure`
(EV, display only), `auto_exposure`, `white_balance = K | none` (display white point: a
Bradford adaptation from a Planckian white at K to D65, so candlelight at 2000 K reads as warm
white; the raw PFM is untouched), `fresnel_floor` (sampling knob, 0–0.49).

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
