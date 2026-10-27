# Scene asset credits

The four scan scenes use external assets kept locally under the ignored `assets/`
directory. Portable saved download records are [Sketchfab](assets/sketchfab.json) and
[Poly Haven](assets/polyhaven.json), with original source/licence texts under
`docs/assets/`. These records describe the local capture; current download availability
has not been rechecked. See [gallery asset setup](GALLERY_WORKFLOW.md#curated-scene-asset-setup). Licence labels below reproduce the saved download
records. The Sketchfab pages for the two CC Attribution assets could not be
reopened during the 8 October 2026 audit; no unverified licence version is added.

## Museum and other scans

| Asset and original source | Author / collection | Saved licence | Current scene use |
|---|---|---|---|
| [The Jagiellonian Globe](https://sketchfab.com/3d-models/jagiellonian_globe-55a10e6015af4447a66f34b272d4bf1a) | Virtual Museums of Małopolska | CC Attribution | kunstkammer |
| [2018.281 Peacock Table Lamp](https://sketchfab.com/3d-models/peacock_table_lamp-a24bcb6b7a2e448fad661004f403ae12) | Cleveland Museum of Art | CC0 Public Domain | peacock lamp |
| [1964.88 Armor for Man and Horse](https://sketchfab.com/3d-models/armor_man_and_horse-faca2ad557f64809b598c3a646a8804f) | Cleveland Museum of Art | CC0 Public Domain | kunstkammer |
| [Closed helmet](https://sketchfab.com/3d-models/closed_helmet-c4bd07612f424fcd9f516cbccc2554e6) | Virtual Museums of Małopolska | CC0 Public Domain | kunstkammer |
| [Rapier of D. João de Mascarenhas](https://sketchfab.com/3d-models/rapier_mascarenhas-cd753effd9b0465b8ab84d6003fe7acd) | Morbase / Museu Virtual | CC0 Public Domain | kunstkammer |
| [Astrolabe](https://sketchfab.com/3d-models/astrolabe-530bd21edcdf4c4695cc4c8af856d6bf) | Virtual Museums of Małopolska | CC0 Public Domain | kunstkammer |
| [Gregorian reflecting telescope (1700s)](https://sketchfab.com/3d-models/gregorian_telescope-385fa9f09a064c8b8d81e5085e2d76f4) | Virtual Museums of Małopolska | CC0 Public Domain | gregorian |
| [Pocket sextant](https://sketchfab.com/3d-models/pocket_sextant-45cf60e598904f26ac8c9f8401e711a7) | Virtual Museums of Małopolska | CC0 Public Domain | kunstkammer |
| [ETH: Tellurium with Lunarium](https://sketchfab.com/3d-models/tellurium_lunarium-e4c9f36a40f3479885f27380f8286b2d) | ETH Library_Collections | CC Attribution | kunstkammer |
| [Coptic prayer book](https://sketchfab.com/3d-models/coptic_prayer_book-500ec4621d764f0dadb8edc93b8700ae) | Världskulturmuseerna | CC0 Public Domain | kunstkammer, peacock lamp |
| [Chest, ZKWawel 1558](https://sketchfab.com/3d-models/chest-c580f18e25b740e49ce8dd0f8cfd3247) | Virtual Museums of Malopolska | CC0 Public Domain | kunstkammer, gregorian |

The [Peacock Table Lamp collection record](https://www.clevelandart.org/art/2018.281)
identifies the maker as Tiffany Studios and the designer as probably Clara Wolcott
Driscoll, c. 1902. The scene credits the scan to Dale Utt. Material classification,
full-resolution reflectance derivatives, object placement, and the lamp's recomputed
smooth shade normals are described in the corresponding scene review notes.
Original scans remain intact. These derived surface models are optical scene
interpretations; the source scan is not a measured spectral BRDF.

## Poly Haven

All the following model and texture assets are [CC0](https://polyhaven.com/license).
Author names come from the saved original `info.json` records.

| Asset | Author(s) | Current scene use |
|---|---|---|
| [Gothic Cabinet 01](https://polyhaven.com/a/GothicCabinet_01) | Kirill Sannikov | kunstkammer |
| [Marble 01](https://polyhaven.com/a/marble_01) | Rob Tuytel | kunstkammer, invisible window |
| [Marble Bust 01](https://polyhaven.com/a/marble_bust_01) | Rico Cilliers | kunstkammer |
| [Oak Veneer 01](https://polyhaven.com/a/oak_veneer_01) | Jenelle van Heerden | kunstkammer, gregorian, peacock lamp, invisible window |
| [Quatrefoil Jacquard Fabric](https://polyhaven.com/a/quatrefoil_jacquard_fabric) | colormass, Rico Cilliers | kunstkammer |
| [Round Wooden Table 01](https://polyhaven.com/a/round_wooden_table_01) | Ulan Cabanilla | peacock lamp |
| [White Plaster 02](https://polyhaven.com/a/white_plaster_02) | Rob Tuytel | kunstkammer, gregorian, peacock lamp, invisible window |
| [Wood Floor](https://polyhaven.com/a/wood_floor) | Dimitrios Savva | gregorian, peacock lamp |

Ochre plaster, fumed oak, burgundy brocade and the darkened pedestal table are
project-created pigment/reflectance derivatives at original image dimensions.
Their source paths and formulas are preserved in `docs/assets/` and the asset-only
helpers under `tools/scene_assets/`. Original one-time curation helpers also changed
scene layouts and must not be rerun against the finished scenes.
Runtime texture budgeting can reduce uploaded dimensions; this does not replace
the downloaded originals. The default budget is part of the benchmark identity.

Daytime skies are project-generated Hosek–Wilkie atmospheric-model radiance maps,
not photographic HDRIs. [daylight source notes](assets/daylight_SOURCE.txt) record the model,
normalization and sun matching. Each scene retains its analytic solar emitter;
the baked map excludes the solar disc. The invisible-window and Kunstkammer
updates do not alter night skies.

## Lunar albedo

The Gregorian uses the [CGI Moon Kit](https://svs.gsfc.nasa.gov/4720/) as surface
reflectance, not emission. Credit: NASA's Scientific Visualization Studio;
Ernie Wright (USRA), Noah Petro (NASA/GSFC), with LROC camera-team source data.
The source describes these processed colour maps as optimized for aesthetics;
they are not a calibrated spectral albedo measurement.

## Invisible-window target and frozen product

The target photograph `assets/caustic_panel/cristo-vidriera.jpg` was supplied by
the user on 6 October 2026. On 26 October 2026 the user confirmed that the project
has permission to use it as the solver target and to publish renders derived from
it; the manifest's Window entries are marked `cleared` accordingly. Its photographer,
original source and church are not recorded here. Captures before that date record
the earlier `unresolved` status in their sidecars. No CC0 or public-domain claim is
made for the photograph or the derived image.

The photographed window depicts the Sacred Heart with the banderole “Ecce Cor
quod tantopere amavit homines”. The supplied photograph truncates the bottom;
the solver's crop preserves the actual aspect ratio and adds no invented image.
`models/caustic/sacred_heart_panel.json` records the original photograph SHA-256,
crop, design wavelength, forward prediction and frozen mesh hash. The photograph
itself is not redistributed: it stays in the ignored `assets/` directory, and the
repository carries only the frozen mesh, its product record and the renders.

The inverse-caustic tool is Dylan Missuwe's `poisson_caustic_design` (MIT), based
on Yue et al., TOG 2014. Its pinned version and local corrections are in the panel
sidecar; its licence is `assets/tools/poisson_caustic_design/LICENSE`. That tool
licence is separate from the target photograph's rights. The finished PMMA mesh
is ordinary optical matter; the scene does not emit or overlay the target image.

## Downloaded references not used in these four scenes

These entries are retained for provenance, not credited as visible scene content.
The removed chapel's scans and the unavailable antique-table wishlist are not
silently treated as installed scene dressing.

| Asset and original source | Author / collection | Saved licence | Current scene use |
|---|---|---|---|
| [Vault, Beauvais cathedral, France](https://sketchfab.com/3d-models/beauvais_vault-939ab33fef8c421888dbe7eba15ff1fc) | Zoilo | CC Attribution | Not used in the current four scenes |
| [Rose Window - Valencia Cathedral, Spain](https://sketchfab.com/3d-models/valencia_rose_window-628a1dba139849b282f7280feec1f2a0) | egiptologo91 | CC Attribution | Not used in the current four scenes |
| [Stained Glass Window, Southwark Cathedral](https://sketchfab.com/3d-models/southwark_stained_glass-9f24973fc9c344539a6105a1e7357fb4) | Thomas Flynn | CC Attribution | Not used in the current four scenes |

[Antique Table](https://sketchfab.com/3d-models/antique-table-39a58c86837b4baaa67e325e87a7387d),
Luke Arkell, is a CC BY 4.0 wishlist reference in the Gregorian review. The archive
was unavailable and the model is not used. The scene uses the Wawel chest instead.
