# Kunstkammer window joinery audit, 9 October 2026

Static physical-fit candidate only. The main scene is unchanged; adoption and
benchmark workload identity belong to the root review. No renderer, build, GPU
or image validation was run for this audit. The candidate retains all materials,
lights, detector settings and assets; asset/model paths are absolute so it can
be rendered from this evidence directory. `prepare_candidate.py` reproduces it
and `bounds_audit.json` records exact decimal box bounds, overlap lengths and
source/candidate SHA-256 identities.

## Confirmed original overlaps

- Bottom rail z [.7375,.8225] overlaps the sill and LeftSillWall by 12.5mm.
  Top rail z [4.1875,4.2725] overlaps LeftLintel by 22.5mm.
- Every rail y [-2.96,1.56] crosses Jamb1 y [-.065,.065] and extends 25mm
  into each end jamb. These are positive volumes, not numerical contacts.
- Jamb0 y [-3.065,-2.935] extends 65mm into LeftFront; Jamb2
  y [1.535,1.665] extends 65mm into LeftRear.
- Sill x [-3.875,-3.225], y [-3.1,1.7], z [.62,.75] overlaps plaster
  in LeftSillWall and the front/rear end walls. Its exposed top is coplanar
  with the old plaster top at z .75; deleting the limestone cap would change
  the intended visible material.
- Bead2 radius .033 at x -3.47, y 1.6 extends 3 mm behind the inner wall plane
  x=-3.5 and into LeftRear for y > 1.6. The other two beads occupy the open
  aperture at their y coordinates and have no such backing-wall overlap.

## Exact candidate changes

| Part | Candidate extent/change | Preserved visible geometry |
| --- | --- | --- |
| Rail0 | z [.75,.8225], height .0725, center z .78625 | Top edge .8225; buried bottom clipped |
| Rail1 | z [2.0575,2.1425] unchanged | All vertical edges |
| Rail2 | z [4.1875,4.25], height .0625, center z 4.21875 | Bottom edge 4.1875; buried top clipped |
| All rails | Two y spans [-2.935,-.065] and [.065,1.535], lengths 2.87/1.47, centers -1.5/.8; x unchanged | Aperture and exposed rail ends; removed portions were inside jambs |
| Jamb0 | y [-3,-2.935], width .065, center y -2.9675 | Exposed inner edge -2.935 |
| Jamb2 | y [1.535,1.6], width .065, center y 1.5675 | Exposed inner edge 1.535 |
| Jamb1 | Unchanged | Central divider |
| Sill | Unchanged | Limestone cap and its full exposed top |
| Bead2 | x -3.47→-3.467, other coordinates/radius unchanged | A documented 3 mm outward fit; tangent to x=-3.5 |
| Bead0/1 | Unchanged | No proven backing-wall overlap |

The plaster is partitioned around the original stone volume, rather than
moving the cap or changing the union's outer boundary:

- LeftSillWall: full footprint below z .62, plus x [-3.95,-3.875] outer
  strip for z [.62,.75].
- LeftFront/LeftRear: full footprints below z .62 and above z .75; at cap
  height z [.62,.75], retain outer x [-3.95,-3.875] strips and the
  non-notch inner spans x [-3.875,-3.5] at y [-5,-3.1] / [1.7,3].

The exact-decimal audit finds 17 original positive-volume overlap pairs among
these window boxes, zero in the candidate. Tangent contacts are retained; this
is not a promise of zero render diagnostics. WindowBead2 tangency is checked
analytically, not by its axis-aligned bounding box. Float32 conversion, contact
ordering and all geometry outside this window scope still require renderer
validation. No global wainscot/furniture audit or change was made.
