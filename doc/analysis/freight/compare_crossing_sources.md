# NPVM freight: network crossings versus interview places

Run the paired comparison with the working configuration:

```bash
uv run -m analysis.freight.compare_crossing_sources config_dib_10.yml
```

By default, Network uses the saved OD-specific routed assignments beside the
NPVM matrices. To compare direct network-border assignment instead, supply an
existing network GeoParquet explicitly, without rebuilding network conversion:

```bash
uv run -m analysis.freight.compare_crossing_sources config_dib_10.yml \
  --network-crossings /path/to/border_crossing_links.parquet
```

The output is `<analysis_path>/npvm_crossing_source_comparison/index.html`. It embeds
the charts, data and interactive crossing map; `crossings.html` also opens the map
separately. Detailed trip tables, candidate probabilities, actual crossing counts,
summary statistics and paired endpoint displacements are exported as Parquet.

The module is also a Synpp stage, `analysis.freight.compare_crossing_sources`.
Its explicit source overrides compare the same NPVM demand with saved routed
network crossings (or direct network crossings when a file is supplied) and
interview-place crossings, regardless of the production source flag. The
`npvm_freight_routing_overwrite` default is false; enable it explicitly to
regenerate saved routes.

## Controlled comparison

Both sources use the same matrices, sampled OD counts, agent ordering and seed.
Original zone identities define entering, leaving and transit. Transit is one
trip and two foreign endpoints. NPVM model-domestic geography includes
Liechtenstein/enclaves. Counts are at the configured sample fraction, with no
expansion to 100%.

The report compares total/mean/median/percentile straight-line trip distances,
departure-hour distributions, distinct crossing coordinates used and paired
foreign-endpoint displacements. Scope controls select domestic, entering, leaving,
transit, all international or all freight. It checks domestic endpoint coordinates
and departure times rather than assuming these controls remain identical.
Distances are between generated endpoints; they are not routed kilometres or
estimates of route accuracy. Displacement is a difference between two allocations
for the same sampled agents, not an observed geographic error.

The network source uses eligible truck-permitted roads (car only when no truck
mode exists) and road-class weights. The interview source uses importance scores
and its entry/exit matches; the reference does not supply truck restrictions.
Direct network and interview sources retain the best two adjacent levels
in-zone or the nearest two distinct points otherwise, with distance-weighted
fallbacks. The routed source instead uses three foreign road access candidates
and OD-specific paths, retaining paired transit entry/exit points. See
[NPVM generation](../../synthesis/freight/npvm/README.md) for the exact rules.

## Existing inputs and refreshed references

The standalone reader reuses exact existing large prerequisites and the cached
final PT network and corresponding detailed geometry. It refreshes the small
`data.cross_border.interview_places` and `data.cross_border.road_candidates` stages
so edits to the interview layer are represented. Their intermediate outputs go
under the comparison's `prerequisites/` directory, never into upstream stage cache
directories. It does not run a population build, Java network conversion or a
MATSim simulation. A missing large prerequisite fails explicitly. Shared network
reader binary caches may be reused/created normally.

This produces a separate report and leaves the NPVM-versus-legacy and freight/LCV
reports intact. The crossing map uses the attributed swisstopo WMTS background by
default, with an opacity control and optional NASA imagery. Tables and markers
remain available if background tiles fail.

## Demand map

The report embeds `demand.html`, an NPVM zone map shared by the two crossing
sources. Select domestic, all international, entering, leaving, transit, or both
domestic and international demand. Zone shading shows sampled origin or
destination demand; click a zone to see connected zones and their trip counts.
The zone incoming/outgoing control is relative to the selected zone, separately
from the Switzerland traffic filter. Green entry and orange exit markers show
sampled foreign-endpoint counts for the active selection. Transit contributes
two endpoints but remains one trip. Time and source filters apply to all counts.

Clicking a zone opens a popup and fills a persistent sidebar with entering,
leaving, transit, domestic, total-unique-trip, origin and destination counts.
These totals use source OD zone membership, the selected source and time range;
they show all traffic categories even when the map's traffic filter hides some.
A trip with both endpoints in the zone counts once in the total. Transit here
means a foreign-to-foreign OD associated with that source zone, not measured or
routed passage through a Swiss zone.
