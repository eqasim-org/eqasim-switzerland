# NPVM freight trip generation

`synthesis.freight.trips` selects its source using `use_npvm_freight`, which defaults
to `true`. The NPVM implementation returns exactly the existing freight columns:

```text
agent_id, origin_x, origin_y, destination_x, destination_y, departure_time, vehicle_type
```

Coordinates are EPSG:2056; departure times are seconds after midnight. Agent IDs
are consecutive integers starting at zero. Both rigid trucks (LW) and truck
combinations (LZ) use the existing `truck` mode and fleet definition, preserving
compatibility with MATSim, LCV ID offsets and other freight consumers. They remain
separate categories in the intermediate OD table.

## Stages and ordered steps

1. **`synthesis.freight.npvm.od`** reads the two daily NPVM matrices. It supports
   VISUM `$O` / `$O;D4` text matrices, reads the 00:00–24:00 interval and matrix
   factor, and excludes the `$NAMES` footer. Duplicate OD pairs are summed;
   invalid IDs, negative/nonfinite demand and non-daily matrices raise errors.
   Each cell is multiplied by `input_downsampling` and
   `npvm_freight_demand_factor`. Seeded stochastic rounding turns fractional
   demand into integer trip counts: floor plus one with probability equal to
   the fractional remainder. Zero sampled counts are removed.
2. **`synthesis.freight.npvm.zone_locations`** loads Swiss/foreign NPVM zones and
   STATENT establishments. Establishments are equally weighted; Swiss zones
   without establishments use their centroids. Additional zones without published
   polygons use the supplemental centroid CSV. Source files remain unchanged.
   **`synthesis.freight.npvm.crossing_points`** selects one crossing source and
   returns these domestic locations plus foreign crossing coordinates and
   sampling probabilities, ready for trip generation. `locations` remains a
   compatibility wrapper around this stage for existing analysis consumers.
3. **`synthesis.freight.npvm.trips`** expands sampled counts into one row per trip,
   independently samples origin and destination points, and samples departure
   hours from `freight/departure_times.csv` plus uniform seconds within the hour.
   It assigns IDs and returns the seven-column result.
4. **`synthesis.freight.trips`** returns the NPVM result directly. With
   `use_npvm_freight: false`, it registers only the original GTE/GQGV dependencies
   and retains the original seeded trip/departure-time logic.

All random draws use the configured `random_seed`. Stages copy data before
modification. Matrix and zone-file stat tokens invalidate their caches when
inputs change in place. The population build and MATSim do not need to run to
validate or build these freight-only stages.

## Configuration

The working configuration `config_dib_10.yml` uses NPVM through the new default.
To select explicitly:

```yaml
config:
  use_npvm_freight: true
```

To build only freight, use a configuration with:

```yaml
run:
  - synthesis.freight.trips
```

Then run `uv run -m synpp your_config.yml`. Set `use_npvm_freight: false` to select
the old survey method. The dependency graph changes with this option; the
unselected method's source stages are not required by freight.

Advanced defaults, all relative to `data_path` unless absolute paths are supplied:

| Option | Default |
|---|---|
| `npvm_freight_lw_file` | `npvm/LW_NPVM_2023.mtx` |
| `npvm_freight_lz_file` | `npvm/LZ_NPVM_2023.mtx` |
| `npvm_freight_swiss_zones_file` | `npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg` |
| `npvm_freight_foreign_zones_file` | `npvm/Verkehrszonen_Ausland_NPVM_2023.gpkg` |
| `npvm_freight_additional_centroids_file` | `npvm/NPVM_2023_additional_zone_centroids.csv` |
| `npvm_freight_demand_factor` | `1.0` |
| `npvm_freight_use_border_crossings` | `true` |
| `npvm_freight_crossing_source` | `null` (defaults to `route`; explicit choices: `route`, `network`, `interview`) |
| `npvm_freight_routing_overwrite` | `false` (read saved routed crossings; `true` reroutes and replaces them) |
| `npvm_freight_routed_crossings_file` | `null` (`freight_routed_crossings.parquet` beside the configured LW matrix) |
| `npvm_freight_route_border_crossings` | `true` (compatibility option when crossing source is not explicit) |
| `npvm_freight_use_interview_places` | `false` (compatibility option when crossing source is not explicit) |
| `npvm_freight_border_crossings_file` | `null` (use configured network-stage GeoParquet) |

NPVM 2023 matrices represent average-weekday vehicle demand, including their
international OD entries. They are used as supplied, with no automatic weekend
adaptation. The existing passenger `include_cross_border` option does not remove
international freight, consistent with the legacy freight method.

**The legacy `enable_scaling` / freight-projection factor is not applied to NPVM.**
That factor is calibrated against the GTE/GQGV survey demand, not these NPVM
matrices. Setting population `scaling_year: 2025` does not itself turn NPVM 2023
freight into 2025 freight. Use a suitable matrix release and/or an explicitly
chosen `npvm_freight_demand_factor` for that adjustment. `input_downsampling`
always applies once to NPVM demand.

## Select and reuse crossing points

Use one source in the configuration's `config` section:

```yaml
npvm_freight_crossing_source: route  # route, network, or interview
npvm_freight_routing_overwrite: false
```

- `route`: read the saved OD-specific entry/exit candidates from
  `freight_routed_crossings.parquet`, beside `npvm_freight_lw_file` in the input
  data directory. This does **not** load the network or route anything.
- `network`: use the exported border links and choose directional candidates by
  foreign region, hierarchy and distance.
- `interview`: use `data.cross_border.interview_places` through its existing
  directional road matcher and importance scores.

To create or refresh the routed file, set `npvm_freight_routing_overwrite: true`
for a freight run, then set it back to `false`. Explicit overwrite reroutes on
subsequent runs even if a Synpp result is cached. A complete, validated result is
written to a temporary file and replaces the saved file only after success.
`npvm_freight_routed_crossings_file` can override the destination, relative to
`data_path` or as an absolute path.

The saved table contains source/destination zone IDs, entry/exit coordinates and
link IDs, foreign access coordinates, travel times and normalized probabilities.
It retains OD pairs because a foreign zone may use different entrances for
Swiss destinations, and transit entry/exit points must remain paired. Counts and
trip departure times are not persisted here: later stages sample crossings from
these probabilities for their own demand.

The file covers the OD pairs in the configuration used to generate it. If a new
sample or matrix requests additional pairs, reuse fails with an explicit message
asking for overwrite. Seed, network or source-data changes do not automatically
reroute when overwrite is false; enable it deliberately when updating crossings.
The previous validated 10% routed results have been saved for the working
configuration. Downsampling is still applied only to demand.

An explicit `npvm_freight_crossing_source` takes precedence over the older
`npvm_freight_use_interview_places` and `npvm_freight_route_border_crossings`
booleans. When the source is omitted, those booleans retain their previous
selection behavior. `npvm_freight_use_border_crossings: false` disables all road
crossing sources and keeps geometric boundary placement.

## Official inputs and additional centroids

The [NPVM 2023 release](https://zenodo.org/records/23009199) provides the truck
matrices in `3_Fahrtenmatrizen_Strasse_NPVM_2023.zip`, Swiss polygons and the
foreign-zone archive. The latter contains
`Verkehrszonen_Ausland_NPVM_2023.gpkg` (whose ID column is `ID_Zone`, compared
with `No` in the Swiss layer). The input directory also contains the supplemental
centroid CSV and a `.source.json` sidecar recording its source and checksum.

The 21 additional IDs used by these LW/LZ matrices are absent from the published
polygon layers. Their coordinates were extracted from `Network/Zone.bin.zst`
inside the same release's `5_NPVM_2023_DWV_MIV_Umlegung.ver`. Only the ZIP directory
and that member were fetched using HTTP ranges, with the member CRC verified.
The binary ID/x/y layout was checked against the published Swiss centroid layer:
7,977 of 7,978 coordinates agree within 1 cm; zone 520301001 differs by 269 m
between releases. The supplemental file keeps the exact VISUM coordinates in
EPSG:2056 rather than estimating missing locations. No binary parsing or download
occurs during pipeline execution. Alternative matrices may need their own
matching supplementary centroids, using columns `zone_id,x,y,name`.

## OD explorer

[The HTML explorer](../../../analysis/freight/trips_html.md) follows
`use_npvm_freight`: it reads aligned sampled NPVM OD counts for NPVM freight and
uses the original NUTS reconstruction for legacy freight. It does not add public
columns to the freight trip contract. Additional source zones without polygons
are shown as circles at their source coordinates; their generated trip endpoints
remain on the Swiss border. Existing standalone HTML files remain snapshots of
the source configuration at generation time. Rebuild the relevant caches and
regenerate the report to inspect the new freight source.

## Road crossings and weighted freight assignment

The default network source now uses the OD-specific routing described below.
The direct crossing-pool rules in this section apply when
`npvm_freight_route_border_crossings: false`; interview scoring remains separate.

`matsim.scenario.network.convert_osm` exports general network data to
`border_crossing_links.parquet`: final link/node IDs, every allowed `modes` value,
the original highway tag, full directed geometry, exact intersection, `in`/`out`
direction and a routing point shifted one metre right. It tags the corresponding
links `ch_border_crossing=true`. It does not rank roads or filter modes for freight.
See [network crossing export](../../../matsim/scenario/network/border_crossings.md).

The freight rules live in `synthesis.freight.npvm.crossings`. `ROAD_HIERARCHY`
groups roads as follows; ramps share their parent level:

| Rank | Roads | Relative weight per point |
|---|---|---:|
| 0 | motorway, motorway_link, trunk, trunk_link | 8 |
| 1 | primary, primary_link | 4 |
| 2 | secondary, secondary_link | 2 |
| 3 | tertiary, tertiary_link | 1 |
| 4 | unclassified, residential, living_street, service, track, road | Excluded |

Unknown highway types and non-road links are excluded. Multiple highway tags are
resolved to their highest eligible class here, leaving the network export intact.

For each foreign zone, separately for `in` and `out`:

1. **Filter modes and classes.** If `truck` occurs anywhere in the crossing export,
   use only crossings whose modes include `truck`. Otherwise use `car`. Exclude
   the final road category above. Do not silently fall back to car within a zone
   or direction when truck mode exists elsewhere. Missing eligible directions fail.
2. **Remove duplicate physical points.** Split links can identify the same boundary
   intersection. Group intersections within the same millimetre coordinate and
   direction, retaining the best class and then the smallest link ID.
3. **Choose candidates.** Match the zone against the unshifted intersection or
   foreign approach point. Within a zone, keep its best present rank and exactly
   the next rank (`best + 1`), if present. Motorway + primary + secondary keeps
   motorway/primary; motorway + secondary keeps motorway only. When no crossing
   belongs to the zone, choose the two nearest distinct eligible points by distance
   to the zone geometry (or one if only one exists). Class and link ID resolve ties.
4. **Sample trips.** Normalize the per-point weights above within that candidate
   pool and sample with the configured random seed. A motorway point receives
   twice the probability of a primary point. Points in the same class have equal
   weight within a zone. For the nearest-two fallback, divide each hierarchy
   weight by `1 + distance_to_zone_m / 1000` before normalizing. Distance is from
   the exact intersection to the region geometry. For equal road classes, the
   farther point receives less probability. Higher road class can still outweigh
   a distance disadvantage. The added 1 km keeps the formula finite at zero distance.

The location stage returns `border_points[direction][zone_id]` sampling arrays and
matching `border_probabilities` arrays. `border_assignments` has one row per
candidate/zone/direction, with link ID, modes, `assignment_mode`, hierarchy,
`probability`, selection method and original/shifted coordinates. Probabilities
sum to one per zone/direction; the number of candidates can vary.

Trip expansion uses `in` for foreign origins and `out` for foreign destinations,
including both endpoints of transit. Domestic points still sample establishments
equally. Geometric placement (`npvm_freight_use_border_crossings: false`) retains
its original single-point behavior. The public seven-column trip DataFrame and
`truck` vehicle type remain unchanged, including when car mode filters crossings.

This is a zone-based assignment heuristic, not observed crossing counts or a full
OD route choice. Coordinates shifted right do not force the final link ID: the
inspected cached Java `FreightAssignment` replaces freight coordinates with nearby
facilities chosen using link coordinates.

The default network dependency is conditional and acyclic. Older crossing exports
must be regenerated if they lack modes, directions or shifted-point metadata.
For an existing GeoParquet, set `npvm_freight_border_crossings_file` (relative to
`data_path` or absolute). This removes the conversion dependency and validates
file changes. Standalone analysis readers reuse existing prerequisites and do
not launch Java/network conversion.

## Optional interview-place source

Enable the alternative in the configuration's `config` section:

```yaml
npvm_freight_use_border_crossings: true
npvm_freight_crossing_source: interview
```

For the interview source, `npvm_freight_border_crossings_file` is
ignored. Disabling `npvm_freight_use_border_crossings` still selects geometric
placement and does not request either crossing source.

The canonical `data.cross_border.interview_places` layer has road/PT labels and
an `importance` score (currently 1–5), but no direction field. Freight reuses
`data.cross_border.road_candidates`, which already consumes that layer and matches
entry and exit roads using the final PT network and its corresponding geometry.
It reads that producer's `directional_links.csv` alongside its returned point
table; upstream caches are read-only. The existing file is reused, not converted
into another CSV. Normal Synpp execution refreshes these dependencies when the
selected interview reference changes. This source therefore still depends on
network construction through the existing directional matcher.

`from_interview_places(places, directional_links=None)` is the adapter in
`synthesis.freight.npvm.crossings`:

- Keep road references only, omit zero scores and duplicate aliases identified
  by the producer; reject invalid scores and coordinates.
- Preserve original point IDs and scores. `hierarchy = maximum importance − score`
  and `sampling_weight = importance`. Inside a zone, keep the best score and the
  next score level. For scores 5 and 4, weights are 5:4; for nearest-two fallbacks,
  divide these weights by `1 + distance to zone in km` before normalization.
- Reuse each point's existing entry/exit link match. Project its source location
  onto that directed curve and move one metre right. These projected anchors
  need not be exact national-boundary intersections. The original interview
  coordinate is retained as `foreign_x/foreign_y` for zone matching.
- If no directional curve is supplied for a point, treat it as bidirectional at
  its source coordinate, with `right_offset_m: 0` and
  `direction_source: assumed_bidirectional`. Its `link_id` is then a synthetic
  interview reference, not an actual network link. No road bearing is invented. An
  explicit input `direction` (`in`/`out`, or `entry`/`exit`) is respected.

The adapter returns the same fields needed by the location/trip sampler, with
`crossing_source: interview_places` and source/score/direction audit columns.
Interview references describe car-accessible road crossings; they do not provide
truck access restrictions. Their `assignment_mode` is therefore `car`, rather
than inventing truck permissions. This changes the crossing selection basis, not
the freight vehicle type or daily OD demand. Geometry, source points and cached
input tables are copied before transformation. Reports record which source was
used; the default production configuration is not switched automatically.

## OD-specific routing from external road access points (default network source)

`npvm_freight_crossing_source: route` reuses the saved routing table by default.
The following steps run only with `npvm_freight_routing_overwrite: true`.
Select `network` for foreign-zone-only pools, or `interview` for interview places.

1. `synthesis.freight.npvm.network_access` reads **`network.pkl` from
   `matsim.scenario.network.convert_osm`**, not network XML. It attaches that
   producer's matching `detailed_network.csv` to a local copy. Truck-permitted
   roads are used whenever the mode exists, otherwise car-permitted roads.
   Local roads remain usable inside the routing graph but excluded local/unknown
   crossing classes cannot cross the Swiss border.
2. For every foreign zone with sampled positive demand, sample **three positions
   on eligible foreign road segments intersecting that zone**. Eligible access
   classes are motorway/trunk, primary and secondary, including their ramps.
   Positions are sampled on the clipped road geometry, not on water or arbitrary
   polygon land. Link-class weights are 4:2:1 (equivalent to the existing rank
   weights). Sampling uses the configured seed. Links must lie wholly outside
   Switzerland so partial-link direction and travel time remain unambiguous.
3. If no eligible foreign road intersects the zone, use the **three nearest
   distinct physical roads**, projecting the zone onto each full directed road
   geometry. Distances are line-to-region distances. Opposite links with the same
   endpoint pair represent one physical fallback road. Use fewer only when the
   network contains fewer than three eligible roads. Weights include hierarchy
   and `1 / (1 + distance_to_zone_m / 1000)`.
4. `synthesis.freight.npvm.routed_crossings` snaps a representative establishment nearest each domestic zone's mean
   to a permitted node (inside Switzerland for physically Swiss endpoints), and routes each access
   candidate against that Swiss endpoint. Searches are batched by foreign access
   node; outbound searches use the reversed graph. Repeated node-pair paths are
   reused. Each directed path uses free-flow `length / freespeed` travel time.
   This is an OD-zone representative approximation; it does not route every
   individual establishment-to-establishment trip or model congestion.
5. Transit combines an exterior approach, a positive-length route entirely in
   Switzerland, and an exterior departure. It considers the three nearest
   reachable entry/exit candidates in the appropriate exterior components and
   rejects identical physical entry/exit points. This explicitly conditions
   foreign-to-foreign demand on using Switzerland; it is not an unconstrained
   Europe-wide route inference.
6. Candidate probabilities combine the access weights with
   `1 / (1 + route_travel_time_seconds)` and normalize per OD pair. Transit entry
   and exit are sampled jointly from the same candidate route. The returned trip
   coordinates are the path's first Swiss entry and last Swiss exit, with the
   directed one-metre right offset. Swiss endpoints retain establishment sampling.
   Access coordinates and selected link IDs remain in the routed stage's audit
   DataFrame. Unreachable OD pairs raise an explicit error; demand is never dropped.

`network_access` returns the road/graph result for Synpp caching;
`routed_crossings` returns the OD candidate/probability table. The seven-column
freight result, vehicle type and input downsampling remain unchanged. Actual
available road coverage is used; `border_offset` defaults to 50 km around the
simulation region, but the original OSM extract and cleaning constrain coverage.
Changing the sampling seed changes trip sampling. Regenerating access/routing
results with the new seed requires explicit overwrite; no network conversion is
needed when its inputs remain unchanged. The general converter and its border
GeoParquet remain independent of freight hierarchy and routing policy.

NPVM model-domestic zones include Liechtenstein and enclaves. If such an OD
routes wholly outside Switzerland, it retains its sampled foreign road access
coordinate rather than inventing a Swiss crossing. `outside_model_domestic` and
`uses_swiss_crossing` flag these candidates in the routed audit. Source-zone
incoming/outgoing classification therefore remains distinct from actual Swiss
border passage. Separate seeded streams for crossing selection and domestic
sampling/departure times keep Swiss endpoints and times comparable across the
network and interview crossing sources.
