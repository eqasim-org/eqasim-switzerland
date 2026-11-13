# Opening-hours regions for donor matching

`data.spatial.opening_hours_regions` returns a GeoDataFrame in EPSG:2056 with
`opening_hours_region`, `municipality_ids` (a JSON list) and dissolved `geometry`.
The default is three regions. Regions can be disconnected: they group similar
opening patterns, not geographical neighbours.

The stage depends only on `data.osm.businesses` and `data.spatial.municipalities`,
avoiding a dependency cycle with matching or synthetic destinations. It uses
parsed typical weekday schedules for shops and leisure businesses separately.
For each municipality it computes the fraction open at the midpoint of each
half-hour: 48 shopping and 48 leisure features. Closed schedules contribute
zero; overnight schedules wrap across midnight. Features have equal scale,
and shopping/leisure receive equal weight regardless of their relative counts.

`opening_hours_region_buffer_m: 5000.0` starts with businesses inside the
municipal polygon and within 5km of its boundary. When either purpose has too
few observations, a common radius expands in
`opening_hours_region_buffer_step_m: 2500.0` increments until it contains at
least `opening_hours_region_min_businesses: 50` shopping businesses **and**
50 leisure businesses with parsed schedules. These are business observations,
not synthetic activity counts. A mixed-purpose business can count once for
each purpose. Nearby municipalities share observations, smoothing profiles
without forcing all nearby municipalities into the same region. An initial
buffer of zero starts inside the municipality and still expands as needed.
Distances are in EPSG:2056 metres; buffers can cross borders and water.
Only business selection is buffered: returned geometries remain the original,
non-overlapping municipal polygons dissolved by cluster.

K-means uses `random_seed`, 20 initializations and
`opening_hours_region_clusters: 3`. Labels are ordered by evening availability
for reproducibility, but should not be treated as permanent classifications
across updated OSM extracts. Each municipality receives one vote in clustering.

There is no arbitrary geographic cap that would prevent reaching the target.
Expansion stops once both counts are sufficient. If the entire input dataset
cannot supply a purpose's target, the available observations are used and the
shortfall is explicitly flagged; that purpose retains the overall-profile
fallback. A finite dataset-extent bound guarantees termination. Counts,
target-met flags and imputation flags are exported. If there are too few distinct
profiles for the requested cluster count, the stage raises an error.
Each business has one vote in a municipality's neighbourhood. It can contribute
to several overlapping neighbourhoods, but has only one vote in the overall
fallback profile. `shop_businesses` / `leisure_businesses` count buffered
observations; `shop_local_businesses` / `leisure_local_businesses` count only
the municipality itself. `buffer_m` and `buffer_expansions` record each
municipality's final radius and number of extensions. The same random seed
and K-means settings apply. Spatial-index distance queries avoid repeatedly
constructing complex buffered polygons.

The municipality layer can include the configured foreign simulation area as an
additional unit. Swiss territory is subtracted from that unit first, so Swiss
businesses are attributed to their actual municipality and cluster geometries do
not overlap over an area. Shared boundaries are resolved deterministically.

Outputs in the stage cache are `regions.parquet`, `municipalities.parquet` and
`profiles.parquet`, plus `summary.json` with the seed, cluster sizes and
imputation counts, initial/step/maximum/median buffer distances, target shortfalls
and unbuffered support diagnostics. All
assignments use the same returned region geometry.

The municipality stage returns a geometry layer and an old-to-current ID
mapping; this clustering uses its current geometry layer only. Inspect the
neighbourhood radii, target flags and cluster sizes before
interpreting the regions as observed local differences. A configured foreign
perimeter, when present, remains one aggregate unit; it does not supply
individual French municipality boundaries.

To rerun only this stage against existing dependency caches in Synpp debug mode:

```python
from synpp import Synpp

pipeline = Synpp.build_from_yml(
    "config_dib_10.yml", run=["data.spatial.opening_hours_regions"])
pipeline.debug = True
regions = pipeline.run_pipeline()[0]
```

This executes just the clustering stage, reusing the business and municipality
caches without freshness validation. The seed passed to K-means is read from
`context.config("random_seed")`; it is also recorded in `summary.json`.

`analysis.opening_hours_regions` generates the review HTML and profile figure
under `analysis_path/opening_hours_municipalities`, alongside copies of the
Parquet tables and summary. The HTML includes optional municipal boundaries,
local/buffered business counts and adaptive radii, a swisstopo basemap with opacity control and
a table that remains visible when tiles fail. Display polygons are simplified
by 40m; exported Parquet geometries retain their original detail.

## Matching integration

`consider_opening_hours: true` enables this dependency in V1, V2 and V3.
When false, none requests this stage or the business extraction. The shared
`synthesis.population.matching.opening_hours.assign_regions` helper spatially
joins `home_x`/`home_y` for donors and synthetic recipients. It preserves row
order and identifiers. Outside points use the nearest region; boundary and
nearest-distance ties prefer the lowest region ID. Non-finite residence
coordinates raise an error instead of silently allowing any donor.

`opening_hours_region` is a strict donor constraint:

- V1 partitions before the relaxation recursion, including its final fallback.
- V2 adds region equality to employment/student retrieval blocks. It is not a
  model feature and does not change the learned input representation.
- V3 adds it only to the exact hard block at inference, outside the optional
  categorical distance and any learned feature inputs.

An empty compatible region pool follows each matcher's unmatched/removal
policy; it never borrows a donor from another region. This can reduce support
in rare demographic groups, which should be checked in matching diagnostics.

These are empirical profiles of OSM-mapped businesses, not legal
opening-hour rules. They represent typical weekday behaviour and can reflect
OSM coverage differences. The exported profiles/counts should be reviewed
before treating the clusters as calibration assumptions.
