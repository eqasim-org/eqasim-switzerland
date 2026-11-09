# Opening-hours regions for donor matching

`data.spatial.opening_hours_regions` returns a GeoDataFrame in EPSG:2056 with
`opening_hours_region`, `canton_ids` (a JSON list) and dissolved `geometry`.
The default is three regions. Regions can be disconnected: they group similar
opening patterns, not geographical neighbours.

The stage depends only on `data.osm.businesses` and `data.spatial.cantons`,
avoiding a dependency cycle with matching or synthetic destinations. It uses
parsed typical weekday schedules for shops and leisure businesses separately.
For each canton it computes the fraction open at the midpoint of each
half-hour: 48 shopping and 48 leisure features. Closed schedules contribute
zero; overnight schedules wrap across midnight. Features have equal scale,
and shopping/leisure receive equal weight regardless of their relative counts.

K-means uses `random_seed`, 20 initializations and
`opening_hours_region_clusters: 3`. Labels are ordered by evening availability
for reproducibility, but should not be treated as permanent classifications
across updated OSM extracts. Each canton receives one vote in clustering.

For a purpose with fewer than `opening_hours_region_min_businesses: 20`
observations in a canton, the overall observed purpose profile is substituted.
Counts and imputation flags are exported. If there are too few distinct
profiles for the requested cluster count, the stage raises an error.

The canton layer can include the configured foreign simulation area as an
additional unit. Swiss territory is subtracted from that unit first, so Swiss
businesses are attributed to their actual canton and cluster geometries do
not overlap over an area. Shared boundaries are resolved deterministically.

Outputs in the stage cache are `regions.parquet`, `cantons.parquet` and
`profiles.parquet`. All assignments use the same returned region geometry.

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

These are empirical profiles of OSM-mapped businesses, not legal canton
opening-hour rules. They represent typical weekday behaviour and can reflect
OSM coverage differences. The exported profiles/counts should be reviewed
before treating the clusters as calibration assumptions.
