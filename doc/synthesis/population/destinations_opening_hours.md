# Destination opening hours

`synthesis.population.destinations_opening_hours` enriches the existing
destination catalogue with separate typical weekday hours for shopping and
leisure. It reads `synthesis.population.destinations`, joins NOGA from
`data.statent.statent` using destination/enterprise IDs, and reads the parsed
business schedules from `data.osm.businesses`.

Add the stage to the configuration's existing `run` list:

```yaml
run:
  - synthesis.population.destinations_opening_hours
config:
  destinations_opening_hours.match_radius_m: 50.0
  destinations_opening_hours.local_radius_m: 2000.0
  destinations_opening_hours.max_radius_m: 10000.0
  destinations_opening_hours.min_donors: 3
  destinations_opening_hours.max_neighbors: 128
```

These settings show the defaults. Existing input/scenario configuration is
still required. Distances use EPSG:2056 metres. Require
`0 <= match_radius_m <= local_radius_m <= max_radius_m` and
`max_neighbors >= min_donors >= 1`.

## Assignment rules

Only destinations offering shopping or leisure receive schedules. Other
purpose options remain `not_applicable`. A facility offering both gets two
independent schedules.

1. All destinations offering shopping or leisure are considered, including
   foreign destinations and other non-STATENT sources. Without NOGA, nearby
   businesses of the same broad purpose supply observed or inferred hours.
   Non-STATENT destinations no longer default to 24-hour opening.
2. STATENT NOGA determines compatible OSM business groups: supermarkets,
   bakeries, pharmacies, restaurants/cafes, bars, fitness centres, etc. The
   nearest compatible OSM business within 50 metres supplies its full schedule.
   This is a proximity-based association, not a confirmed identity match.
3. Otherwise, use the most common complete opening schedule among nearby
   compatible businesses. Consider up to the nearest 128 donors within 10 km;
   use those within 2 km when at least three are available. Expand to 10 km
   otherwise. One or two donors are accepted but labelled `sparse_*_mode`.
   Each OSM object has one vote. Ties prefer the nearest supporting donor,
   then its OSM ID for reproducibility.
4. Known subtypes never borrow schedules from incompatible subtypes. Missing
   or unresolved subtypes can use a broader purpose pool, explicitly labelled
   `*_purpose*`. A supplied NOGA that contradicts the destination purpose is
   labelled `noga_purpose_mismatch` rather than assigning unrelated hours.
5. If no suitable donor is found, hours remain `unknown`. A directly
   matched, known closed schedule is retained; closed-only businesses are not
   used to infer other destinations' hours.

`data.activity_purposes` supplies the shared classification used by STATENT,
OSM and microcensus labels. Retail follows NOGA 47. Leisure includes gastronomy
56, arts 90, museums/heritage/gardens 9102/9103/9104, gambling 92, sport and
recreation 93, and cinemas 5914. The input uses
[BFS NOGA 2008](https://www.bfs.admin.ch/bfsstatic/dam/assets/344621/master),
so museum/garden codes use `9102`/`9104`, rather than the previous `912`/`914`.
Accommodation and services such as hairdressers are not shop/leisure donors.

## Result and provenance

The Python result is the enriched GeoDataFrame, preserving destination IDs,
row order, geometry and existing offer flags. The stage writes:

- `destinations.parquet`: the complete enriched destination catalogue.
- `opening_hours.parquet`: one audit row per eligible destination/purpose.
- `summary.json`: source/status counts, settings and time units.

The catalogue adds `noga`, `is_statent_destination` and these fields under both
`shop_` and `leisure_` prefixes:

| Field suffix | Meaning |
| --- | --- |
| `opens_at`, `closes_at` | Typed Parquet clock times for the first opening and final closing |
| `closes_day_offset` | 0 for same-day closing, 1 for next-day closing |
| `open_intervals` | JSON minute intervals; retains lunch breaks and overnight sessions |
| `hours_status` | `open`, `closed`, `unknown` or `not_applicable` |
| `hours_source` | Direct proximity, local/regional/sparse inference or failure reason |
| `hours_business_type` | Compatible business group used for matching |
| `hours_donor_osm_id`, `hours_distance_m` | Representative donor and its distance |
| `hours_radius_m` | Radius used for the match or inference |
| `hours_donor_count`, `hours_support_count`, `hours_support_fraction` | Considered donors and support for the selected schedule |

Times are local to `Europe/Zurich`. `[[0,1440]]` means 24 hours, represented by
00:00 opening, 00:00 closing and offset 1. `[[1080,1560]]` means 18:00–02:00.
Use the complete intervals to respect lunch breaks; the first/last clock times
alone do not imply continuous opening. `[]` means known closed, while null
means unknown or not applicable.

The OSM input supplies typical hours across open weekdays and seasons, as
described in the [business-stage guide](../../data/osm/businesses.md). These
schedules are typical clock times, not date-specific holiday/season calendars.
Donor support measures agreement in the selected sample, not a calibrated
probability that an individual facility follows that schedule.

## RDA assignment

`consider_opening_hours: true` (default) enables these schedules in
`synthesis.population.spatial.secondary.locations_rda`. Setting it to `false`
uses the original destination catalogue and does not request this stage or
`data.osm.businesses`. The NN assignment is unchanged.

Only shopping and leisure are filtered. The whole activity, from arrival to
the next departure, must fit a continuous open interval. Lunch breaks and
overnight sessions are respected, with typical weekday schedules repeated
daily, including after midnight. These are typical hours, not dated seasonal
or weekend calendars. Final variable activities use the same 30h horizon
(extended to at least one minute after arrival) as the activities stage.

`opening_hours_tolerance_minutes: 10.0` allows a total of ten minutes of
boundary mismatch per activity: early arrival plus late departure must stay
within that budget. It is not ten minutes at each end. Set it to zero for
strict coverage. Lunch breaks remain separate intervals; a stay cannot bridge
them unless its mismatch to one continuous interval fits the budget. Known
closed schedules remain excluded. The filter does not alter trip times.

When opening hours are enabled, the three donor-matching stages also use the
[municipality opening-hours regions](../../data/osm/opening_hours_regions.md) as
strict residence-region masks.

Schedules are parsed once into numeric arrays before multiprocessing. A
Numba JIT kernel checks candidate eligibility without parsing JSON or scanning
the destination catalogue during assignment. The existing nearest-neighbour
search doubles its candidate count when necessary to obtain up to 20 eligible
destinations, then uses the existing employee weights. Public-transport access
filtering applies to the schedules and destinations together.

Unknown schedules remain unrestricted, since they provide no evidence of a
closure; known closed schedules are excluded. If no eligible or unknown-hours
destination exists for a requested interval, assignment raises an explicit
error rather than silently assigning a known closed facility. Trip times are
not changed by this filter.

## RDA performance

Distance sampling uses binary search in sorted CDFs rather than scanning every
survey distance for every draw. Distributions for each fixed trip leg are
selected once per assignment problem and reused across retries. Duplicate CDF
values retain the original strict-comparison rule; non-monotone or non-finite
CDFs retain the original scan. Random draws remain in the same order.

The compiled gravity loop reuses its scratch arrays across iterations and
caches its compiled code. Destination coordinates are extracted using
vectorized geometry access. Free-chain sampling avoids allocating a full array
of candidate indices on every draw. Solver thresholds and iteration limits
are unchanged.

A focused synthetic benchmark (50,000-entry CDFs, four trip legs, 10,000 draws)
measured approximately 4.7× faster distance sampling; 3,000 warmed-up gravity
calls measured approximately 3× faster execution. Fifty seeded complete
assignment solves matched the previous destination IDs, coordinates, validity
flags, objectives and iteration counts exactly. These component timings do
not establish an overall speedup for a full population run; routing/detour
lookups, spatial searches and worker load balance may still dominate.
