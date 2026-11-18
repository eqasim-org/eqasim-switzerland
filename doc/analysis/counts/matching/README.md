# Traffic-count matching

`TrafficDataMatcher` is the public matching API. A count record has one of two
meanings:

| Mode | Meaning of the observed flow | Required MATSim links |
|---|---|---|
| `directional` | Flow in one direction | One link in that direction |
| `bidirectional` | Flow aggregated across both directions | Two opposite links |

Directional mode is selected automatically when the prepared count data has
both `osm_id` and `angle`. The OSM ID selects the way and the angle selects its
directed MATSim link. Geneva and Zurich currently use this mode.

Directional matching tries the supplied OSM ID, then historical `old_osm_id`
provenance. If neither yields a compatible direction, it searches within 50 m
of the actual detector point and accepts only links within 45 degrees of its
supplied angle. Supplied MATSim link IDs and old link midpoints are never used
to choose a link. The nearest compatible candidate wins; no opposite-direction
fallback is allowed. If nothing qualifies, the detector remains unmatched.

All other current canton datasets contain bidirectional totals. Their callers
therefore pass `mode="bidirectional"` explicitly. Point stations are matched to
the closest valid opposite pair. Line stations use the closest parallel link in
each direction. A missing opposite link means that the station is left
unmatched; a one-link fallback would undercount the simulated bidirectional
flow.

## Swiss ASTRA OSM overrides

Edit `analysis/counts/cantons/ch_overrides.py`, dictionary
`STATION_OSM_OVERRIDES`, to override a Swiss station. Keys are station numbers;
each value lists the measured branches contributing to that station's total.
Stations 225, 525, 547 and 241 have four branches each; station 24 has two.
The registry also contains reviewed footprints for 21 stations from the
complex-junction blacklist. Only stations in `REINSTATED_STATION_IDS` bypass
that blacklist, and only after a complete Swiss override resolves. The current
selection keeps the 11 stations with hourly GEH strictly below 6 in the
`rda_not_finished` iteration-100 weekday comparison: 6, 35, 123, 157, 285, 297,
507, 598, 618, 760 and 852. Stations 68, 83, 112, 277, 283, 296, 577, 675, 799
and 847 remain blacklisted. Their anchors remain available for diagnostics.
The original four overrides and station 241 are outside this additional GEH
selection. Volume, distance, flow-discrepancy and outlier filters are unchanged.

`analysis.counts.target` uses the same override dictionary, retains the override
flag when grouping matched branches, and calls the same manual-exclusion filter.
The approved list is fixed from that reviewed simulation, so pre-simulation
target generation requires no simulation outputs or GEH calculation. Registry
and shared-filter edits invalidate the target through its validation hash.

Each branch contains `group`, `osm_id`, LV95 `x`/`y`, `angle` (degrees
counter-clockwise from east), and ASTRA `direction` (1 or 2). Positions and angles
select the right directed segment when one OSM way produces several MATSim links.
Do not add one branch per lane: a MATSim link already includes all its lanes.
For example, station 225's Bern branch is:

```python
dict(group="bern", osm_id="290361969", x=2627767.399, y=1240563.993,
     angle=-91.8, direction=1)
```

The Swiss run passes this dictionary to `TrafficDataMatcher.match` through
`station_overrides`. It resolves current OSM IDs or recorded `old_osm_id` values,
within 50 m and 45 degrees of the local directed tangent. It does not fall back
to unrelated roads. Missing groups, duplicate selected links, or adjacent serial
links in the same direction leave the entire station unmatched. Other stations
retain normal matching. OSM ways may be split or replaced over time; review the
dictionary when an override no longer resolves. Removing a station from the
dictionary restores automatic matching.

Comparison sums the distinct branch flows once against one observed station
total and requires statistics for every selected branch. Four branches still
represent ASTRA's two directions for GEH; the saved results include
`measurement_groups_matched` separately. The Swiss stage saves
`station_osm_overrides.csv` with resolved current link IDs, distances, angles and
whether current or historical OSM provenance was used.

The Swiss stage also saves `station_filter_audit.csv` before dropping matched
stations, with observed/simulated totals, manual exclusion, low-volume and
flow-discrepancy flags. These flags distinguish a resolved spatial match from a
record subsequently removed by existing comparison filters.

The sensor inventory lives beside the ASTRA workbooks under
`<counts_path>/Switzerland/astra_sensors/`:

- `sensor_inventory_20261118.json`: complete downloaded service snapshot.
- `sensors_20261118.csv` and `.gpkg`: all 1,915 sensor locations and attributes.
- `reviewed_measurement_groups.csv` and `.gpkg`: 81 reviewed groups at 26 stations.
- `sources.json`: service query, CRS, date and identifier conventions.

The original `Precise_det_loc.csv` is preserved. The group exports document the
Python registry; edits to those exports alone do not change pipeline matching.
Optional `sensor_ids` in a registry entry records the lane sensors represented
by that group. Repeated normal/overtaking lane sets can belong to separate
branches at the same station: group by physical carriageway, not just ASTRA
direction or lane name. Station 68 has two measured branches in one official
direction; its GEH direction count is one rather than two.

The November 2026 review recovered spatial matches for excluded stations 6, 35,
68, 83, 112, 123, 157, 277, 283, 285, 296, 297, 507, 577, 598, 618, 675, 760,
799, 847 and 852. Station 241 was added separately. Six excluded stations have
no sensor records in this snapshot (69, 164, 221, 311, 540, 690); six others have
unresolved branch/virtual-station footprints (257, 278, 605, 845, 846, 851).
Nearest-sensor matching would select duplicate or serial links for these latter
cases. They require a reviewed counting formula or additional road/lane evidence
before reintroduction.

Combined quality filters allow these explicit mixed mainline/ramp footprints,
while retaining distance and volume checks. Mixed road types are labelled
`mixed` in summaries instead of attributing the whole station total to each
branch's type. Matching edits invalidate the Swiss stage through its validation
hash. The overrides do not alter observed counts or incomplete-year handling;
Rolle's 2025 observation still uses the existing nine-month fallback.

## Files

- `matcher.py`: public API, mode validation and orchestration.
- `road_matching.py`: focused angle, point-pair and line-pair algorithms.
- `counts.py`: loading and normalizing authority count data.
- `network.py`: MATSim network access and geometry handling.
- `compare.py`: observed-versus-simulated flow calculation.
- `results.py`: per-canton result persistence.
- `plots.py`: static and interactive analysis plots.
- `geometry_io.py`: detailed-network geometry parsing.
- `network_from_prepare.py`: pre-simulation SynPP network stage used by the
  calibration target.

## Transcality detailed simulation tooltips

`correct_transcality_counts` defaults to `true`. Set it to `false` under the
pipeline YAML's `config:` section to disable all manual/SITG total and range
corrections and retain detector `316.2`. This switch applies to daily and
peak-hour preparation and therefore to comparisons and calibration targets.
Existing matching, completeness and construction filters still apply. Changing
the switch selects a separate stage cache; disabled corrections do not read the
optional SITG correction file.

Transcality preparation applies the reviewed daily corrections in
`cantons/transcality.py:DAILY_FLOW_OVERRIDES` and removes detector `316.2`.
The ASTRA detectors `astra_200_4`, `astra_200_5`, `astra_200_6` and `2071.2`
share one corrected total of 28,327 vehicles/day. Their existing profiles are
scaled by a common factor before network matching and aggregation, preserving
their relative contributions. The other corrected daily totals are 11,207 for
`2017.1`, 6,691 for `315.2`, 5,533 for `017.0.102`, 11,196 for `017.0.103`
and 2,276 for `035.0.106`. The code comment records the supplied SITG source.

Hourly profiles and lower/upper envelopes use the same scaling factor; these
remain scaled Transcality profiles, not newly measured hourly data or quantiles.
Rounding preserves the exact corrected daily sum. Peak-hour preparation and
calibration targets use the same corrected inputs, and peak-hour cache validation
tracks changes to the shared preparation code. An incomplete four-detector group
raises an error rather than assigning its full total to a partial match.

Additional SITG corrections are read from
`<counts_path>/Transcality/sitg/transcality_overrides_2025.csv` when present.
This file is reviewed against a saved official snapshot; the pipeline does not
download data during execution. Its content hash invalidates the daily and
peak-hour preparation caches when the selected corrections change. Explicit
manual totals and removals take precedence and cannot be replaced by this file.

The November 2026 download contains 698 points, saved as
`counts_20261118.json`, `.csv` and `.gpkg`, alongside layer metadata and source
information. The selection uses `TJOM_ANNEE == 2025` and positive `TJOM`
(weekday daily traffic), not the layer edit date or all-days `TJM`. Only unique
shared `NO_SIREDO`/detector IDs within 75 m and 45 degrees of the supplied
Transcality travel direction are eligible. SITG bearings are converted from
clockwise-from-north to counter-clockwise-from-east. Camera IDs with no exact
shared identifier are left unchanged; matching totals or proximity alone do
not establish identity.

The working definition of a large difference is at least 25% of the original
Transcality total **and** 1,000 vehicles/day. Six additional detectors qualify:
`6027.2` → 9,917, `6091.1` → 6,323, `6091.2` → 7,844, `6092.1` → 4,906,
`6092.2` → 6,075 and `3172.2` → 6,800. Their profiles and ranges use exactly
the same scaling process as the manual corrections. The comparison table,
selected rows and `selection_metadata.json` preserve the decision and thresholds.
These are corrections to weekday profiles, consistent with the supplied manual
`TJOM` corrections; all-days `TJM` is saved but is not substituted automatically.

To download and regenerate the selection from the repository root:

```bash
uv run -m scripts.update_transcality_sitg --counts-directory /path/to/traffic_counts/Transcality
```

Use `--reuse-snapshot --snapshot-date 20261118` to rebuild from the saved
GeoPackage without downloading, or change the two `--minimum-*-difference`
thresholds. Comparisons always use profiles before the additional SITG
corrections, making repeated selection runs stable. Rebuild the counts stages
after updating the selected file.

Daily and peak-hour Transcality maps optionally read
`output_detailed_link_statistics.csv.gz` from the configured count-analysis
simulation directory (`analysis.counts.simulation_path`, or its usual output
directory fallback). Only matched current simulation link IDs are retained.
Multiple detectors matched to one link do not multiply its simulated traffic.

Click a count point to open its full statistics and daily chart in a right-side
panel, replacing the settings panel. Hover shows only station/link IDs, observed
and simulated flow, GEH and differences. The settings panel can be collapsed;
filter parameters expand only when filtering is enabled. Returning from station
details preserves all settings.

Transcality settings include an initially disabled matched-link connector checkbox.
Red connectors run to the nearest point on each matched link's real geometry;
hidden observations do not leave connectors behind. Clicking a point highlights
its marker and complete directed link in orange until returning to settings or
selecting another point. The details panel places the daily chart first and uses
compact horizontal bars for vehicle categories. Short category labels do not
change the underlying counts or their scaling.

The station details show raw simulated vehicle passages for cars, trucks, bikes,
buses, and the file's French/cross-border/Swiss/unknown car categories.
These categories are subsets of cars, not unique people or extra vehicles.
Point colors can also use French cars, French + cross-border cars, Swiss cars,
or foreign-car percentage (`100 × (French + cross-border) / total cars`).
The foreign-car percentage is also shown in station details; it is undefined
when total cars is zero. Total cars includes unknown-origin cars, not buses,
trucks or bikes.

Another available color metric is
`100 × Swiss / (French + cross-border)`. The ratio is included in the details;
it can exceed 100%. A zero denominator or unavailable statistics produces an
undefined ratio (grey color), not zero. Counts use raw simulated passages for
the selected daily or peak period; color limits remain adjustable.
Peak-hour maps sum these details over the configured start-inclusive,
end-exclusive window, while retaining the full-day chart.

The profile overlays MATSim motor traffic in orange on the observed blue curve
and grey Q25–Q75 envelope. Cars and trucks are expanded by
`1 / input_downsampling`; scheduled buses remain unscaled, and bikes are excluded
from motor traffic. This scaling convention is stated in the details panel. The
detailed output profile is separate from the existing comparison flow, which
may average several iterations. GEH, scatter plots and calibration targets are
unchanged. Missing files or unmatched detailed-statistics links leave the
existing observed-only profile intact. Missing hours for represented links are
zero-filled; hours at or beyond 24 are not folded into the daily profile.
