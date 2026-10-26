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

All other current canton datasets contain bidirectional totals. Their callers
therefore pass `mode="bidirectional"` explicitly. Point stations are matched to
the closest valid opposite pair. Line stations use the closest parallel link in
each direction. A missing opposite link means that the station is left
unmatched; a one-link fallback would undercount the simulated bidirectional
flow.

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

The station details show raw simulated vehicle passages for cars, trucks, bikes,
buses, and the file's French/cross-border/Swiss/unknown car categories.
These categories are subsets of cars, not unique people or extra vehicles.
Point colors can also use French cars, French + cross-border cars, Swiss cars,
or `100 × Swiss / (French + cross-border)`. The ratio is included in the details;
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
