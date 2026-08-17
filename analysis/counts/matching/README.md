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
