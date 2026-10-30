# TL and TPG comparisons

`analysis.pt.all_comparisons` groups the Lausanne/TL and Geneva/TPG stop and line
reports, plus the car/PT cross-border maps. Run the individual stages when only
one area or transport mode needs inspection. The line reports also write direction
asymmetry maps and optional passenger profiles from simulation legs/trips/persons.
Missing optional demographic fields remain unknown in those profiles.

The default references are TL 2024 and TPG 2025. Existing processed count files
are reused: `analysis.pt.tl_rebuild_counts` and `analysis.pt.tpg_rebuild_counts`
default to `false`. Enable them explicitly to regenerate operator tables from
raw counts. TPG rebuilding uses `analysis.pt.tpg_year` (2024 or 2025).
Without processed 2025 direction counts the aggregate stage omits the TPG line
map with a warning; the stop comparison can still run. Rebuild the TPG counts
once to produce the companion `_by_direction.csv` file.

The comparisons read routes/stops from `data.gtfs.cleaned/output` and supplement
them with simulated stops. Operator, line and direction identity is retained.
Outputs have `tl_` / `tpg_` prefixes and reside in the corresponding simulation
comparison folders. New simulation-table readers accept gzip or Zstandard using
the shared output resolver.

Counts are expanded by the configured population sampling fraction. Passenger
profile sample sizes remain unexpanded. The default `analysis.pt.night_hours: 5`
uses next-day departures for the overnight part of the simulated day; line-flow
profiles reconstructed from hourly boarding/alighting totals are approximations,
especially across hour boundaries. They should not be interpreted as exact
vehicle-level passenger trajectories.

Local validation covers processed-file selection, stop identifiers, direction
availability, sample scaling, compressed input, missing demographics and matching
configuration defaults. Full maps still need the user's simulation and operator
inputs; these files are not committed with the code.
