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

## Days of the counts

The simulation is an average working day. Every reference therefore averages all
the days from Monday to Friday of its year: Fridays, school holidays and the
public holidays that fall on a weekday included. Saturdays and Sundays are left
out. TL keeps its dates by weekday, whatever their timetable type; TPG keeps the
weekday school and public holidays (`WORKDAY_TYPES`); Lemanis 2024 averages all
months and weekdays, weighted by their number of days in 2024; Lemanis 2022 uses
its Monday–Friday table and SBB its working-day average (DWV/TMJO). The processed
TL and TPG tables must be rebuilt once (`analysis.pt.tl_rebuild_counts` /
`analysis.pt.tpg_rebuild_counts`) for these days to apply.

## Stop identifiers

All the comparisons identify a stop by its station code, `gtfs_utils.stop_code`:
the DiDok number of a Swiss SLOID (`ch:1:sloid:1120:0:3` → `8501120`) or of an
older numeric id, the UIC code of an SNCF id, else the id itself. The operator
tables keep these codes, so they stay valid when the GTFS changes its ids.

## MBC counts (Morges region)

`analysis.pt.TL_comparison.mbc_2024` adds the MBC lines (bus 701 to 736 and the
BAM train, `R56` in the GTFS) to the Lausanne comparisons. With
`analysis.pt.include_mbc: true` (default) both TL stages add them to the TL
counts, and their reference is called "TL + MBC". The raw files are the 2024
line diagrams in `analysis.pt.mbc_raw_counts_path`
(`Lausanne_passenger_counts/MBC/H24`); the 2025 ones (`H25`) are not used, so
that both operators count the same timetable year.

The diagrams give, per scheduled trip, the boardings and alightings at each stop
averaged over the year, not daily counts. The stage therefore:

- weights the weekday day types by their number of days (school term 190, school
  holidays 64, and the 10 public holidays on a weekday, which run the Sunday
  timetable; the BAM splits school weekdays into Monday–Thursday and Friday).
  A line without a sheet for a day type (706) has no trip on those days;
- puts each trip in the hour of its passage at each stop, from its departure
  time and the GTFS travel times of the line;
- estimates the day-to-day standard deviation from the TL counts
  (`var = a × mean^b`, fitted on the TL stops/hours), for the confidence
  intervals of the stop map. Minimum, maximum and quantiles stay empty.

Stops are matched to GTFS stations by the DiDok numbers of the `Feuil1` sheets,
otherwise by name (the diagram names have lost their accents). The stage writes
`mbc2024_agg_workday.csv` (TL format), the line directions and the stop
crosswalk in its cache. The stop map only shows the stops inside
`analysis.pt.tl_perimeter`; the line map shows the whole lines.

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
