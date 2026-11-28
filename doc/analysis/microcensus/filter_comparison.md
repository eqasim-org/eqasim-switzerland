# Microcensus filtering comparison

Run from the repository root:

```bash
uv run -m synpp config_dib_10.yml --run analysis.microcensus.filter_comparison --debug true
```

Debug mode reuses the existing `data.microcensus.trips` and
`data.microcensus.persons` caches without validation. This is useful for inspecting
exactly the processed data currently used in other analysis; it does not rebuild
the survey preparation. Omit debug mode if the processed stages need validation.

Outputs go to `<output_path>/microcensus_filter_comparison/`: a standalone
`report.html`, separate PNG charts, and CSV tables with summaries, distributions,
modal distances and diary-retention diagnostics. Raw files are read from
`<data_path>/microcensus/`; source files and upstream stage results are not modified.

The report compares original/processed all-day and weekday cohorts, for national,
Geneva and Zurich canton, and Geneva and Zurich municipality residence areas.
Weekdays follow `tag` 1–5, matching the pipeline. Municipalities use the original
survey residence BFS codes, not modern boundary overlays.

Person-level measures use WP over all selected respondents, including those
without trip records. Trip-level measures repeat the person's WP for every trip.
No extra domestic, airport, age or matching-eligibility restrictions are imposed.
The processed denominator is the person-stage roster, not only people who still
have trips. The diagnostics explicitly identify retained people whose original
trip records disappear. Original recorded trip counts are used consistently;
reported T_WEGE discrepancies are counted separately, without inventing trips.

Crow-fly distances are calculated from raw LV03 coordinates transformed to LV95,
and compared with processed `crowfly_distance`. Distances are not truncated;
foreign travel can therefore substantially influence averages. Missing coordinate
codes are excluded from raw distance statistics, with counts reported. Modal
labels follow the trip-stage mapping and passenger-stage override; `_loop` is
removed in processed modal summaries. This means PT includes taxi, plane and
coach, consistently with that stage's existing convention.
