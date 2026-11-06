# RDA opening-hours comparison

`analysis.rda.run` requests two executions of
`synthesis.population.spatial.secondary.locations_rda`, overriding only
`consider_opening_hours` (`false` and `true`). Both use identical trip times,
population, primary anchors, calibration distributions and initial worker
seeds. As retries consume random numbers, differences can propagate to later
assignments; this is a paired run comparison, not an isolated causal estimate.

Outputs are under `analysis_path/rda`:

- `incoming_distance_histograms.png` and `incoming_distance_ecdf.png`: incoming
  Euclidean distances for shopping/leisure, nationally and for residents around
  Lausanne, Geneva and Zurich. Plots show 0–50km; summary statistics use all
  finite distances, including the tail.
- `destination_relocations.png`: paired old/new destination displacement.
- `city_destination_changes.png`: changes in destination activity counts in
  500m cells around the three cities, including visitors from other areas.
- `summary.csv`: counts, mean/median/p90 incoming distances, missing distances,
  changed-destination shares, relocation distances, known-hours coverage and
  compliance for the complete stay among known schedules.
- `convergence.csv`: valid-problem shares and numbers of assigned activities.
- `paired_changes.parquet`: person/activity keys, both destination IDs and
  coordinates, incoming distances and paired changes.
- `metadata.json`: spatial scope, definitions and interpretation notes.

City samples are home locations within `analysis.rda.city_radius_m` of each
city centre (default 15,000m). These circles are not administrative boundaries.
Incoming distances account for a changed previous secondary destination too.
Unknown hours are unrestricted by RDA and are reported separately from known
schedule compliance. Trip purposes and times are identical between variants.

## Focused Slurm run on existing population inputs

```bash
sbatch run_rda_comparison.sh config_dib_10.yml
```

The script requests one task, 24 CPUs, four hours and 8GB per CPU. RDA uses
24 worker processes; BLAS/OpenMP threads are limited to one per worker to avoid
oversubscription. The existing proxy/Maven initialization is retained.

`analysis.rda.run_cached` creates a unique `rda_comparison_<timestamp>` directory
under the configured working directory. Input pickles and cache directories
are linked read-only by convention; upstream files are never changed. The
newest complete cache for each required input is selected, with exact paths
recorded in `input_manifest.json`. This explicit cache reuse does not validate
upstream freshness against changed code. In particular, it reuses the existing
business extraction; use an extraction covering the simulation region.

Synpp debug mode executes exactly five stages in this isolated workspace:
fresh trip times, fresh inferred destination hours, both RDA variants, then the
comparison. Population synthesis, primary assignment and detour-matrix
construction are not rerun. Outputs of the RDA stages remain separately cached
and results can be inspected even if later plotting fails. The source
configuration is read in memory without copying credentials into reports.

A normal Synpp run with `analysis.rda.run` uses normal dependency validation
instead. Concurrent comparisons should use distinct `analysis_path` values
to avoid overwriting exported reports.
