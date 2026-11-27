# Four municipality activity zones

`data.microcensus.activity_chains_clusters` is the single public zoning stage.
It estimates four daily trip-rate fields and returns four whole-municipality
activity groups. Matching uses the `activity_chain_cluster` category at home.

## Inputs and preparation

The stage consumes prepared `data.microcensus.persons`,
`data.microcensus.activity_chains` and `data.microcensus.trips`, plus
`data.statpop.persons`, `data.spatial.municipalities` and
`data.spatial.swiss_border`. It never opens raw Microcensus files.

Survey preparation retains `reported_trip_count` on persons. The activity-chain
stage exposes `valid_diary`: a processed `home` chain with a positive reported
trip count is missing travel information and is excluded from zoning and
matching comparisons. Genuine zero-trip days remain. This audit does not change
the production matchers' existing donor eligibility.

Use the configured diary day and positive survey weights. Count trips arriving
at work (including secondary work), shopping and leisure; total trips include
home returns and all other purposes. Verify these counts against the prepared
chains. Homes must be finite and inside Switzerland.

## Method and defaults

```yaml
activity_zones_neighbours: 400
activity_zones_min_bandwidth_m: 2000.0
activity_zones_min_cell_m: 2000
activity_zones_max_cell_m: 32000
activity_zones_target_cell_population: 15000
activity_zones_geometry_step_m: 1000.0
activity_zones_gaussian_smoothing_m: 0.0
activity_zones_high_population_share: 0.16
```

1. Build population-adaptive 2–32 km cells from permanent main-residence STATPOP
   residents aged 6+. Dense cities receive smaller cells. All eligible residents,
   including repeated home coordinates, retain their full population weight.
2. Estimate work, shopping, leisure and total trips per person with a
   survey-weighted bisquare kernel reaching 400 nearby respondents, with a 2 km
   minimum radius. Include all respondents within that minimum radius. Sparse
   places borrow evidence over a wider physical radius. Export effective local
   sample size and bandwidth for inspection.
3. Interpolate the four fields to the contour geometry grid. No extra Gaussian
   smoothing is applied by default. The 1 km grid controls geometry precision;
   it does not imply 1 km survey evidence.
4. Reserve group 1 for the highest total-trip intensity tail, targeting 16% of
   the eligible resident population. Fit three further population-weighted
   k-means groups from all four standardized rate fields, using 50 seeded starts
   and an explicit convergence check. Extract shared continuous boundaries.
5. Give each entire municipality the contour group containing the largest
   number of its residents. Exact vote ties choose the lower group; shared
   municipal boundary points choose the smaller municipality ID. Empty units
   use their largest intersecting contour area. Check coverage and no split
   municipality. This does not force a multi-municipality agglomeration together.

These fixed settings were selected for regional matching fit. The stage does
not run a bandwidth search, a Potts cell partition or alternative cluster
experiments. No rate is rescaled to obtain four trips per day.

## Code and returned result

The stage defines `configure`, then `execute`, then its stage-specific helpers.
`activity_zones/surfaces.py`, `contours.py` and `municipalities.py` contain the
spatial calculations; these are helper modules, not separate Synpp stages.
Imported helper source participates in stage cache validation.

The result is a dictionary containing `municipalities`, `regions`, `profiles`,
`profiles_comparison`, `city_comparison`, `population_flows`, `metadata`, `cells`
and `contours`. `municipalities` has one row per municipality with its label and
resident-vote audit. `regions` contains the actual dissolved final geometries.
Observed profiles use survey weights; model profiles integrate fields over
actual STATPOP homes. The distinction is preserved in all exports.

`analysis.microcensus.activity_chains_clusters` writes maps, rate fields,
survey-support figures and a standalone `report.html` under
`analysis_path/activity_chains_clusters/`. The main map is
`regions_municipalities.png` (also SVG). Inputs and upstream caches are read-only.

The old alternative stage names, legacy municipal-only algorithm and smoothing
sweep scripts have been retired. Historical generated reports remain available.
See [the smoothing decision](../../analysis/matching/activity_smoothing.md) and
[the paired V1/V2/V3 comparison](../../analysis/matching/activity_chain_zones.md).
