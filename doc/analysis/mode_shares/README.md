# Mode-share analysis and interactive spatial comparisons

`analysis.mode_shares.run` retains the existing static figures and also writes
`mode_shares/spatial_mode_shares.html` beneath the selected simulation output.
It replaces the separate municipality-arrivals analysis.

Both `analysis.mode_shares.simulated` and `analysis.mode_shares.target` return
`spatial_trips`, a dataframe produced by `ModeShareAnalyzer.spatial_trips()`.
This contains the **same filtered domestic-resident trips as the global
mode-share summaries**, with person/trip identifiers, mode, weight, origin,
destination and residential coordinates. Spatial export does not resample trips.
MATSim trips with `airport` as either origin or destination activity are removed
inside the analyzer, before computing any mode shares. Existing mode, distance,
person/subpopulation and destination-purpose filters remain in place.

The HTML supports:

- Drawing and editing a polygon or circle.
- Clicking a circle centre or entering latitude/longitude and a radius in metres.
- Municipality and Canton buttons beside the drawing controls, automatically
  displaying the corresponding clickable boundaries and selection dropdowns.
- Searching municipalities by name/BFS number and selecting any Swiss canton.
- Selecting arrivals, departures, either endpoint, both endpoints, or residents.
- Comparing weighted survey shares with unweighted simulated shares, plus raw
  trip counts and distinct people represented by those trips.
- Downloading the selected statistics as CSV.

“Residents” selects trips whose **home** is inside, including trips outside the
area. This is still a trip-based mode share; non-travellers do not contribute.
“Either endpoint” counts a trip once. Shapes include their boundaries. Survey
shares repeat the respondent's existing `person_weight` for each retained trip;
MATSim trip weights are one. The simulation's sample counts are not expanded to
full-population totals. Municipality and canton selection use original boundary geometry.
Boundary-edge indexing keeps canton selection responsive without simplifying
the query geometry. Circles use great-circle distance; coordinates are stored at integer microdegree
precision (roughly 0.1 m). Missing locations cannot satisfy spatial conditions
but remain in unrestricted totals. Coverage is displayed in the HTML.

The report embeds compressed columnar trip arrays, municipality/canton boundaries, and
pinned Leaflet/Leaflet.draw assets. It opens directly from disk and computes
selections in a Web Worker. A modern browser with `DecompressionStream` is
required. Only swisstopo background tiles require internet; attribution and an
opacity control are included. The first report build downloads UI libraries;
later builds reuse copies in the report's `assets/` directory. Embedded
trip records make full national reports relatively large.

Run only these analysis stages, reusing upstream pipeline caches:

```bash
uv run --no-sync -m synpp config_dib_10.yml --run analysis.mode_shares.target --run analysis.mode_shares.simulated --run analysis.mode_shares.trip_distributions --run analysis.mode_shares.run --debug true
```

All four stages are requested explicitly so debug mode rebuilds the changed
analyzers as well as the final report. This does not rebuild the population or
run MATSim. Simulation selection follows `analysis.counts.simulation_path`, or
`output_path/output_id/simulation_directory` when that override is empty.

## French residents

When `include_external_population` is true (the existing default), `target` and
`simulated` also return compact French mode-share summaries. `run` writes four
figures, two CSV tables and `analysis_sources.json` into
`<simulation directory>/mode_shares/french/`. When false, the French survey
is not loaded and these additional summaries and figures are not generated.
Existing files from earlier runs are left untouched.

The comparison covers all French residents' trips and, separately, trips with
origin **or** destination in Switzerland (including both directions and trips
entirely within Switzerland). Residence is identified by canton FR, not
citizenship. The simulation reuses the main analyzer's trip filters and tables.
The target is the same Annemasse/French Geneva-region survey used by
`data.microcensus.shares`; it does not cover the entire simulated French region.
Survey shares use `trip_weight` and crow-fly `DOIB` distance, with Swiss endpoints
identified from survey zone country labels. Simulation shares use equal trip
weights and Euclidean distance. Both use the same distance bins; empty bins
remain undefined. Survey curves are solid and simulation curves dashed.
