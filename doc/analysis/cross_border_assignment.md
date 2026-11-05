# Early cross-border assignment analysis

Run this stage to inspect assigned crossing demand after crossing selection and
Swiss cross-border modelling, without running the later licence/car models,
population scaling/sampling or whole-population MATSim preparation:

```bash
uv run -m synpp config_dib_10.yml --run analysis.cross_border.assignment
```

`--run` replaces the configuration's requested end stages for this invocation.
The configured cache is reused. There is no dependency on
`matsim.simulation.prepare`, activity-chain matching or secondary-location
assignment. The default reads Swiss residents directly from
`synthesis.population.models.cross_border`, together with full foreign/transit
demand from `data.cross_border.generate_od`. The road network and upstream
cross-border inputs/models must still be current; crossing selection can itself
require travel-time queries. The analysis reads the production count-guided
point overrides when enabled; it does not draw a separate allocation. See
[count-guided assignment](../data/cross_border/README.md#count-guided-physical-crossing-shares).

## Optional actual-sample comparison

The default `cross_border_assignment_include_sample: false` compares full demand
before population scaling and sampling. It does not depend on driving licence,
car ownership, car availability, census scaling, population sampling or the
sampled transit-exit stage. Sample and expanded columns remain blank, not zero.
The map opens on full demand, and official-count ratios use `full_total`.

For the previous report including the actual population sample, set:

```yaml
cross_border_assignment_include_sample: true
```

This additionally requires the licence/car models, scaling and sampling. Full
Swiss counts then come from `data.census.selected` (after configured scaling),
and sampled counts from `synthesis.population.sampled`. Sampled transit exits use
`data.cross_border.through_crossings`, with its own limited route queries.
Official-count ratios in this mode use expanded sampled totals.

The early mode currently supports `census: statpop`; another census requires the
actual-sample mode. Early full counts precede projection-year scaling even if
`enable_scaling: true`. Diagnostics and the report label the population basis so
these counts are not mistaken for a projected simulation population.

## Outputs

The stage returns `<analysis_path>/cross_border_assignment` and writes:

- `index.html`: map, crossing table and survey entry controls. Select an interview
  place or crossing; dim the background with the visibility slider. The map uses
  swisstopo / geo.admin.ch tiles without an API key. Leaflet and tiles require internet;
  the aggregate tables remain usable offline.
- `crossings.csv`: full-population, raw sampled and sample-expanded crossing
  counts, split by direction and foreign visitor / Swiss / transit category;
  source FIDs, aliases, network capacity, importance and selection weights.
- `interview_places.csv`: physical entry/exit totals and the existing survey
  calibration controls, including 2015/2021 entries and target entries.
- `assignment_counts.csv`: aggregate counts by cohort, crossing, category,
  direction and assignment source. No person identifiers are exported.
- `reference_checks.csv`: whether optional observed-count references matched a
  current canonical FID and its coordinate.
- `diagnostics.json`: scope, sampling fraction and unlocated crossings.

When the optional sample comparison is enabled, “Sample expanded to 100%” divides actual sampled events by
`input_downsampling`. It is not a fresh random sample or a reweighting of the
survey. No replacement sample is drawn in early mode.

## Counting rules and limits

- Foreign visitors: one assigned entry plus an assumed return exit at the same
  crossing, consistent with the constructed return tour.
- Swiss cross-border residents: one event in the direction already selected by
  the resident cross-border model. Repeated use of a survey donor does not merge
  distinct population people.
- Transit: entry plus exit. For sampled demand, use the existing through-crossing
  result. An inferred exit is assigned to a reference only if its exact network
  exit link identifies one physical candidate. Full-population exits are located
  only when the reported exit code identifies one physical reference. Remaining
  exits stay explicitly unresolved; they are never added to their entry place.
- Only the pipeline's `car` survey category is counted (which also includes
  motorcycles). Car passengers and public transport are excluded.
- References sharing the same directed road pair within an interview place are
  combined under their canonical point, with source FIDs shown as aliases.

These are **assigned crossings, not routed traffic counts**. They omit ordinary
Swiss/French traffic, freight, later population/trip exclusions and changes made
by route repair during preparation. They cannot detect loops or incidental border
crossings. Use the prepared-plans analysis for those questions.

The survey controls are **entries before exclusions and population selection**,
not total physical traffic in both directions. They also precede geographic
exclusions of external-population regions. Swiss entry/exit demand is selected
separately; transit may leave at a different interview place. Consequently the
controls are context for diagnosis, not an assertion that every two-direction
site total must equal twice an entry control.

For groups calibrated from counts, displayed capacity/importance weights are the
original prior only, not the fitted shares. Otherwise the displayed selection
weight and prior group share precede OD feasibility and
travel-time screening. They are not the final choice probabilities for every
individual.

## Optional official traffic-count references

No new configuration is required. This file also drives production count shares
when `cross_border_count_assignment` is enabled. If present, the stage reads
`traffic_count_references.json` in the selected `cross_border_reference_path`
directory (default: `<data_path>/crossborder`). To use a different file:

```yaml
cross_border_assignment_counts_file: crossborder/my_count_references.json
```

The option defaults to `null`; relative paths resolve under `data_path`.
The JSON input is a list of records, for example:

```json
[
  {
    "fid": 253,
    "name": "Les Verrières",
    "x": 2525107.736543532,
    "y": 1194776.8435780548,
    "low": 4756,
    "period": "2023 annual weekday average",
    "day_type": "weekday",
    "vehicle_scope": "all_motor_vehicles",
    "source_url": "https://www.astra.admin.ch/dam/fr/sd-web/z6BJ81AzKCJQ/jahresergebnisse_2023.xlsx",
    "note": "OFROU station 173, about 740 m inland; not an exact border count."
  }
]
```

Coordinates are the **crossing reference anchor in LV95**, not the traffic
counter. A reference is rejected if its FID is missing/noncanonical or the anchor
has moved over 50 m. This prevents silent attachment to a different crossing.
`high` is optional and defaults to `low`; a range must explain whether it is a
range of periods or another kind of uncertainty. Values must be positive,
full-scale vehicles/day in both directions. Duplicate FIDs are rejected.

Keep this input with the shared data, not generated CSVs in Git. Update it from
published counts, retaining the period, day definition, counter location/offset,
vehicle coverage and source. The initial local Group O references use OFROU 2023,
Doubs road counts and Vaud's September 2024 campaign. They mix years and day
bases, so absolute comparisons are indicative. With count assignment enabled,
the upstream joint stage uses normalized within-group shares; this report does
not perform another calibration or treat their sum as a population target.

Official all-motor-vehicle counts include traffic absent from the early report.
`indicative_ratio_low/high` therefore describe a comparison, not a calibration
error. The report makes coverage and timing differences visible.

## Basemap

Use the swisstopo national map via geo.admin.ch WMTS (Web Mercator / EPSG:3857).
This avoids the direct OpenStreetMap tile service and CARTO's API-key requirement.
The dimming slider remains available. The service is [free without registration](https://www.geo.admin.ch/en/general-terms-of-use-fsdi),
subject to fair use; no online service guarantees permanent availability.
Tiles were checked successfully at Lausanne, Basel and Vallorbe. Retain swisstopo
attribution. Coverage/detail outside Switzerland is limited; marker and table
access remains available if the background cannot load.

## Cache safety

The stage validates its template and optional reference input. It also checks
that the source GeoPackage row order, coordinates and importance agree with the
cached interview-place table before mapping FIDs to generated BCP IDs. A mismatch
raises a clear error instead of mislabelling old assignments. Close GIS editors,
ensure GeoPackage edits are saved/checkpointed and rerun affected upstream stages.
A prior-run preview cannot establish the effects of newly edited crossing points.
