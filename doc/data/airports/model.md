# Municipal allocation of fixed airport passenger counts

See the [airport workflow](README.md) for the stage graph, source inventory,
trip generation and source-refresh procedure.

`data.airports.model` returns an `AirportModel`, calibrated on Geneva evidence.
`data.airports.geneva` and `data.airports.zurich` depend on that stage and use it
to disaggregate their **existing Swiss passenger totals**. The model supplies
spatial probabilities, not additional airport demand. Every original regional,
profile and directional total is preserved, including its integer allocation.

## Reused inputs

- `data.spatial.municipalities`: current Swiss municipal polygons and historical
  municipality ID mapping, returned as `(municipalities, mapping)`.
- `data.statpop.municipality_counts`: a new lightweight aggregation of
  `data.statpop.persons`. Counts permanent main-residence records using reporting
  municipality IDs, remapped to current municipalities. It does not use sampled
  synthetic persons, household-size restrictions or recovered-household filters.
- `data.statent.statent`: already located establishments with municipality IDs and
  detailed local NOGA codes. Hotels are distinct establishments with NOGA `551*`.
  The combined `559*` category cannot distinguish student dormitories reliably,
  so **all dormitories are excluded**, as agreed. No employment or capacity
  weighting is applied to hotels.
- `data.spatial.cantons`: canton assignment for municipal representative points.

For the current data, these yield **8,962,258 residents**, **3,722 hotels** and
**2,136 municipalities**. Population is from the pipeline's 2023 STATPOP source;
STATENT establishments are from 2022. Twenty-six water/uninhabited polygons with
neither residents nor hotels are excluded from the probability support.

Coordinates use EPSG:2056. A municipality's mean valid STATPOP home coordinate
is used when it lies in the polygon; otherwise an interior representative point
is used. This is a municipal approximation to Euclidean access, not routed
travel time. Airport reference coordinates are configurable approximate points:
GVA 46.238333 N / 6.109444 E, ZRH 47.458056 N / 8.548056 E. They represent the
airport, not a selected terminal door.

## Gravity mixture

For municipality `i` and airport `a`, with distance `d` in kilometres:

```text
population_weight[i] = population[i] / (distance[i,a] + d0)^beta_population
hotel_weight[i]      = hotels[i]     / (distance[i,a] + d0)^beta_hotel
```

Normalize each component over Swiss municipalities. For residents travelling
away, use the population component. For visitors/circuit travellers:

```text
visitor_probability = (1 - hotel_mixture) * population_probability
                      + hotel_mixture * hotel_probability
```

Population and hotel count exponents are fixed at one. The four fitted
parameters are the positive distance offset `d0`, two distance-decay exponents,
and the visitor hotel-mixture coefficient. The hotel component does not assert
that a traveller stays at a particular hotel. The population component also
allows visitors staying with friends or family.

Zurich lacks an observed resident/visitor breakdown, so its Swiss probabilities
mix the two Geneva profiles using Geneva's observed Swiss profile proportions.
This mixture and the coefficients are transferred assumptions, not measured
Zurich municipality shares. There is no airport-competition term in this first
model; distance is the agreed simple proxy. A small long-distance tail remains.

## Historical calibration, then latest update

The archived source is the [OFAC/Intraplan report, December 2014](https://www.bazl.admin.ch/fr/geneve-protocole-de-coordination),
Annex 3. The PDF is saved at
`airports/geneva/data_from_websites/intraplan_2014.pdf`. Physical page 30,
printed page 24, figure 3-12 identifies these 2013 flows, in thousands:

| Canton | Passenger movements, thousands |
| --- | ---: |
| Genève | 4364 |
| Vaud | 3305 |
| Valais | 538 |
| Fribourg | 305 |
| Neuchâtel | 253 |
| Bern | 873 |

The reviewed observations and source metadata are
`airports/geneva/historical_swiss_2013.csv` and its sibling JSON. Other map labels
are not used where their geographic association was less certain. The old
objective compares **conditional shares among these six cantons**, so omitted
cantons are not treated as zero. These are historical regional reconstructions
from the report, not a new survey of individual municipal origins.

1. Estimate all four parameters against the six historical canton shares, using
   several deterministic starting points and weak regularization. Historical
   masses are approximated using the available contemporary census/hotel data,
   and the historical Swiss resident/visitor mixture is assumed to equal the
   current one. These approximations are recorded in the calibration report.
2. Hold the historical distance offset fixed and update **three parameters**:
   population decay, hotel decay and visitor hotel mixture. Targets are the 2025
   survey's separate resident and visitor shares for Genève, Vaud, Valais and
   other Swiss cantons, normalized within Switzerland. They provide six
   independent shares across the two profiles. Weak historical anchors stabilize
   the fit; rounded data need not be fitted exactly by the unconstrained model.
3. During Geneva allocation, normalize model weights **inside each observed
   region/profile row**. This enforces the latest observed group totals exactly,
   even where the fitted unconstrained model has residual error. For Zurich,
   normalize over all Swiss municipalities within its existing Swiss total.

The raw and fitted shares, objectives, coefficients, parameters at bounds and
limitations are exported to `airports/models/calibration.json`. The model is an
approximate allocation mechanism. Four fitted coefficients with six historical
regions do not establish strong statistical identification or causal effects.
The historic fit must not be described as reconstructing a 2013 municipality
population or accommodation inventory.

## Returned model and caching

```python
allocation_model = context.stage("data.airports.model")
allocation_model.parameters
allocation_model.municipalities
allocation_model.probabilities("ZRH")  # normalized Swiss municipal probabilities
allocation_model.probabilities("GVA", profile="outgoing")
```

Municipalities and probability arrays use the same sorted row order. The stage
returns the fitted object, not a filename. The airport stages call its
`allocate(regional_trips, airport)` method to produce municipality rows.

The default model folder is `<data_path>/airports/models/`:

| File | Content |
| --- | --- |
| `parameters.json` | Reusable coefficients, Swiss visitor fraction and calibration provenance; authoritative parameter store |
| `model.pkl` | Current model object, also cached by Synpp; rebuilt from saved parameters and current municipal inputs |
| `signature.json` | Content signature and model artifact digest |
| `municipalities.parquet` | Municipal population, hotel counts, canton and representative coordinates |
| `calibration.json` | Parameters, historic/latest comparison and provenance |

By default the stage loads `parameters.json` and returns a model using those
coefficients with the current municipal inputs. It does not read calibration
PDFs or re-estimate coefficients when source data change. Missing or invalid
parameters produce an error asking for an explicit overwrite; there is no
automatic first fit.

Set `airport_model_overwrite: true` to estimate and save replacement parameters.
This also bypasses Synpp's model-stage cache on every run, so set it back to
`false` after calibration. Estimation records the calibration input signature,
comparisons and provenance. Normal loading leaves `parameters.json` and
`calibration.json` unchanged.

The returned model's `diagnostics.current_input_summary` describes the current
municipal features. The saved report's `diagnostics.input_summary` describes
the inputs used for the last calibration; loading fixed parameters does not
replace that training history.

Allocation signatures cover the saved parameter file, current municipal inputs,
airport coordinates and model code. Synpp tracks census/spatial parents as
dependencies. Updated census inputs or edited saved parameters update airport
allocations without re-estimation. Airport output signatures include this model
signature as well.

For example, under the pipeline's `config` mapping:

```yaml
airport_model_folder: airports/models
airport_model_overwrite: false  # true only when explicitly recalibrating
```

Configuration options declared by the model stage:

| Option | Default / meaning |
| --- | --- |
| `airport_model_folder` | `airports/models`, relative to `data_path` or absolute |
| `airport_model_overwrite` | `false`; load saved parameters. Set `true` to estimate and replace them on every run |
| `airport_model_historical_csv` | `airports/geneva/historical_swiss_2013.csv`; requires sibling JSON with PDF path/checksum |
| `airport_model_airports` | Configurable GVA/ZRH longitude/latitude mapping |
| `airport_model_distance_offset_km` | 50 km initial/regularization anchor, used only when overwriting; the historical fit estimates the actual offset |

The latest survey uses the same `geneva_airport_pdf` and
`geneva_airport_survey_csv` options as the Geneva stage, avoiding an independent
copy of the current observations. Model calibration never depends on the
Geneva OD stage's result, so the dependency graph has no cycle.

## Airport outputs and unresolved foreign demand

The airport stages now return and cache municipality-level Swiss rows. Fields
include `municipality_id`, `municipality_name`, `canton_id`, `distance_km`,
`municipality_probability`, `origin_municipality_id`,
`destination_municipality_id` and `allocation_model_signature`.
`municipality_probability` is conditional on the original regional/profile row;
original `region_id` and its survey probabilities remain provenance fields.
Airport endpoints remain `GVA`/`ZRH`; Swiss geographic endpoint strings use
`CH-MUN-<BFS ID>`.

The main Parquet/CSV and `.municipality_od.csv` contain these rows.
`.regional_od.csv` remains an aggregated audit of the original regional totals.
For Geneva, the municipality CSV sums over resident/visitor profiles; the main
Parquet retains them. Largest-remainder integer allocation runs within each
original row, preserving the original integer margins as well as expected
counts. The existing timing outputs and sampling API remain available.

**Foreign groups are retained rather than allocated to Swiss municipalities.**
Their municipality ID is null and `spatial_resolution` is
`unresolved_foreign_region`; Swiss rows have `spatial_resolution=municipality`.
French and other-country municipal census/hotel inputs are not supplied by these
Swiss stages. Keeping the unresolved rows preserves every airport passenger
without inventing foreign municipal origins or dropping their demand.

## Validation

Focused tests cover source-population filtering/remapping, hotel-only counts,
distance response, normalization, historical fit/latest partial update, exact
regional/profile/directional and integer conservation, foreign residuals,
parameter-dependent allocation and cache invalidation. The real-data execution
can reuse existing read-only upstream caches; no full synthetic-population or
MATSim build is needed to check these airport stages. A Synpp dry run verifies
the combined dependency graph separately.
