# Scale airport ODs to the population sample

`data.airports.ods` takes the full-population results of
`data.airports.geneva.ods` and `data.airports.zurich.ods`. It returns one combined pandas
DataFrame with an `airport` column (`GVA`/`ZRH`), keeping only Swiss ground
endpoints. It has no dependency on the generated synthetic population: it uses
the same configured `input_downsampling` fraction as `synthesis.population.sampled`.

```text
geneva.ods + zurich.ods → ods → trips
full-population ODs   sampled ODs   coordinates, modes, departure times
```

## Counts and rounding

For every available annual, period and daily passenger count:

```text
scaled count = full-population floating count * input_downsampling
```

The rate must be finite and in [0, 1]. A rate of 0 produces zero demand; a rate
of 1 preserves the source integer allocations. Other rates scale the original
floating counts, avoiding an extra rounding error from scaling already rounded
municipal counts.

Integer allocation is deterministic and hierarchical within each airport:

1. Sum expected counts by ground direction, observed region and traveller profile
   (Zurich uses `pooled`).
2. Apportion the airport's rounded total between these groups by largest remainder.
3. Apportion each group's integer total between its municipalities using its
   expected municipal proportions.

This preserves the rounded airport total and rounded regional/profile margins.
Annual, period and daily integer counts are rounded separately, as in the source
OD stages; an integer average day need not multiply exactly to the integer year.
No extra directional or round-trip multiplier is applied.

The source OD stages remain unchanged and their caches are read-only. Foreign
counts are excluded rather than redistributed to Switzerland. Scenario-specific
`geneva_airport_demand_factor` / `zurich_airport_demand_factor` apply upstream;
these are separate assumptions, not the simulation sampling rate. Do not set
them to 0.1 merely to select a 10% population sample.

## Contract and exports

The combined table retains source provenance and observed probabilities. It adds:

| Column | Meaning |
| --- | --- |
| `airport` | `GVA` or `ZRH` |
| `od_id` | Airport and original source-row index, assigned before scope filtering |
| `sampling_rate` | Configured `input_downsampling` |
| `sample_weight` | Inverse rate; zero for zero-rate rows, which generate no trips |
| `full_population_<count>` | Floating source count before sampling |
| `full_population_<count>_integer` | Original integer source count, where supplied |
| `<count>` | Scaled expected count |
| `<count>_integer` | Integer number to generate |

`<count>` can be `annual_trips`, `period_trips` or `daily_trips`.
`profile` is `outgoing`/`incoming_other` for Geneva and `pooled` for Zurich.
The stage rejects already scaled inputs. `synthesis.airports.trips` expands
`daily_trips_integer` exactly once and carries `sampling_rate`/`sample_weight`
into its output; it performs no demand scaling.

`result.attrs['summary']` reports full-population and scaled Swiss-endpoint
counts by airport, plus excluded foreign totals. `result.attrs['signature']`
identifies the inputs, sampling fraction and scaling code.

The default folder is `<data_path>/airports/ods/`; override it with
`airport_ods_folder` (relative to `data_path` or absolute):

| File | Content |
| --- | --- |
| `ods.parquet` | Returned combined OD table |
| `ods.csv` | Inspectable copy |
| `summary.json` | Rate, rounding method and full/scaled/excluded counts |
| `signature.json` | Input signature and output checksums |

Synpp tracks both source stages. Export signatures detect missing/modified
files; unchanged inputs load the signed Parquet. Model coefficients are never
re-estimated by this stage.

## Configure and inspect

Inside your existing pipeline configuration:

```yaml
run:
  - synthesis.airports.trips  # automatically runs ods first
config:
  # Retain your other required census/spatial inputs.
  input_downsampling: 0.1
  airport_model_overwrite: false
```

Remove the former `airport_trips_sample_rate` option. The trip stage rejects it
when specified, preventing an accidental second reduction.

With the current sources and `input_downsampling: 0.1`:

| Airport | Full Swiss-endpoint average day | Scaled expectation | Integer movements |
| --- | ---: | ---: | ---: |
| Geneva | 29,975.44 | 2,997.54 | 2,998 |
| Zurich | 55,997.07 | 5,599.71 | 5,600 |

This is **8,598 sampled passenger movements**, including any movements later
reported as unresolved because their municipality has no usable accommodation.
It is a dated validation example, not a hard-coded target. In every airport,
located trips plus unresolved accommodation counts equal the scaled integer
demand. Foreign endpoints are outside that equation.
