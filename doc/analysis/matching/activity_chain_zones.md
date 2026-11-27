# Activity zones in V1, V2 and V3 matching

The default `matching_spatial_feature: activity_chain_cluster` uses the
[four whole-municipality zones](../../data/microcensus/activity_chains_clusters.md)
from `data.microcensus.activity_chains_clusters`.

`workflow.preprocess` attaches the same home zone to prepared survey donors and
synthetic recipients through `spatial_feature.add_regions`. It preserves IDs,
row order and upstream frames. Homes on a municipal boundary choose the smaller
municipality ID. Finite homes outside coverage use the existing deterministic
nearest-region fallback, which is logged.

- V1 and V3 replace `population_density_class` at the same position in their
  preference list. Importance and relaxation rules are unchanged; zones are
  not hard donor masks.
- V2 adds the zone to its categorical geography inputs. It retains continuous
  population density and any other existing density inputs. The zone vocabulary is
  fitted only on training donors, with an unknown-category index for unseen
  values. Training artifacts fingerprint the actual schema and zone values.
- `matching_spatial_feature: density` remains available as a control. It omits
  the zone input entirely for V2 and restores the density preference for V1/V3.

Age eligibility, explicit home-only bypasses, collective donor restrictions,
opening-hours masks, other features and output `(mapping, removed_ids)` contracts
are unchanged. The current zones require STATPOP; an unsupported census is
rejected explicitly.

## Running and resuming the comparison

The stage `analysis.matching.activity_chain_zones_experiment` runs six production
variants: density-only and zone-enabled V1, V2 and V3. Each V2 variant fits its
own model with identical training options, household split rules and seed.
There is no global feature-schema mutation or reuse of a model trained on the
other variant's inputs. Model and mapping checkpoints are local to each variant.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run -m analysis.matching.run_activity_chain_zones \
  --config config_dib_10.yml
```

For a focused comparison on an already built population:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run -m analysis.matching.run_activity_chain_zones \
  --config config_dib_10.yml --cached-inputs
```

This mode reads trusted unique local upstream caches and refuses ambiguous
selections. It rebuilds persons through the real preparation stage to expose
the diary audit, verifies every pre-existing person column is unchanged, then
rebuilds chains and the current zones in memory. Existing population, anchors,
trip and geography caches are read-only. Results are exported to the configured
analysis folder; no MATSim run is launched. V3 threads stay within the current
CPU allocation; small numerical batches can be limited through environment
variables as above.

The output directory is `analysis_path/matching/activity_chain_zones_<identity>/`.
Its manifest includes source/population hashes, zoning metadata, implementation
identity and V2 settings. Completed mappings are reused only under that identity.
The zoning report is `analysis_path/activity_chains_clusters/report.html`.

## Cohorts and scores

All six variants use exactly the same analysis recipients. Exclude explicit
home-only bypasses, collective recipients restricted to home-only donors,
children below the survey age, optional forced-home truck-driver flags, and
any recipient lacking a valid processed diary in any variant. Genuine survey
nontravellers remain. Exclusion flags can overlap; the CSV records each count.

Compare residence-based national rates, all 26 cantons, and Lausanne, Geneva
and Zurich municipalities for work, shopping, leisure and total trips. The MZ
reference uses positive survey person weights. Regional expected counts equal
the weighted regional MZ mean multiplied by the common recipient population;
full-scale equivalents divide by the input sampling fraction. These are
cohort-standardized expectations, not official observed canton totals.

`regional_trip_counts.csv` contains rates, counts, relative errors and effective
survey support; `improvements.csv` compares each matcher to its density control.
`error_summary.csv` gives equal-region and population-weighted errors by purpose.
`balanced_scores.csv` averages normalized rate errors equally over the four
purposes and canton/city levels within each matcher, with equal-region weighting
inside a level; it also records the paired percentage reduction.
`coverage.csv`, `exclusions.csv`, `protocol.json`, model training reports and
mapping checkpoints support auditing. The HTML embeds PNG figures; PNG/SVG
files are also saved separately. Render saved tables without matching via:

```python
from analysis.matching.activity_chain_zones_report import render
render("/path/to/activity_chain_zones_<identity>")
```

The zones and matching reference share the fitting Microcensus survey. These
comparisons assess descriptive regional diary fit from one paired seed, not
independent predictive accuracy or MATSim performance. Inspect city and purpose
tradeoffs and low-support flags alongside any aggregate score.

## Completed default comparison

The recorded run also enabled the separate STATENT company-density extension.
The activity-zone change does not introduce that extension; reproducing the exact
V2 numbers below requires it. The runner preserves whichever base density inputs
are available.

The cleaned-code run is saved at
`analysis_path/matching/activity_chain_zones_25b978dc4c87/`. All six variants
retain all 910,885 input residents. The shared comparison cohort contains
838,759 non-forced-home recipients, with a weighted reference of 37,780 valid
MZ respondents. This six-variant intersection differs from the earlier
four-variant and smoothing-sweep cohorts. V1/V3 donor mappings and every
municipal label/geometry exactly reproduce the previously selected setting;
small changes in their published aggregate errors reflect the common cohort.

| Matcher | Balanced error, density → zones | Reduction | Cantons with better total trips |
| --- | --- | --- | --- |
| V1 | 0.07073 → 0.04625 | 34.6% | 20/26 |
| V2 | 0.06286 → 0.06123 | 2.6% | 13/26 |
| V3 | 0.07330 → 0.05266 | 28.2% | 21/26 |

Equal-canton mean absolute total-trip rate errors change from 0.13508 to 0.06949
for V1, 0.13315 to 0.12791 for V2, and 0.13358 to 0.07994 for V3. Improvements
are not uniform across purposes or cantons. V2's aggregate improvement is modest.

| Municipality | V1 total-trip error | V2 total-trip error | V3 total-trip error |
| --- | --- | --- | --- |
| Lausanne | −5.34% → −0.24% | −2.60% → −1.30% | −4.74% → −0.21% |
| Geneva | −4.60% → +1.13% | −2.13% → +0.03% | −3.93% → −1.61% |
| Zurich | +3.88% → +2.56% | +4.92% → +3.77% | +2.92% → +1.75% |

The V2 density-only fit stopped after 29 epochs (best epoch 17), and the zoned
fit after 47 (best epoch 35), both at the configured validation plateau. Their
training/validation membership, numerical imputations/scaling and base
categorical vocabularies are identical. Only the second fit includes the four
activity-zone categories. The benchmark does not establish independent
predictive improvement: zone learning and regional references use the same
survey, and only one paired seed is evaluated.

`independent_audit.json` records the independently reconstructed common cohort,
reference rates, scaling checks and unchanged V1/V3 mappings. Key figures are
`balanced_error_summary.png`, `city_total_trip_errors.png`,
`canton_total_trip_errors.png`, and the default zoning map at
`analysis_path/activity_chains_clusters/regions_municipalities.png`.
