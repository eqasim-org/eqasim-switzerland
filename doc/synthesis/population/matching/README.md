# Microcensus diary matching

V1 is unchanged. `synthesis.population.matched` still selects V1 by default;
set `matching_algorithm: v2` to use V2. The working configuration is not changed
by this revision.

V1 uses an ordered, survey-weighted recursive hot deck with age bands and
progressive relaxation of categorical matching cells. V2 has a different goal:
learn which shared characteristics predict behaviour, allowing nearby ages,
different income classes and different household compositions to share donors
when their learned behaviour is similar. V1's bins are therefore **not** copied
into V2.

## Code organization and estimator interface

V2 implementation lives in `synthesis/population/matching/V2/`. The uppercase
`V2` is part of the Python package name. `matched_v2.py` registers the pipeline
dependencies, prepares inputs, calls the estimator, exports diagnostics and
preserves the existing `(mapping, removed_ids)` downstream contract. V1 does
not import this package and remains unchanged.

| Module | Responsibility |
| --- | --- |
| `model.py` | `EmbeddingGRUModel`: person encoder, recurrent decoder and prediction heads; normalized batch encoding |
| `losses.py` | `MatchingLoss`: categorical sequence, masked regression, trip-count, reconstruction, latent-norm/spread losses and optional class weights |
| `training.py` | `EmbeddingTrainer`: weighted training, optimizer/scheduler, household split, early stopping and restoring the best state |
| `features.py` | Shared survey/population harmonization, household counts, feature inventory and fitted `FeatureProcessor` |
| `targets.py` | `SequenceTargetBuilder`: ordered diary targets, missing-data masks and behavioural scales |
| `donors.py` | `DonorSelector`: employment/student eligibility, nearest neighbours, survey weighting and optional transport balancing |
| `wrappers.py` | `DiaryMatcher` public fit/transform/predict API and `MatchingPrediction` result |
| `artifacts.py` | `ModelStore`: compatible model loading, complete training fingerprints and atomic writes |
| `configuration.py` | Pipeline option declarations/validation and `MatchingOptions`; no independent defaults registry |
| `inputs.py` | Read-only preparation of stage inputs and source eligibility |
| `diagnostics.py` | `MatchingDiagnostics`: embeddings, support, donor reuse, inventories and stage report exports |
| `runtime.py`, `provenance.py` | Optional Torch loading/batch conversion and implementation fingerprints |

For already harmonized donor/recipient frames:

```python
from synthesis.population.matching.V2 import DiaryMatcher, MatchingOptions

options = MatchingOptions.from_config({"random_seed": 518745})
matcher = DiaryMatcher(options, cache_directory=local_model_directory)
matcher.fit(donors, donor_trips)
prediction = matcher.predict(population)
mapping = prediction.mapping  # person_id, mz_person_id in recipient order

possible_days = matcher.predict(population, n_draws=20, random_seed=518745)
donor_ids = possible_days.donor_ids  # shape (people, 20); -1 means unmatched
embeddings = matcher.transform(population)
```

`fit` returns the matcher. `predict` uses a fresh explicitly seeded RNG, so
repeated calls with the same people/options/seed are reproducible. It returns
donor IDs, donor row indices, matching diagnostics, pool summaries and unmatched
person IDs. The single-draw `.mapping` property intentionally rejects ensembles
to prevent silently dropping their uncertainty. Preprocessors, fitted model,
donor copies, embeddings and training report are retained as `processor_`,
`model_`, `donors_`, `donor_embeddings_` and `training_report_`. Caller dataframes
and global NumPy/Torch random states are preserved.

The estimator expects V2's harmonized schema: donors use `mz_id` and positive
`person_weight`; recipients use `person_id`. Both include all shared feature
columns and binary employment/student status. In the pipeline, use
`V2.inputs.prepare_inputs(context, constants)` before filtering recipients, so
household counts are recovered on complete households. The stage itself keeps
survey-age eligibility and its existing removal/output conventions. For
pipeline settings already declared by `configure`, use
`MatchingOptions.from_context(context)` instead of `from_config`.

The former `embedding_v2.py`, `features_v2.py` and `donors_v2.py` modules are thin
import forwarders for existing local scripts. New code imports `V2` directly.
Private stage helpers also forward for older callers; analysis uses the public
estimator. There is one model implementation and one matching path.

The reorganization preserves the network, objective weights, targets, scaling,
hard masks and stochastic sampler. A seeded CPU fixture checks exact fitted
tensor equality against the pre-refactor implementation. Model artifact schema
4 fingerprints the separated training modules; older model artifacts cannot be
verified using the new fingerprint and are retrained when a new matching run is
requested. Existing saved evaluation CSV/NumPy evidence remains readable by the
lightweight report stages without loading or retraining those models.

## V2 inputs and masks

Only these two flags define the inference donor pools:

- `employed`: employed people match employed people, and non-workers match
  non-workers. The census and survey employment codes are harmonized first.
- `is_student`: students match students, and non-students match non-students.
  Working students consequently match working students. Use the explicit
  population flag; when no flag exists, the pipeline's `employment_status`
  contract identifies STUDENT=2 and EMPLOYED_STUDENT=3.

No age, income, sex, child-presence, geography, car-access, collective-housing or
work-type matching cells are applied. In particular, there are no age barriers
at 15 or 24, and no preferred exact-age or income cell before nearest-neighbour
retrieval. The model can learn an age-dependent relationship without a bin edge.
Survey-age eligibility still applies; this is not a donor-age constraint.

The model uses these shared inputs, audited against the cached survey and
sampled population and their producer code:

| Group | Inputs | Treatment |
| --- | --- | --- |
| Age and resources | `age`, `income_class` | Continuous age and original 0–8 income codes; training-only standardization, no recoding into broader groups |
| Household composition | `household_size`, `N_children_under_3`, `N_children_under_6`, `N_children_under_12`, `N_children_under_18`, `N_adults` | Numeric counts, not presence flags; no household-size clipping |
| Residential density | `population_density` | `log1p` of residents within 2.5 km of home, computed by the same density pipeline on both sides; it is not residents/km² |
| Residential context | `ovgk`, `municipality_type`, `canton_id`, `sp_region` | Original PT-quality categories A/B/C/D/none and categorical spatial inputs |
| Work context | `employed`, `is_student`, `employment_status`, `work_location_type`, `commute_distance`, `job_position` | Categorical status/location type/job; log commute distance; survey job codes mapped using the repository's common classification; job is unknown for non-workers on both sides |
| Mobility resources | `driving_license`, `car_availability`, `number_of_cars_class` | Harmonized categorical inputs, with no inference mask |
| Personal context | `sex`, `marital_status`, `is_swiss` | Shared categorical definitions; no sex mask |
| Border proximity | `dist_home_to_border` | `log1p` distance, retaining an explicit missing indicator |

The architecture accepts 11 numeric inputs plus missing indicators and 15
categorical inputs. Missing numeric values use training-partition medians;
unseen categories use a reserved unknown index. No imputation/scaling/category
vocabulary is fitted on held-out diary outcomes. Density's predictive value is
measured by held-out grouped permutation rather than assumed from urban/rural
aggregate differences.

### Recovering real household counts

`data.statpop.statpop` originally creates Boolean child-presence fields. Later
model stages also collapse under-18 counts. Merely casting these fields to
integers would give incorrect household composition.

V2 reconstructs each cumulative child count and the number of adults from the
**complete sampled household's ages**, before filtering eligible persons.
Reported household size must equal the number of sampled members before the
reconstruction is accepted. It changes only the matching stage's private copy.
If household membership is incomplete, genuine numeric counts are retained;
ambiguous presence-only positive values become missing, rather than invented
counts of one. Known zero presence remains zero. Outputs flag reconstructed
households and report feature missingness. Running an analysis on a person-level
subsample must therefore prepare features on the full household table first.

For ARE, exact ages and some household composition fields are unavailable.
Interval midpoints are explicitly identified as approximations. Exact-age
validation currently targets STATPOP and rejects ARE rather than pretending its
midpoints are observed ages.

### Available variables deliberately not used

The audit found the following additional fields. Availability alone is not
sufficient for a valid common matching feature.

| Fields | Reason |
| --- | --- |
| `age_class`, `household_size_class` | Coarsened duplicates; V2 uses richer raw information |
| IDs, original STATPOP IDs, household IDs, zone/municipality IDs | Identifiers or high-cardinality locations; used for joins, household reconstruction and independent splits, not as predictors |
| `home_x`, `home_y` | Available but excluded by default to limit location memorization; density, PT quality and categorical geography encode residential context |
| `nationality` | Not identically represented in both files; use the harmonized shared `is_swiss` flag |
| `is_crossing_the_border` | Defines the domestic donor universe; border-crossing travel is modelled separately |
| `N_adults_30_64`, household licence counts, `hh_oldest_age`, `is_head` | Present in the population cache but not in the survey-person table under the same definition |
| `number_of_bikes_class`, bike availability, PT subscriptions, education level, parking, occupation | Survey-only in the inspected STATPOP matching inputs; inventing synthetic values would create spurious similarity |
| Dates, workday/weekend, survey weights | Scenario selection and statistical weighting, not demographic predictors |
| Diary chains, modes, durations, travel distances, being outside Switzerland/car passenger | Training outcomes, donor metadata or later imputed outcomes; not recipient covariates |

The stage exports `feature_inventory.csv` with actual missingness, number of
values and which variables define masks. Check it when changing census source
or upstream models.

## Behaviour learning and donor transfer

The grouped static encoder is supervised through a recurrent diary decoder.
Its objectives include activity purposes, modes, departures, trip distances,
trip counts, travel durations and destination activity durations. Missing times
are masked. Non-travellers have a home/no-trip label, zero totals, and no
fictitious travel-time observations. Log-scaled distances and trip counts are
not clipped at 30 km or five trips. Survey weights enter training and scaling.

An inner household/respondent split selects the best epoch using held-out
behavioural prediction, with early stopping and gradient clipping. The latent
space then retrieves similar donors within the two mandatory status flags.
Sampling uses distance and survey weights with the configured seed. **One
complete observed diary** is transferred; truncating training sequences never
truncates transferred diaries.

`matching_v2_raw_feature_weight` and `matching_v2_balance_strength` both default
to **0**. The first can explicitly add raw-covariate distances; the second can
apply soft donor-marginal regularization on the candidate graph. Both are
optional ablations, not assumptions that national survey totals describe the
target population. `matching_v2_minimum_donors` is obsolete: there are no ordered
soft matching cells to relax.

Matching supplies a respondent ID, not survey coordinates. Later spatial stages
assign destinations, later departure stages can adjust timing, and mode choice
can change modes. Driver feasibility is consequently not another hidden V2
matching mask.

The return contract remains `(mapping, removed_person_ids)`. Underage persons
retain `mz_person_id = -1`. Eligible persons with no donor in their exact
employment/student stratum are removed through the existing downstream
contract; neither mask is relaxed.

## Caches, diagnostics and evaluation

The schema-3 model lives in the V2 stage cache. The older global working-folder
artifact is neither reused nor changed. With
`overwrite_matching_embedding_model: false`, reuse requires a matching hash of
features, weights, trips, training settings, split rows and implementation.
Synpp validation tokens also include imported helpers so changing only a helper
does not silently reuse an old stage result.

Stage outputs include `matching_diagnostics.csv`, `donor_reuse.csv`,
`feature_inventory.csv`, `matching_report.json` and the compatible
`mz_embeddings/mz_embeddings.csv` export. Aggregate activity-chain agreement is
no longer exported as an accuracy test.

The two analysis stages now live at:

- `analysis.matching.compare_matching`
- `analysis.matching.matching_embedding`

They share `analysis.matching.validation`, which evaluates unseen people,
probabilistic diary predictions, population covariate overlap and representation
quality. See the [analysis protocol](../../../analysis/matching/README.md).
