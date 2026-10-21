# Neural secondary-location assignment

The NN assigns secondary activities to the existing synthetic population and
primary locations. It uses `synthesis.population.destinations`, including French
facilities when enabled. Its output remains `(locations, convergence)`, indexed
by the following activity's `trip_index`.

Configure scenarios in `config_dib_10.yml` and run from the repository root:

```yaml
config:
    locations_algorithm: nn
    secondary_nn_use_target_distance: false
```

```sh
uv run -m synpp config_dib_10.yml
```

## Purpose eligibility and spatial masks

Secondary diary labels share the base facility offers: `work_secondary` uses
`offers_work` and `education_secondary` uses `offers_education`. The catalogue
has no general household `offers_home` field, so home visits retain its existing
`offers_home_secondary` eligibility unless an explicit `offers_home` is supplied.
Shop, leisure and other activities use their own offers. Artificial OSM facilities
remain eligible only for leisure and other activities.

At assignment time all four experts receive masks built from the actual facility
pools. Leaf support is propagated through the coordinate tree to middle and root
cells; padded alternatives stay false. The short expert uses the same support.
An all-false mask raises an error rather than creating a uniform distribution.
The final draw stays in the selected leaf and requested purpose; it cannot switch
to an `other` pool or widen to a sibling cell.

## OSM context and additional destinations

`data.osm.secondary_features` reads the configured `data_path/osm/osm_file`
extracts whose filenames contain `switzerland`, before transport-network
filtering. It handles nodes, ways and multipolygon areas, deduplicates overlapping
extracts by OSM identity, and projects to EPSG:2056. The Swiss extract defines the
coverage; no additional country containment checks or boundary clipping are run. Disk-backed temporary node locations
are removed after each input file. Incomplete geometries are counted in logs.
The stage returns a GeoDataFrame and writes `secondary_features.parquet`.

Themes include aerialways, cable-car stations, ski pistes and winter-sport
areas, parks, gardens, viewpoints, named tourist attractions, places of worship,
lake shorelines, public shoreline paths and riverside paths. Waterside access is
inferred from eligible pedestrian OSM paths within `secondary_nn_osm_water_path_buffer_m`
(default 30 m) of water. Only shoreline sections within that distance of a public
path contribute shoreline length; whole lake perimeters are not model features.

Visitor eligibility is decided in `data.osm.secondary_features`. Destination
enrichment and H3 aggregation consume every retained feature without applying a
second `public` filter. The retained `public` flag means eligible for visitor
assignment, including inferred access and ticketed facilities; it does not assert
public ownership or unrestricted access. Original `access`, `foot`, `garden_type`
and the decision's `access_reason` are retained for inspection.

Explicit pedestrian permissions follow `foot` before general `access`. Private,
no-access, members-only, unknown and conditional permissions are excluded.
Closed, disused, abandoned and construction features are excluded. Explicit ski
permissions are also considered for ski infrastructure, so `foot=no` with
`ski=yes` does not remove an operating ski lift. A fee or private
ownership alone does not exclude a visitor destination. `access=customers` is
accepted for ski infrastructure, cable-car stations, named tourist attractions,
places of worship and botanical/arboretum/community gardens, since these are
potential visits to the facility rather than through travel. An explicit
`access=private` still excludes a station or church unless a pedestrian permission
overrides it.

Without access tags, parks, viewpoints, named attractions, places of worship,
passenger aerialways and ski infrastructure are retained by their visitor role.
Freight/industrial aerialways and stations with `aerialway:access=no` are excluded.
Water provides geographic context only: shorelines and riverside features still
require an eligible pedestrian path nearby. Untagged paths, footways, pedestrian
ways and steps are retained; tracks and cycleways need explicit pedestrian access.

Gardens receive stricter checks to avoid residential backyards. Residential/private
garden types (including `garden=residential` and the common `residental` typo),
and gardens tagged as residential buildings or land use, are excluded even if named.
Other gardens are retained with explicit access, botanical/arboretum/community
classification, or a name accompanied by a public operator or tourist-attraction
tag. A named, untyped garden can also be recovered when an eligible path enters
its interior; a path merely along its boundary is insufficient. A name alone
is insufficient. Restricted or uncertain garden polygons are removed from
otherwise eligible overlapping paths, POIs and areas. These are documented
heuristics based on available OSM tags, not guarantees of access at every location.

The rules follow the OSM distinction between [access and ownership](https://wiki.openstreetmap.org/wiki/Access),
[customer visits](https://wiki.openstreetmap.org/wiki/Tag:access%3Dcustomers),
[garden types](https://wiki.openstreetmap.org/wiki/Key:garden:type), and
[passenger versus freight aerialways](https://wiki.openstreetmap.org/wiki/Aerialway).

When the NN algorithm is selected, the destination stage fills gaps in the
catalogue. A proposed point is suppressed within
`secondary_nn_osm_destination_radius_m` (default 50 m) of an existing company
with employment or an activity offer, or of an already accepted proposed point. Stations
and explicit attractions take priority over area interiors. A park/garden point
uses an internal public path where available. A connected waterside path section
gets one midpoint, not one destination per OSM vertex or per H3 cell. The 50 m
rule may deliberately suppress a place beside an existing company even if the
company has a different activity offer.

Added facilities have stable `OSM_...` IDs, `number_employees=1`, and **only**
`offers_leisure` and `offers_other`. All work, education, shopping and secondary
home/work/education offers are false. They receive municipality and PT class
attributes and keep `osm_kind`/`osm_name` provenance. These new records do not
supply primary employment. RDA does not schedule this enrichment.

At each H3 level the model receives aerialway/piste/shoreline/public-water-path
lengths, park/garden/ski areas, and viewpoint/attraction/station/worship counts.
Lengths and areas are measured by intersection with the actual clipped cell
polygons. Polygon attractions count once using a representative point. These
features are included in all four experts, their scalers and saved wrappers.

## Person and origin attributes

All four experts and the selector use driving licence, bike availability,
GA/general subscription, half-fare subscription, regional/route subscription,
and one-hot home ÖV classes A/B/C/D/no class, in addition to the existing age,
sex, employment, car availability, income, chain, time, purpose and progression
inputs. Population assignment reads the actual
`synthesis.population.models.subscriptions` output, rather than silently treating
missing subscription flags in `enriched` as no subscription. Survey subscriptions
are encoded with the same GA-first category priorities. Home accessibility is
recomputed from the actual home point and the common `data.spatial.ovgk` layer;
overlapping polygons use the best class.

The selector also receives origin-to-home/work distances, nearest opportunity
cell distances, and H3 supply summaries within 1.5, 5 and 20 km by default.
Purpose-eligible facility counts and nearest eligible cells come from the same
catalogue and coordinate leaves used for assignment. Quantities are summed and
fractions averaged around a 250 m origin grid. These are spatial supply
approximations, not routed travel-time catchments.

All four experts, with or without target distance, and the distance-free gate receive
remaining distance budget, remaining trip count, budget per remaining trip,
consumed-budget fraction, remaining counts of each purpose, and the distance,
trip offset and type of the next known primary anchor. `chain_state.py` computes
the same vector for survey fitting and synthetic assignment. Only future plan
purposes and primary coordinates are read; future secondary coordinates are not
inputs. The shared daily budget is retained as an existing synthesis input.

## Destination distribution loss

Both target-distance configurations use destination cross-entropy regularized
by exact discrete CRPS for origin/home distances and a spatial energy-score
approximation by default (`secondary_nn_distance_loss_kind: crps_energy`).
The score evaluates the full predicted distribution instead of just its mean.
Distance scores reward the correct radius; spatial scores additionally penalize
wrong directions. Target distance remains an input when enabled, and the 1.5 km
expert-routing threshold and final facility distance reweighting are preserved.
No cross-fitting or mixture gate is scheduled for distance-conditioned models.

Set `secondary_nn_distance_loss_kind: relative` to compare against the previous
relative Huber penalty on expected origin/home distances. This selects the
legacy `secondary_nn_distance_loss_weight` (default 0.07) and
`secondary_nn_distance_loss_short_floor_m` (default 100 m). The new loss uses
`secondary_nn_distribution_loss_weight` (default 0.1). Both options retain the
new remaining-chain features, so this comparison isolates the loss choice.

For the default distribution loss, spatial energy uses fixed angular projections of exact CRPS, multiplied by pi/2: each
projection is a proper score and their angular average approximates Euclidean
energy without quadratic memory in the candidate count. All scores use the
fixed `secondary_nn_distribution_scale_m`, never the observed trip length as a
normalizer. The same objective selects stopping epochs. This removes the earlier
asymmetry between equal absolute over- and under-prediction errors.

For the distance-free configuration, after selecting an epoch count on an
internal person-separated holdout, an expert is reinitialized and fitted for
that many epochs on every supplied development row. The outer validation and excluded cross-fitting fold never enter this refit.

## Cross-fitted probabilistic expert mixture

The historical `short_trip_classifier` stage name is retained, but it now saves
a probabilistic mixture gate rather than a cost-difference regressor.

1. `training_split` reserves 20% of weekday survey people for untouched outer
   validation. The other 80% are development people. Their cross-fitting fold IDs
   are fixed by the configured seed; development also has person-separated gate
   fitting and stopping partitions.
2. `cross_fitted_experts` requests three fold-specific sets of the four expert
   stages, using `secondary_nn_crossfit_fold`. For each fold, experts and their
   scalers exclude that whole fold and all outer people. Predictions for the
   excluded fold form the gate's out-of-fold development data.
3. The normal four expert stages use all development people, including refitting
   after epoch selection. Their predictions on outer validation people provide
   the untouched final evaluation. There is no permanent 30% gate reservation.
4. The gate fits a small regularized neural network. Its normalization is fitted
   only on gate-fitting people, and stopping uses development stopping people.
   It is then reinitialized and refitted on all out-of-fold development rows for
   the selected epoch count. Outer outcomes are used only for final diagnostics.

For each eligible nonstationary trip, both experts provide exact probability mass
for the same observed coordinate leaf and ten reproducible final-facility draws.
Using the common leaf space avoids requiring a survey coordinate to equal a
catalogue company point. Both likelihoods use exactly the masks and local
candidate query used during assignment. A hierarchy likelihood is the product
of its three conditional probabilities. Unsupported observed leaves are counted
in validation coverage; a numerical likelihood floor prevents infinite losses.

The gate predicts `g(x)` and minimizes:

```text
-log(g(x) * P_local(observed_leaf | x)
     + (1-g(x)) * P_hierarchy(observed_leaf | x))
+ crps_weight * CRPS(mixture trip distances, observed distance)
+ energy_weight * Energy(mixture destinations, observed destination)
```

The ensemble scores include within- and between-expert pairwise distances,
using unbiased within-expert pair estimates. They reward calibrated spread as
well as location and distance accuracy. Observed destination/leaf, target distance
and expert outcomes are supervision only, never gate inputs. At assignment time
an expert is drawn with probability `g(x)` using the person's supplied seeded RNG.
The no-distance prediction ignores the supplied target distance apart from the
existing upstream stationary-trip handling. It does not threshold `g(x)` or a
predicted cost difference.

The cross-fitting stage writes `expert_predictions.parquet` and `fold_audit.json`.
The gate writes `expert_comparisons.parquet`, `selector_partitions.parquet`,
`validation.json`, and `validation_by_distance.csv`. Diagnostics report observed
and predicted means/tail shares, local-expert probabilities and unsupported
leaf coverage. A mean-distance ratio outside 0.7–1.3 emits a warning; it does not
silently force predictions to match survey marginals. Person-held-out validation
is distinct from observed-origin report replay and does not establish geographic
or sequential-rollout generalization. Compare these diagnostics and population
behavior after refitting; distribution improvement is not guaranteed by unit tests.

Optional controls and defaults:

| Configuration | Default |
|---|---:|
| `secondary_nn_crossfit_folds` | 3 |
| `secondary_nn_selector_validation_fraction` | 0.2 |
| `secondary_nn_selector_draws` | 10 |
| `secondary_nn_selector_context_radii_m` | [1500.0, 5000.0, 20000.0] |
| `secondary_nn_distance_loss_kind` | `crps_energy` |
| `secondary_nn_distribution_loss_weight` | 0.1 |
| `secondary_nn_distribution_scale_m` | 1000.0 |
| `secondary_nn_energy_projections` | 8 |
| `secondary_nn_selector_crps_weight` | 0.1 |
| `secondary_nn_selector_energy_weight` | 0.1 |
| `secondary_nn_selector_max_iter` | 300 |
| `secondary_nn_selector_learning_rate` | 0.01 |
| `secondary_nn_selector_weight_decay` | 0.001 |
| `secondary_nn_selector_patience` | 20 |
| `secondary_nn_short_query_radius_m` | 200.0 |
| `secondary_nn_short_min_candidates` | 6 |

`secondary_nn_crossfit_fold=-1` selects production expert fitting; nonnegative
values are internal fold stages. Leave it at its default in run configurations.
The old `selector_person_fraction`, `selector_spatial_loss_weight`,
`selector_loss_scale_m`, `selector_min_samples_leaf` and `selector_l2` controls
are obsolete. Three folds require four expert sets in total, plus the selected-
epoch refits; fold checkpoints and comparisons are cached for reuse. Fold-only
models skip country-wide diagnostic maps but retain fitting histories.

The local training cutoff remains 1500 m. When target distance is disabled,
final facility sampling omits target-distance reweighting. Observed/matched
distances still supply training supervision, identify stationary trips below
10 m, and contribute existing day-budget summaries. The switch does not remove
every survey distance from the pipeline.

## Coordinate hierarchy and compatibility

Coordinates are indexed independently at H3 resolutions 5, 7 and 10. Middle and
leaf IDs encode the coordinate path. A fine H3 cell crossing a coarse boundary
has a separate node in each occupied branch. Node polygons are intersected in
EPSG:2056 with every ancestor. Training labels, feature aggregation, destination
pools, sibling fallbacks and short-range lookup all use this shared tree.

**Retrain all four experts in both configurations after this update, and the
mixture gate for the distance-free configuration.** Both expert schemas now
include remaining-chain features; loading older expert checkpoints raises a
retraining message. The default `overwrite_*_model_if_exists` options are true.
Separate target-distance configurations still use different expert cohorts, so
comparisons between them are fitted-pipeline comparisons rather than pure feature
ablations. Better held-out scores and population behavior must be checked after
retraining; the new loss alone does not guarantee empirical improvement.

Run focused checks, including calibration and a small real stage chain:

```sh
uv run -m pytest tests/test_secondary_nn_distribution.py tests/test_secondary_nn_enrichment.py -q
```
