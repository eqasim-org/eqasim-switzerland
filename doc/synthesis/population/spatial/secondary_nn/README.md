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

The hierarchy and distance-free experts use destination cross-entropy regularized
by exact discrete CRPS for origin/home distances and a spatial energy-score
approximation by default (`secondary_nn_distance_loss_kind: crps_energy`). The
distance-enabled short expert overrides this with asymmetric relative-distance
scoring, described below.
The score evaluates the full predicted distribution instead of just its mean.
Distance scores reward the correct radius; spatial scores additionally penalize
wrong directions. Target distance remains an input when enabled, and the 1.5 km
expert-routing threshold and final facility distance reweighting are preserved.
No cross-fitting or mixture gate is scheduled for distance-conditioned models.
All destination experts use continuous training with rotating warm-up folds and
a reserved person holdout. This is distinct from the distance-free gate's expert
cross-fitting stages.

Set `secondary_nn_distance_loss_kind: relative` to compare against the previous
relative Huber penalty on expected origin/home distances. This selects the
legacy `secondary_nn_distance_loss_weight` (default 0.07) and
`secondary_nn_distance_loss_short_floor_m` (default 100 m). The new loss uses
`secondary_nn_distribution_loss_weight` (default 0.1) for the hierarchy and
distance-free experts. The distance-enabled short expert has its own
`secondary_nn_short_distribution_loss_weight` (default 0.05). Set this option
to null to inherit the shared coefficient. The override is ignored without
target distance. Both options retain the
new remaining-chain features, so this comparison isolates the loss choice.
`secondary_nn_short_distance_loss_kind: relative` switches only the
distance-enabled short expert to relative scoring, leaving the hierarchy's loss
choice unchanged. This is now its default: `secondary_nn_distance_loss_weight`
(0.07) multiplies both origin and home penalties, and the origin underestimation
multiplier is 2. Set `secondary_nn_short_relative_underestimation_factor: 1` to
recover the symmetric legacy objective. There is no CRPS, energy or auxiliary
term in relative mode.

The completed 0.1/0.2/0.3/0.5 distance-enabled comparison selected **0.1** by
matched trip-distance errors; it remains the shared distribution-loss default.
See the [weight findings](<../../../../analysis/secondary locations/loss_weight_findings.md>).

In optional `secondary_nn_short_distance_loss_kind: crps_energy` mode, the
distance-enabled **short-range expert only** additionally uses
`secondary_nn_short_relative_distance_loss_weight` (default 0.05) times:

```text
Huber((sum_j p_j * distance_from_origin_j - observed_trip_distance)
      / (observed_trip_distance + 100 m))
```

This restores a smaller version of the legacy expected-distance penalty,
targeting the supplied trip radius. The home-distance penalty is not duplicated.
The auxiliary term is added independently of the short distribution coefficient,
and the same combined objective is used for fitting, validation/stopping and
any full-data refit. Its coefficient is a conservative starting value, not a
winner established by the completed sweep. Set the auxiliary weight to zero and
the short distribution override to null to recover the sweep's shared-coefficient
pure distribution objective. It is disabled when target distance is omitted or
when `secondary_nn_distance_loss_kind: relative` already selects the legacy loss.
Retrain the short-range expert and rerun assignment to evaluate the addition.
New short-range checkpoints are saved under their stage cache path, so trials
with different loss configurations do not overwrite the previous shared model.

For the default distribution loss, spatial energy uses fixed angular projections
of exact CRPS, multiplied by pi/2: each
projection is a proper score and their angular average approximates Euclidean
energy without quadratic memory in the candidate count. Distribution scores use the
fixed `secondary_nn_distribution_scale_m`, never the observed trip length as a
normalizer. The same objective selects stopping epochs. This removes the earlier
asymmetry between equal absolute over- and under-prediction errors.

## Shared validation and minimum exploration

All four experts and the learned selector default to
`secondary_nn_model_validation_strategy: rotating_holdout`. Each expert version
trains one network with one optimizer, without a restart or final refit:

1. Reserve 10% of people before the first gradient update. All trips of a person
   stay together. Reserved people never contribute training gradients.
2. Divide the remaining 90% into five fixed person folds. During the first
   `secondary_nn_model_min_epochs: 30` epochs, use four for training and one for
   diagnostic validation. Randomly select a different validation fold every
   `secondary_nn_model_validation_rotation_epochs: 3` epochs. These are ten
   three-epoch blocks; a previous validation fold can subsequently train.
3. From epoch 31, train on the entire development 90% and validate only on the
   reserved 10%. Patience and best-checkpoint selection begin in this phase.
   Restore its best checkpoint without refitting on the reserved people.

The holdout fraction is `secondary_nn_model_validation_holdout_fraction: 0.1`.
Warm-up validation is a diagnostic signal rather than independent validation.
Its values are not compared with the reserved holdout to choose a checkpoint.
Network, optimizer and scheduler state persist across rotations and the phase
transition. Maximum epochs must exceed the warm-up period.

The schedule applies separately within each allowed expert training pool; it
never admits the selector's outer validation people or an excluded cross-fitting
fold. `secondary_nn_crossfit_folds` remains three for unbiased selector inputs.
There are four production expert fits with distance and sixteen expert fits
without distance, plus one selector fit, rather than 24 and 96 network fits.

`cross_validation_folds.json` records reserved and rotation people;
`loss_history.json` records the phase, validation fold per epoch, best final-phase
checkpoint, and zero refit epochs. PNG/PDF/SVG curves mark the phase boundary.
The seed fixes membership and rotations. Feature transforms remain fixed during
training. The selector's outer evaluation remains separate from early stopping.

Historical `cross_validation` (independent folds plus refit) and `fixed` strategies
remain available explicitly. Local/short-range overrides still inherit the
shared controls when null.

## Short-range origin choice sets

Short-range training and prediction call the same origin-neighbourhood query
in `tools.py`, in both target-distance modes. Alternatives are finest H3 cells
around the origin, across all hierarchy branches. Both phases start at
`secondary_nn_short_trip_threshold_m + 1.5 * secondary_nn_short_query_radius_m`
(default 1,800 m), filter cells using actual facility offers for the requested
purpose, and expand by `secondary_nn_short_query_radius_m` (default 200 m) until
at least `secondary_nn_short_min_candidates` (default six) eligible cells are
available, or all eligible cells have been included. The candidate order is
stable; padded training slots remain masked. Secondary purpose names use the
same base facility offers as inference.

The observed destination never determines the neighbourhood or adds an extra
alternative. Training observations with an unsupported observed cell, missing
cell, invalid origin, or observed cell outside the origin neighbourhood are
excluded with explicit reasons. `short_range_choice_sets.parquet` preserves the
person/trip audit, while `short_range_choice_sets.json` contains counts and query
settings. Neither the observed middle-level H3 parent nor its children restrict
the short expert. The default training lower distance bound is the same 10 m
stationary-trip cutoff used during assignment.

The earlier CV-tested hybrid (distribution 0.05 plus origin relative-distance
0.05) remains available with an explicit short `crps_energy` override. The
current default is cross-entropy plus asymmetric relative-distance scoring.
Hierarchy and distance-free experts retain the shared distribution weight 0.1.
Previous sweep reports used earlier choice-set/training versions; their rankings
do not establish performance of the current model.

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
3. The normal four expert stages use development people with their own reserved
   early-stopping holdout. Their predictions on outer validation people provide
   the untouched final evaluation. There is no permanent 30% gate reservation.
4. The gate fits a small regularized neural network. Its normalization is fitted
   only on gate-fitting people, and stopping uses development stopping people.
   It follows the same rotating warm-up and reserved-holdout schedule without
   reinitialization or refitting. Outer outcomes are used only for final diagnostics.

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
| `secondary_nn_short_distribution_loss_weight` | 0.05 (distance-enabled short-range only) |
| `secondary_nn_short_relative_distance_loss_weight` | 0.05 (distance-enabled short-range only) |
| `secondary_nn_short_distance_loss_kind` | `relative` (distance-enabled short expert only) |
| `secondary_nn_short_relative_underestimation_factor` | 2.0 (origin radius only) |
| `secondary_nn_short_sampling` | `neural` |
| `secondary_nn_model_validation_strategy` | `rotating_holdout` (all destination experts and selector) |
| `secondary_nn_model_min_epochs` | 30 |
| `secondary_nn_model_validation_rotation_epochs` | 3 |
| `secondary_nn_model_validation_holdout_fraction` | 0.1 |
| `secondary_nn_model_validation_folds` | 5 |
| `short_range_model_validation_strategy` | null (inherit shared strategy) |
| `short_range_model_min_epochs` | null (inherit shared minimum) |
| `short_range_model_validation_folds` | null (inherit shared folds) |
| `local_model_validation_folds` | null (inherit shared folds) |
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
are obsolete. Three folds require four expert sets in total with no final refits; fold checkpoints and comparisons are cached for reuse. Fold-only
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

## Distance-conditioned short-trip sampling

The assignment default is `secondary_nn_short_sampling: neural`, retaining learned
attractiveness. `secondary_nn_short_sampling: distance` remains an optional
geometric diagnostic. For trips from 10 m through the configured short threshold
(default 1.5 km), it queries the same purpose-eligible H3 neighbourhood around the
origin and evaluates **actual facility coordinates across all returned cells**.
A facility at radius r receives weight exp(-0.5 ((r - target) / sigma)^2), with
sigma = max(`secondary_nn_short_distance_sigma_min_m`,
`secondary_nn_short_distance_sigma_fraction` * target). Defaults are 25 m and 0.1.
Employee counts, car-access weights and neural leaf probabilities do not enter
this distance-only branch. Purpose offers still determine facility eligibility.

Weights are normalized after subtracting the best radial log-weight. If all
available facilities are far from the target, this retains preference for the
closest feasible radius instead of reverting to a uniform or neural fallback.
The chosen facility comes directly from the shared pools; there is no second
within-cell sampling step. Sparse catalogues can still prevent exact matching.

The neural hierarchy above the short threshold and the distance-free short expert
and selector are unchanged. Set `secondary_nn_short_sampling: neural` to use the
learned short expert. Its current distance-enabled training objective is
cross-entropy plus asymmetric relative-distance loss, described below. Training and CV remain
available with either assignment policy; switching this policy only requires
reassigning locations, so completed expert checkpoints can be reused.

The October 26 diagnostic found systematic short-distance underprediction after
origin-based training: observed trips averaging 843 m were predicted at 482 m,
and trips averaging 1,231 m at 852 m. Validation loss continued improving to about
epoch 42, so this did not provide strong evidence that overfitting explains this bias.
Distance weighting had only acted after the neural model chose a leaf and could
not recover facilities in other cells. The paired distance-only counterfactual
predicted 851 m and 1,231 m for those same bands. This supports addressing the
sampling bottleneck, rather than treating another loss weight as a sufficient fix.

The paired short-trip diagnostic reduced mean absolute radius error from 201.7 m
to 45.0 m. The closest feasible facility radius differed from the target by a
median 0.8 m (90th percentile 8.1 m), so missing facilities were not the main
cause of the large systematic errors in this city replay. These are conditional
fitting-record diagnostics, not estimates of independent generalization accuracy.

## Asymmetric relative-distance loss for the distance-enabled short expert

The default learned short expert trains on observed trips from 10 m to 1.5 km.
Training and inference both obtain alternatives from H3 cells around the origin,
starting at 1.8 km and expanding by 200 m until the configured minimum number of
purpose-eligible options is available. The wider alternative set permits
predictions beyond 1.5 km; this is deliberate and is separate from the observed
trip threshold used to choose training examples and route inference.

With `secondary_nn_short_distance_loss_kind: relative`, the per-trip objective is

```text
L = -log p(observed cell) + 0.07 * [a(e_origin) Huber(e_origin) + Huber(e_home)]
e_origin = (sum_j p_j distance(origin, cell_j) - target_distance) / (target_distance + 100 m)
e_home   = (sum_j p_j distance(home, cell_j) - observed_home_distance) / (observed_home_distance + 100 m)
a(e) = 2 if e < 0, otherwise 1
```

`secondary_nn_short_relative_underestimation_factor` sets the multiplier (default
2, minimum 1). Equal relative errors therefore cost twice as much when the mean
origin radius is too short. The home-radius term remains symmetric. Cross-entropy
retains supervision for destination attractiveness; CRPS, spatial energy and the
auxiliary hybrid term are absent in this setting. The main relative coefficient
is `secondary_nn_distance_loss_weight` (default 0.07).

The same objective and masks apply during fold fitting, validation, early stopping
and checkpoint selection. Loss history records the multiplier and other loss
settings. The shared rotating warm-up and reserved-holdout schedule applies. The hierarchy and distance-free short expert use their existing losses
with multiplier 1; the learned selector is unchanged. A multiplier of 1 recovers
the symmetric legacy relative loss. Historical focused loss-trial runners pin 1
explicitly so their old labels do not silently acquire asymmetric scoring.

## Prediction-time soft accessibility weights

`secondary_nn_no_car_no_pt_weight` defaults to 0.05 in location assignment.
For a person without both car availability and a driving licence, each H3
alternative receives weight 1 if it contains a purpose-eligible facility with
OVGK A, B, C or D, and 0.05 otherwise. Support is derived from actual facility
offers and propagated from coordinate leaves to their middle and root nodes.
A workplace with good PT does not establish leisure accessibility in that cell.
The arrays align with each wrapper's candidate order, including masked padding.

Prediction uses `softmax(neural_utility + log(accessibility_weight))`, equivalently
multiplying the original choice probabilities by the positive weights and
renormalizing. Multiplying raw utilities by 0.05 would be incorrect for negative
utilities. The rule applies at all hierarchy levels and within the short-range
origin neighbourhood, preserving learned attractiveness and purpose masks.
All-no-PT choice sets retain their original relative probabilities; no destination
is removed by this policy. Conditional normalization at each level means the
factor specifies local choice odds, not a guaranteed global facility-odds ratio.

Usable car access requires the driving-licence mobility feature as well as car
availability. Missing licence information is treated conservatively. Car users
with a licence retain their original probabilities. Final facility sampling uses
the same 0.05 no-OVGK factor when usable car access is absent, including mixed
PT/no-PT cells. Optional direct-distance short sampling applies the same positive facility
weights across its origin neighbourhood. The detour feature and learned selector remain in place.

This policy is prediction-only. Validation and expert/gate fitting retain their
original objectives and distributions. A value of 1 disables the new cell policy
and retains the historical within-cell 0.01 no-OVGK facility weighting. Values
must lie in (0, 1]. Report replay reads the policy from the completed assignment;
historical assignments lacking the option replay with value 1. Report provenance
and interpretation notes distinguish the imposed soft rule from learned effects.

## Ordered public-transport accessibility and shared preprocessing

All scalar ÖVGK inputs use A=0, B=1, C=2, D=3 and no coverage=4.
Each H3 node contains `ovgk_mean_home`, `ovgk_mean_work`,
`ovgk_mean_education`, `ovgk_mean_shop`, `ovgk_mean_leisure` and
`ovgk_mean_other`: arithmetic means over actual purpose-eligible facilities,
without employee weighting. An absent purpose receives 4; the existing offer
counts distinguish absence from poor accessibility. These inputs supplement
class shares at all three hierarchical levels and in the origin-centred short
expert. Original destination class labels remain available to accessibility
masks and other pipeline consumers.

Person inputs include scalar `home_ovgk`. Immediately after car availability,
PT subscriptions use five mutually exclusive indicators: none, GA, regional,
Halbtax, and Halbtax plus regional. Licence and bicycle availability follow.
The shared `mobility` stage computes survey mobility and home spatial lookups
once and is reused by production experts, cross-fitting folds and selector.

Trip state includes scalar `origin_ovgk`, evaluated at the actual origin rather
than an H3 centroid. Survey origins are joined once in `mz_chains`; prediction
uses a shared polygon index and an exact-coordinate cache, including dynamically
assigned preceding secondary activities. Overlapping polygons use the best
available class consistently with home accessibility.

For CRPS and projected spatial energy training losses, fixed sorted positions
and absolute observation errors are prepared once per expert. The same tensors
are reused throughout warm-up and reserved-holdout training.
Sorting no longer runs in every minibatch and epoch. This trades additional
RAM for computation; the training log records the allocated geometry size.
The geometry cache preserves the loss and its gradients. Speedups of the score calculation do not imply the
same speedup for the complete neural training pipeline. New feature dimensions
require retraining all experts and the distance-free selector.
