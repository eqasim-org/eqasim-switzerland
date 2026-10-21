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

The selector also receives origin-to-home/work distances, nearest candidate-cell
distance and nearby H3 opportunity summaries within 1.5 km and 5 km. Quantities
are summed and fractions averaged. Summaries use candidate representative points
around a 250 m origin grid, rather than expensive per-trip polygon overlays.
This is a local supply approximation; it is not a routed travel-time catchment.
No person identifiers, observed destination coordinates, observed trip distance,
or expert comparison outcomes are selector inputs.

## Distance-free expert selection

With `secondary_nn_use_target_distance: true`, target distance remains an input
and selects the short expert for trips up to
`secondary_nn_short_trip_threshold_m` (default 1500 m). Otherwise, the separate
`short_trip_classifier` stage learns **which expert has lower destination loss**.
Its historical stage name is retained; it now saves `expert_selector.pkl`.

The person split is shared across every expert. By default 70% of weekday survey
persons are available for expert training, and 30% are reserved for selector
training and assessment. Reserved persons are excluded before expert feature
scaling and fitting. No expert is refitted on reserved people afterwards, since
that would change the distributions on which the selector was trained.

For every eligible nonstationary reserved trip, the stage draws ten final
facilities from each expert. Both experts use the same observed origin, person
and chain inputs, with independent reproducible random streams. The observed
destination is used only to calculate supervision. For predictions `X_j`, origin
`O` and observed destination `Y`, the score is:

```text
spatial_error = mean_j ||X_j - Y||
length_error  = abs(mean_j ||X_j - O|| - ||Y - O||)
loss = w * log1p(spatial_error / s) + (1-w) * log1p(length_error / s)
```

Defaults are `w=0.5`, `s=1000 m`. Spatial error penalizes wrong direction and
prevents opposite-side predictions from cancelling. The second term matches
mean trip length. Log scaling limits domination by very long excursions. Both
experts can have high loss: the selector regresses `loss_short-loss_hierarchy`
and picks short only when the predicted difference is negative. It does not
turn all expert mistakes into equally important binary labels.

The regularized histogram gradient booster has seven leaves per tree. Selector
persons are split again: 20% untouched outer validation; the remainder has a
separate 20% person-disjoint early-stopping partition. Scalers see only the
fitting partition. Outer validation is never used for stopping or tuning.

The stage writes `expert_comparisons.parquet`, `selector_partitions.parquet`, and
`validation.json`. Metrics include selected loss, always-short/always-hierarchy
baselines, oracle loss, regret, expert-choice accuracy and short-selection share.
These assess performance on people unseen by both experts and the selector.
They do not establish geographic generalization or accuracy of full-day rollouts.

Optional controls and defaults:

| Configuration | Default |
|---|---:|
| `secondary_nn_selector_person_fraction` | 0.3 |
| `secondary_nn_selector_draws` | 10 |
| `secondary_nn_selector_context_radii_m` | [1500.0, 5000.0] |
| `secondary_nn_selector_validation_fraction` | 0.2 |
| `secondary_nn_selector_spatial_loss_weight` | 0.5 |
| `secondary_nn_selector_loss_scale_m` | 1000.0 |
| `secondary_nn_selector_max_iter` | 150 |
| `secondary_nn_selector_min_samples_leaf` | 40 |
| `secondary_nn_selector_l2` | 10.0 |
| `secondary_nn_short_query_radius_m` | 200.0 |
| `secondary_nn_short_min_candidates` | 6 |

The short-range training cutoff remains 1500 m; the selector may choose either
expert for any nonstationary trip. Ten draws per expert can make this stage
substantially more expensive than the former threshold classifier. `threads`
controls CPU resources and `random_seed` controls reproducibility.

When target distance is disabled, the final facility draw also omits target-distance
reweighting, so selector replay and population assignment use the same expert
behavior. Observed/matched distances still supply expert distance-loss targets,
identify stationary trips below 10 m, and contribute existing person-day distance
summaries. The switch is not removal of every survey distance from the pipeline.

## Coordinate hierarchy and compatibility

Coordinates are indexed independently at H3 resolutions 5, 7 and 10. Middle and
leaf IDs encode the coordinate path. A fine H3 cell crossing a coarse boundary
has a separate node in each occupied branch. Node polygons are intersected in
EPSG:2056 with every ancestor. Training labels, feature aggregation, destination
pools, sibling fallbacks and short-range lookup all use this shared tree.

**Retrain all four experts and the selector after this update.** Existing model
feature schemas and selector checkpoints are incompatible. The default
`overwrite_*_model_if_exists` options are true. The selector runs only when
target distance is disabled; neither selector training nor its person reservation
is scheduled for distance-enabled runs. Consequently, default distance-enabled
experts see more training people; comparing the two settings is not a pure
feature ablation unless their expert training cohorts are also controlled.

Run the focused tests, including a small end-to-end training/reload/prediction:

```sh
uv run -m pytest tests/test_secondary_nn_enrichment.py -q
```
