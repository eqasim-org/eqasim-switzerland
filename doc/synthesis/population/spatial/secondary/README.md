# RDA secondary destination calibration

## Optional French force-model chain solver

The default remains the existing Swiss RDA gravity solver. To use the
force-field chain relaxation from eqasim-france instead, set:

```yaml
locations_algorithm: rda
secondary_activities.chain_solver: force_model  # default: default
```

`secondary_activities.chain_solver: default` restores the original solver and
does not register or calculate force-field dependencies. The neural secondary
location algorithm is unaffected. The stage still returns the same locations
and convergence tables, including destination IDs and trip indices.

The implementation is adapted from
[eqasim-france commit 7d219a0](https://github.com/eqasim-org/eqasim-france/tree/7d219a070917919625c795e5b9fd0071184f6b67/synthesis/population/spatial/secondary/force_model)
under GPL-2.0. It uses the upstream settings:

| Setting | Default |
| --- | --- |
| `secondary_activities.force_model.attractors` | `uniform` |
| `secondary_activities.force_model.cell_size` | 250 m |
| `secondary_activities.force_model.grid_margin` | 2,000 m |
| Spring relaxation coefficient | 0.3 |
| Mean / maximum field displacement per update | 30 / 60 m |
| Maximum chain-relaxation iterations | 1,000 |
| Movement convergence tolerance / history | 250 m / 50 iterations |

The optional stage
`synthesis.population.spatial.secondary.force_model.force_field` constructs
purpose-specific fields from `synthesis.population.destinations`, covering its
Swiss and foreign facilities in EPSG:2056. Under uniform weighting, each facility contributes one unit
of attraction for every offered `shop`, `leisure` or `other` purpose. Other
purposes receive spring relaxation without a purpose field, as upstream.
Employee weights are still applied during final destination selection.

The force is the sum of `weight * delta / distance**3`, with the upstream
125 m regularization for distances below 1 m. Every attractor contributes;
there is no radius cutoff or grid approximation of attractor coordinates.
Serial Numba kernels process grid-row batches in Synpp workers (up to 24),
avoiding a separate full grid and temporary coordinate arrays per worker.
The first field computation can still be expensive: its work scales with
grid-cell count times attractor count. Synpp caches and reuses the fields.

Only the continuous solver for chains with both primary anchors changes.
Single-anchor tails and anchor-free chains retain the current solvers. Sampled
distance distributions, their Swiss resampling, ÖV quality filtering for people
without cars, candidate employee weights, opening-hours checks, detour factors,
country proposal acceptance and optional fallback repair remain in use. Field
attraction is static guidance from the whole catalogue; eligibility by person
and activity time remains enforced by final candidate selection. The existing
Swiss physical objective and 200-attempt outer limit are retained; continuous
force-model convergence alone does not imply a valid destination assignment.

Two numerical details differ from upstream: a zero field stays zero rather
than producing NaNs during normalization, and bilinear interpolation uses
standard corner weights so a grid node returns its own force. All random
directions use the existing worker's seeded generator. With the option off,
the gravity solver parameters and random draws remain unchanged.

### Employment-density calibration

To test the Vaud scenario under the pipeline's `config` section, use:

```yaml
locations_algorithm: rda
secondary_activities.chain_solver: force_model
secondary_activities.force_model.attractors: employment_density
secondary_activities.force_model.canton_ids: [22]
secondary_activities.force_model.density_beta.other: 0.2
```

The exponents for `shop`, `leisure` and `other` default to zero. The region
defaults to all Swiss cantons. Each selected facility receives a weight
proportional to `(1 + jobs_per_km2) ** beta`, normalized to mean one across
the selected region for each purpose. A 2 km Gaussian smooths employment from
the configured STATENT stage on a 250 m LV95 grid. Density uses total surface
area, including lake surface. Facilities outside the region keep unit weight.
The resulting field extends across the region boundary.

Run `synthesis.population.spatial.secondary.locations` to see the effect on
complete plans. For the local paired calibration trial, run from the repo root:

```bash
PYTHONPATH=. uv run python "analysis/secondary locations/force_density_trial.py" config_dib_10.yml output/force_density_trial --draws 12
```

The trial uses weighted MZ 2015 weekday home–other–home tours with home
within 8 km of Lausanne Bel-Air and observed leg length up to 8 km. It fixes
the observed radius, uses one relaxation and Swiss facility selection per
draw, and compares uniform weights with exponents 0.10 and 0.20. The field
uses a regional catalogue and regional normalization. The trial reports the
central hexagon share, total variation across five concentric hexagonal bands,
paired household-bootstrap intervals for the central share change, and mean
absolute leg-length error. The production run also samples distances and
checks complete chains, eligibility, detours and retries.

### Testing facilities for errands and escorts

The survey codes available for ordinary MZ arrivals are 5 (errands and services),
9 (accompanying children) and 10 (accompanying another person). Their MATSim
activity type is `other`. The 2015 question F520.00 records reasons for escorting.
Facility evidence comes from comparing trip endpoint coordinates with STATENT
establishments and GTFS stops. Nearby sectors can overlap and the catalogue
has no explicit private-home visit alternative.

Two analysis scripts use a completed spatial cache and MZ 2015:

```bash
PYTHONPATH=. uv run python "analysis/secondary locations/other_facility_trial.py" config_dib_10.yml output/force_density_trial
PYTHONPATH=. uv run python "analysis/secondary locations/other_facility_snap.py" config_dib_10.yml output/force_density_trial
```

The first script counts weekday visits and respondents, reports facilities
within 25 m of the best-quality geocoded destinations, and estimates facility
preferences using 2015 diaries outside the validation sample. For location-choice
validation it compares the catalogue point nearest the observed destination
with distinct catalogue points found at the same trip radius from the origin.
It reserves Zürich, Vaud and Geneva destinations for validation and excludes
their respondents from estimation. It writes aggregate results and coefficients.
The second script tests the incremental subtype preferences on the Lausanne
return tours, in final facility selection alone and in both the attraction
field and final selection. The 25 m facility-profile error is total variation
across single nearby sectors, multiple sectors and unmatched coordinates.

Enable subtype-specific assignment under the pipeline's `config` section with

```yaml
locations_algorithm: rda
secondary_activities.chain_solver: force_model
secondary_activities.force_model.split_other: true
```

Use `true` for all three subtypes or `[5, 9]` for errands and child escorts.
Unlisted codes retain pooled preferences. The default is `false`. The option
works with uniform or employment-density attractors. Run `synthesis.population.spatial.secondary.locations`. The MZ
reader carries the reported subtype through loop splitting and trip
renumbering. The assignment joins each recipient to the processed donor trip
and uses the subtype in the attraction field and final facility choice.
The broad activity type written to plans stays `other`. Unclassified visits
and initial activities without an arrival code use pooled preferences. French
proposal acceptance still applies once per broad purpose in a chain.

Each Swiss candidate receives a multiplier derived from its distance to each
STATENT sector and to GTFS stops. The features decay exponentially with a 50 m
scale. The coefficients in
[the MZ 2015 preference table](../../../../../calibration/secondary_locations/mz2015_facility_preferences.csv)
are differences from the pooled fit. Sites outside Switzerland have a
multiplier of one. Pass `--export-preferences <path>` to
`other_facility_trial.py` to regenerate the table.

### Calibrating city centres and child escorts

The [MZ 2015 parameter settings](../../../../../calibration/secondary_locations/force_model_mz2015.yml)
can be merged into the `config` section to run
`synthesis.population.spatial.secondary.locations` with the calibrated force model.

`secondary_activities.force_model.shop_centres`, `leisure_centres` and
`errand_centres` each accept a list of LV95 coordinates, radii in metres and
peak inward forces in metres per relaxation step. All three keys use the
`secondary_activities.force_model.` prefix. For a stop of the specified
purpose at distance `r` from a centre of radius `R`, the additional
inward force has magnitude `strength_m * 6.75 * (r/R) * (1 - r/R)^2` inside
the radius. The force vanishes at the centre and at the boundary. Overlapping
centres add their vectors before applying the ordinary solver ceiling of 60 m.
The additional force applies during relaxation between fixed anchors. An empty
list (the default) switches it off for that purpose. Only `errand_centres`
requires code 5 in `split_other`. The MZ 2015 settings apply to shopping within
4 km of Lausanne Bel-Air and leisure within 8 km of Lausanne Bel-Air and
Zürich HB. The national force grids can be reused when changing these settings.

`secondary_activities.force_model.child_escort_school_multiplier` changes
facility selection for MZ code 9. It multiplies the existing choice weight
for eligible Swiss locations within 25 m of a STATENT school or daycare.
One positive number applies to all chains. A list of two positive numbers
applies to chains with one secondary stop and chains with two or more,
respectively. The default is one. Both ordinary facility selection and
Swiss fallback repair use the same weights. The survey benchmark measures
proximity to these establishments and cannot identify the establishment the
respondent entered.

### Switches and automatic Microcensus calibration

Under the scenario's `config` section, select the force solver and set the two independent options:

```yaml
locations_algorithm: rda
secondary_activities.chain_solver: force_model
secondary_activities.force_model.split_other: [5, 9]
secondary_activities.force_model.calibration_mode: auto
```

`split_other: false` pools all other activities. The list above separates errands/services and child escort. `calibration_mode: "off"` uses the uniform base field and skips the additional city and school adjustments. The selected subtypes can still have different facility preferences. `manual` (the default) applies the configured employment-density, city and school settings. `auto` compares settings against the configured Microcensus and supplies the winning assignment to downstream stages. With the split disabled, the school and errand-specific adjustments drop out; shopping and leisure can still be calibrated. To re-estimate facility preferences from the current wave in manual or split-only runs, use `secondary_activities.force_model.refit_facility_preferences: true`. Auto enables this for its candidates whenever subtypes are active.

Use the calibration areas, reference strengths, canton selection, density exponents and school multipliers in the force scenario. The search scales these references; a zero reference stays zero. The default candidate scales are `[0, 0.5, 1, 1.5]`. Centre strengths and density exponents are multiplied by the scale. School weights are raised to that power, so zero gives the neutral weight of one. To vary settings independently, supply a list of candidate profiles, for example:

```yaml
secondary_activities.force_model.calibration_scales:
  - 0
  - 1
  - {shop: 1.5, leisure: [1.0, 0.5], school: [1.0, 1.2]}
```

In a profile, unspecified axes use scale one. The axes are `shop`, `leisure`, `other_services`, `school` and `density`. Per-centre lists follow the order of that purpose's configured centres. School lists follow single- and multi-stop chains. Density lists follow shopping, leisure and other. The example assumes two leisure centres.

Run the usual secondary-location stage. A one-run override for the existing 3% force scenario is:

```bash
uv run -m synpp config_dib_force_3pct.yml --config secondary_activities.force_model.calibration_mode auto --run synthesis.population.spatial.secondary.locations
```

Each distinct candidate runs the production assignment on the scenario sample using the same upstream population, configured random seed and worker count. Allow for one assignment per candidate and a new grid for each distinct density/attractor setting. For a larger production run, the stage exports the selected manual settings for reuse.

The survey benchmark uses weekday respondents aged six and over, their person expansion weights and domestic destinations. One fifth of respondents are reserved deterministically, including during facility-preference fitting. Facility preferences use accurate domestic trips and same-distance alternative sites. The geographic objective averages the errors of supported targets: the total variation of visit shares across nested 1, 2, 4 and 8 km hexagons around each configured centre, and the difference in child-escort school-proximity shares for single- and multi-stop chains. With pooled other activities and density weighting, the configured errand areas supply pooled-other targets. The default minimum is 20 respondents per target per partition (`secondary_activities.force_model.calibration_min_respondents`). The fit partition selects the winner; ties favour the first profile. Holdout error and chain validity are diagnostics. The holdout tests geographic fitting within the same survey wave and synthetic population.

The cache of `synthesis.population.spatial.secondary.force_model.calibration` contains `scores.csv`, `target_shares.csv`, `candidates.json` and `selected_settings.yml`. Target shares are fractions; centre bands 0–3 mean 0–1, 1–2, 2–4 and 4–8 km, and school bands 0/1 mean farther than/within 25 m. Merge the exported settings into the scenario config to reproduce the selection using `manual`. The facility-preferences stage also writes the fitted coefficients and respondent counts. Both the calibration and the assignment use the same data root. When updating the Microcensus, supply the files used by the Swiss MZ reader (`microcensus/wege.csv`, `etappen.csv`, `zielpersonen.csv` and the usual household inputs) and harmonise any changes in field meanings, purpose codes or coordinate systems first. No saved experiment outputs are required.

RDA's `CandidateIndex` normally queries the 20 closest eligible facilities to each
continuous relaxation point, then samples among them with attractiveness
`number_employees ** 0.8`. Multiplying all French attractiveness weights by the
same factor cannot change a query whose 20 candidates are all French.

## French proposal rejection and resampling

`secondary_rda_france_acceptance` controls acceptance of French secondary
proposals. It defaults to **1**, preserving the original destination choices and
RDA random draws. It accepts one probability or a mapping of `shop`, `leisure`
and `other`; omitted purposes use 1. Values must be finite and between 0 and 1.

```yaml
# Current MMT calibration; validate again for different populations or settings.
secondary_rda_france_acceptance:
    shop: 0.30
    leisure: 0.04
    other: 0.01
```

These probabilities were calibrated on a completed MMT scenario before the
opening-hours integration. Recheck participation and distances when enabling
opening hours or changing the upstream diary population.

The earlier candidate-distance penalty experiment has been removed. Candidate
ranking uses physical distance with the original KDTree; the country preference
acts only on complete valid proposals through acceptance and resampling.

The normal nearest-candidate selection, facility attractiveness, opening-hours
checks and physical objective remain in use. After a geometrically valid proposal is generated,
French destinations receive the configured acceptance factor. The factors
multiply across distinct French purposes in that chain; repeated visits for one
purpose incur its factor once. For example, French leisure and other activities
at 0.04 each give a chain acceptance probability of 0.0016. This remains two
purpose factors even when their numeric values are tied for calibration.
Primary anchors and other purposes have no country factor. Swiss-only proposals
pass this check immediately.

A rejection triggers another complete RDA attempt, including distance resampling.
The separate acceptance RNG uses the worker seed plus 1,000,003 (modulo 2^32),
without advancing that stream for probabilities 0 or 1. With the preference
disabled, the physical solver follows the original random stream. As before,
changing the worker partition can change results. With the preference enabled,
additional RDA attempts also change later draws within a worker; comparisons
should focus on aggregate outcomes rather than individual causal effects.

The existing 200-attempt limit includes geometric failures and country rejections.
At the limit, the best physical proposal is retained, preferring valid solutions
over invalid ones. A country preference is never reported as geometric validity.
Zero acceptance therefore does **not** guarantee zero French assignments.

The stage still returns `(locations, convergence)`. The convergence table retains
`valid` and `size`, and now also records `person_id`, `activity_index`, `attempts`,
`valid_candidates`, `rejected_valid_candidates`, `preference_accepted`,
`preference_probability`, `preference_fallback` and `preference_invalid_fallback`.
`preference_fallback` means a valid proposal was retained without passing the
country check; `preference_invalid_fallback` means no geometrically valid proposal
was found and the retained destination is subject to a country factor below 1.
These flags describe enabled preferences, not a census of all French destinations
when preferences are disabled. Rejections and both fallback counts are logged.

The French catalogue dependency is conditional on enabled outbound facilities and
an active preference. Probabilities follow the same PT-accessibility row masks as
facility IDs. Dedicated cross-border agents and the neural secondary model are
unaffected. Changing acceptance invalidates RDA assignment and downstream stages;
a full MATSim simulation is unnecessary for an initial destination comparison.

## Calibration and validation

Use a fixed representative person sample, complete activity chains, the configured
seed, original car-availability groups, distance distributions and detour matrix.
Retain non-travellers and dedicated-border residents in the resident denominator.
Count each person once per purpose and once for the combined secondary-purpose
union. A person can appear in several purpose cells; do not sum those cells to
obtain the union.

For sparse purposes, prefer tying parameters over fitting an independent value to
each observed share. A shared leisure/other acceptance parameter reduces the
number of calibration parameters while preserving separate activity labels and
distance distributions. Fit the deduplicated leisure-or-other union as well as
shopping and the all-secondary union; retain each purpose as a diagnostic. Do not
add leisure and other person counts when individuals can occur in both.
Sharing a factor is an assumption to validate: a good pooled total can conceal an
implausible split between purposes. If that happens, retain separate factors and
use a coarse sensitivity range for the sparse purpose, rather than finely fitting
its point estimate or accepting any result because the sample is small.

Treat uncertainty ranges as sensitivity checks rather than exact target bands.
In the November 2026 MMT experiment, French "other" activities had only 18 MZ
respondents (21 trips). Its 17.2 km weighted mean fell to 10.1 km when one respondent
was omitted. Respondent bootstrapping should keep all trips belonging to a person
together and retain WP weights; without the survey design, it does not give an
official complex-survey confidence interval. Inspect medians and upper tails, but
do not tune a separate distance multiplier to such a sparse mean.

Separate accepted French proposals from both types of fallback before lowering
probabilities. If the remaining demand mostly comes from fallback chains, lower
acceptance is not an effective remedy. Diagnose chain length, modes, primary
anchors and feasibility before changing limits or physical constraints. Keep a
sample not used for parameter selection and check random-seed sensitivity.

Compare weighted raw weekday MZ residents and model residents using the same age,
home perimeter and destination perimeter. Treat purpose/mode cells with few survey
respondents as uncertain. MZ demand includes dedicated cross-border journeys as
well as ordinary diary-based activities; account for already assigned, identifiable
dedicated-border activity demand when determining the remaining RDA target.
Do not calibrate both populations independently to the full MZ target.

Check convergence, changes in trip/anchor distances and assignment runtime.
Inspect purpose shares as well as the unique union and validate proposed values
on a separate sample before activating them. Small cells do not justify finely
tuned purpose-by-mode parameters.

Local regression checks cover the original candidate ranking and sampling,
coordinate preservation, configuration validation, conditional dependencies and
PT mask alignment. Rejection checks also cover complete resampling, accepted
proposals overriding earlier rejected solutions, valid and invalid fallbacks,
repeated-purpose factors and RNG preservation when disabled.

## Optional local repair of French fallbacks

`secondary_rda_france_repair_attempts` defaults to **0** (disabled). A positive
integer enables an additional, bounded local repair after the normal 200 attempts
end with a country-restricted fallback. It requires outbound facilities and an
active `secondary_rda_france_acceptance`; it does not change those probabilities.

```yaml
# Experimental additional repair budget; validate before adopting.
secondary_rda_france_repair_attempts: 50
```

The repair replaces only the selected French facilities whose purpose acceptance
is below 1. It keeps every primary anchor, all other secondary locations, the
activity sequence, travel times and the original sampled distances. For each
replacement, it queries up to 20 nearest eligible Swiss facilities around the
original selected facility. Swiss membership is checked against
`data.spatial.swiss_border`, excluding French catalogue IDs. The original
car-availability/PT-accessibility mask still applies.

The first trial uses the closest Swiss combination. Further trials sample the
same local candidate sets using employment raised to 0.8, with a uniform fallback
for zero total weight. Duplicate combinations are not re-evaluated. The budget
counts proposal draws, so `repair_trials` (distinct evaluated replacements) can be
smaller than the configured limit.

Each candidate must pass the existing **whole-chain** distance and detour checks.
For an already valid fallback, its objective must also be no worse than the
original. An invalid result is replaced only by a valid result. If the stored
distance-sampling or continuous-relaxation result was invalid, local facility
replacement cannot repair that upstream failure and the chain is retained with
`repair_status=upstream_invalid`. No distances are resampled during this step.
If no acceptable combination is found, the original fallback is returned intact.
This is a deliberately limited repair, not a guarantee of Swiss assignment or
convergence.

Repair candidate and objective RNGs use the worker seed plus 2,000,003 and
3,000,001 respectively, modulo 2^32. The normal RDA and country-acceptance streams
are untouched. All replacement candidates within a chain use common detour random
numbers, preventing repeated evaluations from passing merely through a new draw.
Consequently, enabling repair leaves the original proposals and all unaffected
chains unchanged for a fixed worker partition and seed.

Convergence adds `pre_repair_valid`, `pre_repair_probability`, `repair_attempted`,
`repair_trials`, `repair_success` and `repair_status`. Final `valid` and preference
flags describe the returned assignment; a successful Swiss repair passes the
country preference and is separately identifiable through `repair_success`.
When opening hours are enabled, replacement facilities must also cover the
activity time window using the same tolerance as normal destination selection.
Possible statuses are `not_attempted`, `upstream_invalid`, `no_swiss_candidates`,
`not_applicable`, `exhausted` and `repaired`. Attempted and successful repairs are
logged. The stage result remains `(locations, convergence)`.

Validate on identical residents/seeds against disabled repair. Inspect French
participants and activities, visit episodes inferred from complete country
sequences, purpose-specific distances, convergence and concentration near the
Swiss border. Consecutive French activities constitute one inferred visit episode;
activity diaries alone cannot reveal routes that cross and re-cross the border
between those activities. Dedicated outward journeys need a separate, explicitly
labelled budget because they do not contain observed French activity chains.
