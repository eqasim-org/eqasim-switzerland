# RDA secondary destination calibration

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
