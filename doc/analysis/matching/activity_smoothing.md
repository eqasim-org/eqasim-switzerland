# Smoothing chosen for regional diary matching

The default V1/V3 spatial feature remains the four whole-municipality groups.
The matching-oriented settings are now:

```yaml
matching_spatial_feature: activity_chain_cluster
activity_zones_neighbours: 400
activity_zones_min_bandwidth_m: 2000.0
activity_zones_gaussian_smoothing_m: 0.0
activity_zones_high_population_share: 0.16
```

There is still spatial smoothing: a survey-weighted adaptive bisquare kernel
pools nearby respondents, with a minimum 2 km radius. The additional Gaussian
layer is disabled. Municipalities receive one group by resident plurality;
no final group boundary cuts a municipality. The original adaptive 2–32 km
source cells are retained.

## Experiment and comparison rules

The archived smoothing study used a cached-input CLI, rather
than a Synpp stage. It reuses the source cell geometry, estimates new continuous
fields, builds whole-municipality groups and runs production V1/V3 matchers.
Upstream caches remain read-only. It checks that saved source geometry and the
current fitting cohort have matching population/day/seed settings.

The initial settings were 6,400 neighbours + 2 km Gaussian, 800 + 2 km,
400 + 2 km, and 400 + no Gaussian. The controls were the then-current
6,400-neighbour/8 km model and the original municipality-level clustering.
An 800-neighbour/no-Gaussian follow-up was added to examine city tradeoffs.
The objective weights were held fixed for both phases.

All settings were evaluated on **832,194 common recipients**, excluding
forced-home recipients and anyone assigned a missing processed diary by any
variant. Genuine nontravellers remain. The weighted MZ reference contains
37,780 eligible respondents. Every density mapping is checked for exact equality
across runs. Other matching preferences, opening-hours masks, weights, source
eligibility and seed 518745 remain unchanged. Input sampling is 10%.

The selection score averages absolute matched trip-rate error, divided by the
national survey mean for that feature. It gives equal weight to four trip
features, both matchers, and the two geographic levels (26 cantons versus three
cities). Thus population size and the number of canton rows do not overwhelm
city or purpose fit. The historical benchmark is shown but is separate from
selection among the tested lower-smoothing candidates.

## Results

The 400-neighbour/no-extra-Gaussian setting scored 0.05015, versus 0.06622 for
the previous 8 km model and 0.05355 for the original municipality benchmark:
24.3% and 6.3% lower balanced error, respectively.

Equal-canton mean absolute total-trip rate error fell from 0.12167 to 0.07156
for V1, and from 0.11779 to 0.08198 for V3. Both matchers improve total-trip
error against density in 21/26 cantons. The original benchmark's corresponding
mean errors on the same cohort were 0.09163 and 0.08961.

| City | V1: previous 8 km → selected | V3: previous 8 km → selected |
| --- | --- | --- |
| Lausanne | −3.90% → −0.34% | −3.51% → −0.35% |
| Geneva | −2.78% → +1.35% | −3.27% → −1.38% |
| Zurich | +0.11% → +2.60% | +0.44% → +1.75% |

The result is a substantial aggregate gain with a Zurich tradeoff. It is not
uniform improvement for every purpose, region or matcher. Group 1 contains
16.46% of residents after whole-municipality voting and averages 4.09 observed /
4.00 modelled trips/day. Its raw continuous target remains 16%.

This is **descriptive tuning on the fitting survey**, from one paired seed.
It does not establish independent matching or MATSim improvement. The archived
blocked individual-diary prediction score is slightly worse for the 400-neighbour
kernel (1.00653) than for 6,400 (0.99900). Individual diary prediction and regional
matching fit are different objectives; neither score should be presented as
validation of the other. Sparse areas still borrow observations and may have
low effective local sample sizes; the selected field's minimum Kish support
is about 31.6 at the evaluated cell locations.

## Archived evidence and current implementation

The completed study remains under
`analysis_path/matching/smoothing_tuning_5fca93eadd75/`, including its HTML,
CSV scores, candidate fields, maps, plans and selection metadata. The sweep
scripts and alternative stages have been removed from the active code after
selecting the default. The figures and recorded results remain unchanged.

The current implementation is the single
[`data.microcensus.activity_chains_clusters`](../../data/microcensus/activity_chains_clusters.md)
stage, with a focused
[V1/V2/V3 comparison](activity_chain_zones.md). Its option names start with
`activity_zones_`; the older `activity_surfaces_*` and `activity_contours_*`
experiment options are retired.
