# Secondary school selection

`locations.assign_locations` retains the existing base assignment and then calls `secondary.assign_secondary_locations` with an independent random stream. The replacement covers classified domestic students aged 15+ with structural-survey study codes 2–8. Children under 15, foreign/unknown destinations, nonstudents and unresolved classifications retain their existing handling. Tertiary OD assignment runs afterwards as before.

Compatibility uses the 2023 structural-survey codebook (sheet 25) and [BFS NOGA 2008 definitions](https://www.bfs.admin.ch/bfsstatic/dam/assets/344611/master). Establishment NOGA is not a verified programme catalogue; these pools are explicit modelling assumptions.

| Study code | Programme | Allowed establishment NOGA |
|---|---|---|
| 2–3 | Compulsory | 853101, 852002, 852003 |
| 4 | Bridging | 853101, 853102, 853103, 853200 |
| 5 | Specialised secondary | 853102, 853103 |
| 6 | Apprenticeship / vocational secondary | 853200 |
| 7 | Gymnasium / teacher seminar | 853102 |
| 8 | Vocational or specialised maturity | 853200, 853102, 853103 |

The compulsory pool includes special and mixed compulsory schools. Bridging has no dedicated establishment code and retains a broad secondary pool. Code 8 combines vocational and specialised maturity, so it must retain both pathways.

Code 5 cannot safely be restricted to 853103: the saved STATENT extract contains only 26 such establishments, and a strict replay pushed the assigned mean for ages 15–17 to 20.93 km. In the pooled 2021–2023 survey, among ages 15–17 with code 5, 930 observations report local NOGA 853102 versus 207 reporting 853103. Allowing both academic site types is supported by these observations, but does not prove which programmes an individual establishment offers.

For each person, search the full compatible pool and retain the 20 establishments with the smallest absolute mismatch to the structural-survey target distance supplied by [secondary_distances](secondary_distances.md). An independent MZ reference draw supplies rank within the recipient age/programme cohort, not a literal kilometre target; no matched donor is needed. This avoids truncating long commutes at the seven nearest schools and avoids reproducing unsuitable MZ distance tails. Computation uses batches of 256 people to bound memory use. Pure-function callers can omit the model to reproduce the earlier donor-target method; the pipeline always supplies the fitted model for STATPOP.

Sample using a Gaussian distance preference with tolerance `max(200 m, 10% of target distance)` and employee weights clipped to the compatible pool's interquartile range, with a minimum weight of one. Tolerance and employee weights are modelling assumptions, not enrolment capacities. Log-space normalisation avoids numerical underflow without a floor that erases distance preferences. Missing survey target evidence uses the nearest compatible school; an empty compatible pool raises an explicit error.

`education_secondary_method` records compatible SS-distance selection, legacy donor-distance selection for callers without a model, a nearest-school fallback for missing distance, or `not_applicable`. Study level and destination-status columns continue to govern eligibility; the assignment group stays `classified_secondary` for downstream analyses.

Validation should inspect means, medians, upper percentiles, programme groups, municipality types, compatibility, and unchanged destinations outside the cohort. Scarce or incompletely identified school sites can still force distances above the donor target, particularly in rural areas. This selection does not calibrate school capacities or destination OD flows.


## SS destination-zone assignment

The compatible distance assignment is now a complete fallback before applying
`assign_secondary_od_locations`. Classified domestic students aged 15+ use the
separate pooled secondary SS matrix (codes 2–8), restricted to zones containing
programme-compatible schools. Before sampling the zone, use each student's
already fitted SS age/programme distance target to condition the pooled row:

`P(zone | origin, target) ∝ P_pooled(zone | origin) × exp(-0.5 × (error / sigma)²)`

Here `error` is the smallest absolute target-distance mismatch among compatible
schools in that zone, measured from the actual synthetic home. It is not a
municipality-centroid distance. `sigma = max(1,000 m, 0.10 × target)` is a shared
soft tolerance across all programmes and ages. This reuses the existing relative
school-distance tolerance with a larger zone floor (1 km instead of 200 m) and
imposes no hard distance cutoff. The scale is an explicit
modelling assumption, not fitted to the 15–17 aggregate. A zone does not gain
weight merely by containing duplicate school records. The pooled OD row still
supplies direction and zero-probability routes remain unsupported.

The existing distance/employee preference then chooses a compatible
school inside the sampled zone. Unsupported/zero-compatible rows retain
the labelled fallback. Municipality destinations include their quarters; quarter
destinations require exact membership. This changes secondary destinations while
preserving the fallback's compatibility and distance logic.


The correction addresses the pooled matrix transferring long vocational-school
commutes into compulsory and gymnasium programmes. There is still one secondary
OD matrix; no new programme-specific flow tables are estimated. SS distance cells
retain the existing effective-sample-size backoff. Missing SS targets leave the
compatible pooled OD probabilities unchanged, explicitly labelled
`unconditioned_missing_ss_target`. Conditioned assignments record
`education_secondary_zone_distance_status=ss_age_programme_distance` and
`education_secondary_zone_distance_sigma_m`. Unsupported origin/destination
fallbacks remain separate from that missing-distance case.

`locations.validate` already fingerprints this imported helper, so the next
pipeline run invalidates education anchors and their downstream matching outputs.
Frozen-diary replay validation isolates placement effects; a complete rerun may
also change diary donors because the education commute predictor changes.
