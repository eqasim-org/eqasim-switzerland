# Tertiary OD smoothing and validation

The tertiary matrix stage retains the existing `configure`/`execute` structure
and `(pdf, cdf)` interface. It now supplies valid municipality and quarter
distributions even where observations are absent. Other origin levels remain
NaN because they are outside this domestic model. Destination assignment remains
unchanged.

## Method and configurable defaults

Municipalities borrow observed profiles from nearby municipalities, excluding
themselves. Donor profiles use raw observations, not recursively smoothed profiles.
Gaussian distance weights are multiplied by donor effective sample size capped
at ten, limiting large-city dominance.

Quarters borrow their municipality's observations excluding their own. This
independent parent profile is blended with regional evidence. If no independent
parent observations exist, the quarter uses nearby municipalities. The quarter
stage retains its source municipality ID; the existing municipality mapping
translates it to the reference geography.

Observed share is `effective_n / (effective_n + prior_strength)`, with
`effective_n = (sum w)^2 / sum(w^2)`. Expansion weights are thus not mistaken for
independent respondents. Zero-observation origins use full fallback. The model
does not invent a stay-in-the-home-zone destination for unsupported origins.

| Config setting | Default |
| --- | ---: |
| `tertiary_od_prior_strength` | 100 effective observations |
| `tertiary_od_neighbors` | 50 observed municipalities |
| `tertiary_od_bandwidth` | 20,000 metres |

These achieved the best weighted held-out log loss among tested settings, not a
proven global optimum. Strength and donor count are at the upper edge of the
expanded grid. Bandwidth and the donor influence cap were held fixed. The strong
borrowing reflects the sparse empirical cells and can blur local differences.

## Reproduce validation

```bash
python -m analysis.tertiary_education_od_smoothing \
    --data-path /path/to/pipeline \
    --output-path /path/to/smoothing_audit
```

The analysis also supports the standard pipeline stage interface. Each survey
year is held out in turn, using only the other two years' observations. All years
use the existing reference geography. The expanded grid tests strengths
0, 1, 5, 10, 20, 50, 100 and donor counts 3, 5, 10, 20, 50: 35 settings across
three folds, covering 35,098 held-out observations in total.

Strength zero retains raw probabilities for observed origins and supplies
fallback only where observations are absent. The comparison below uses this
baseline with the same 50 donors. Zero predictions are clipped to 1e-12 for
log-loss scoring only; that floor is not added to the model. Brier scores provide
a second probability-distribution check. Lower scores are better.

| Held-out year | Baseline weighted log loss | Selected weighted log loss | Baseline weighted Brier | Selected weighted Brier |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 14.3395 | 5.0094 | 1.0138 | 0.9255 |
| 2022 | 14.3520 | 4.9339 | 1.0152 | 0.9234 |
| 2023 | 14.6904 | 5.5984 | 1.0177 | 0.9298 |

Combined weighted log loss improves from 14.4603 to 5.1802. Unweighted log loss
also improves, and the selected setting has the best combined weighted Brier
score in the grid. Between 6.0% and 8.8% of held-out destinations still receive
zero probability, compared with 46.5%–48.4% for the baseline. Annual changes and
unseen flows remain limitations.

On the full sample, all 2,401 domestic origin rows are finite, nonnegative and
normalized, with monotonic CDFs ending at one. All 238 previously unsupported
municipalities and eight unsupported quarters receive full fallback. All 239
quarter priors have independent parent evidence in this data. Median observed
share is 4.75% for municipalities and 17.89% for quarters. Median furthest donor
distance is approximately 14.8 km.

Decision checks verified exclusion of quarter evidence from parent priors,
effective sample weighting, no invented home-zone flows, zero-origin coverage
and invalid parameter rejection. The general education matrix is unchanged.

Artifacts at `/private/tmp/tertiary_education_od_smoothing` contain held-out
scores, setting comparisons, per-origin fallback shares and the validated matrix.
Only aggregates and probability matrices are exported.

## Next step

Build institution/campus groups and enrolment-based budgets before connecting
the matrix to assignment. Smoothing does not enforce campus enrolment totals or
guarantee a compatible institution in each predicted zone. Keep observed and
fallback contributions reviewable during integration.
