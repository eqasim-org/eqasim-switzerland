# Matching validation and embedding analysis

The [scientific manuscript and Overleaf instructions](paper.md) describe the
paper sources, compiled PDF, evidence and limitations.

## One stage to select saved reports or a full experiment

`analysis.matching.generate_report` is the unified Synpp entry point. It defaults
to `matching_report_mode: saved`, which requires a completed validation directory
and declares no upstream stages:

```yaml
run:
    - analysis.matching.generate_report
config:
    analysis_path: /path/to/analysis
    random_seed: 518745
    matching_report_mode: saved
    matching_report_source: /path/to/analysis.matching.validation__HASH.cache
```

Keep a `working_directory` in the configuration. Run it with
`uv run -m synpp your_report_config.yml`. The existing local
`config_matching_reports_10.yml` selects this saved mode for job 18204004's
evidence. Reports go to `<analysis_path>/matching/rebuilt/report.html`, linking
the comparison and embedding reports. The previous `rebuild_reports` stage and
direct command below remain supported.

To request all predictive tests and reports using current code, select **full**
mode in a copy of the working pipeline configuration:

```yaml
run:
    - analysis.matching.generate_report
config:
    matching_report_mode: full
    # Retain the working config's input/scenario/resource settings.
```

Full mode registers `analysis.matching.validation` as a dependency. It runs
household-disjoint V1/V2 prediction, controls, calibration inputs, grouped
permutation and representation diagnostics, then renders both reports once.
Standard Synpp caching applies: compatible completed validation is reused;
changes to training/evaluation settings or implementation invalidate it. Use a
fresh `working_directory` to request an independent new experiment rather than
deleting or modifying a completed cache. Full mode has the same expensive
training defaults as validation; it is never selected implicitly by missing
saved artifacts. The report code does not run the repository's unit-test suite.

Both population matching and full validation now call the public
`synthesis.population.matching.V2.DiaryMatcher` estimator. The analysis no longer
reaches into the matching stage's private fitting or preparation functions.
Permutation diagnostics use that same estimator's prediction method.

## Rebuild reports from a completed run (no training or matching)

For an existing completed experiment, the simplest command is:

```bash
uv run -m analysis.matching.rebuild_reports config_dib_10.yml
```

This reads the configuration for its cache/output paths and plotting settings.
It **ignores the configuration's `run` list** and does not invoke Synpp. It
loads saved CSV/NumPy evidence only: no model loading, training, donor inference,
population synthesis or simulation. The original cache and reports are preserved.
New reports go to `<analysis_path>/matching/rebuilt/report.html`, with comparison
and embedding reports in its subdirectories. Expect a few minutes for plotting,
CSV reading and household bootstraps, rather than hours of model training.
Actual elapsed time and plotting/bootstrap settings are recorded in
`report_settings.json`.

Automatic source discovery requires exactly one completed
`analysis.matching.validation__*.cache` directory under `working_directory`.
If there are several experiments, select one explicitly; the newest is never
chosen implicitly:

```bash
uv run -m analysis.matching.rebuild_reports config_dib_10.yml \
    --source /path/to/analysis.matching.validation__HASH.cache
```

For use as an independent Synpp stage, copy your working configuration, replace
the entire `run` list, and add the completed validation directory:

```yaml
run:
    - analysis.matching.rebuild_reports
config:
    analysis_path: /path/to/analysis
    random_seed: 518745
    matching_report_source: /path/to/analysis.matching.validation__HASH.cache
    matching_embedding_plot_method: tsne
    matching_embedding_plot_sample_size: 1500
    matching_validation_bootstrap: 500
```

Keep a `working_directory` in that Synpp configuration. The new stage has **no
upstream stage dependencies**, so it cannot schedule the expensive validation.
Missing/incomplete artifacts produce an error instead of triggering training.
The direct command above always renders; Synpp may reuse a completed report
stage until its evidence, code or declared settings change.

Useful direct-command overrides are `--output /path/to/reports`, `--method pca`,
`--sample-size 1000` and `--bootstrap 200`. Defaults use nonlinear t-SNE, with a
linear PCA reference and alternative t-SNE perplexities to check sensitivity.
Numerical thread pools are limited to one during rendering to avoid overhead
on these small plotting calculations. This does not alter any model settings.

### What readers see

Every figure is preceded by a simple account of the experiment/calculation and
instructions for reading its points, ranges and preferred direction. The report
opens with the saved test size and data-derived V1/V2 differences. Detailed
tables are expandable after the figures.

The comparison report includes nine figures:

- Paired predictive skill relative to V1, raw-input neighbours and a stratified
  random donor control, with household-bootstrap intervals.
- Distributions of per-person distance, travel-time and ordered-chain errors.
- Ordered-chain prediction skill for stayed-home, education, work and other
  travelling days, defined by the hidden observed day.
- Calibration of expected daily distance, travel time and trip count.
- Prediction-interval coverage at 50%, 80% and 95%, with interval widths in CSV.
- Paired behaviour skill across age, original income class, child counts and
  residential density groups. These are reporting groups, never matching masks.
- Activity/mode event reliability, randomized ranks and a smooth age-skill curve.

The embedding report includes five figures: the main nonlinear input-coloured
view; the same people coloured by hidden observed behaviour; projection
sensitivity using identical people; learned versus raw-neighbour behaviour
distance; and grouped predictive input contributions. Fold/seed geometry,
stability and projection trustworthiness remain available as tables. The main
view is a sample of one held-out fold and model seed; independently fitted
latent axes are never pooled.

The completed validation is already a **microcensus-to-microcensus experiment**:
each hidden person's own diary and household are excluded from their donor pool.
Allowing someone to match to themselves would give artificially good scores.
The proper scores assess the possible-day distribution against the hidden
person's actual day, rather than treating national survey shares as population
truth.

The original validation version saved scores, predicted means, rank counts,
interval indicators, embeddings and nearest neighbours, but **did not archive
every sampled donor ID or exact observed chain label**. Report rebuilding cannot
recover exact sampled activity-chain frequencies or full predictive distance
distributions. It shows saved chain-probability errors and distance/time CRPS,
calibration and coverage instead. Predicted means are labelled explicitly and
are never presented as distributions of individual days. The saved nearest
neighbours are geometry diagnostics, not the actual stochastic V1/V2 donor
ensembles. Exact donor-draw distributions require richer artifacts from a future
validation run; rebuilding does not silently run that experiment.

## Run a new validation experiment (expensive)

These stages replace `analysis.compare_matching` and
`analysis.matching_embedding`:

```yaml
run:
    - analysis.matching.compare_matching
    - analysis.matching.matching_embedding
```

Copy the working configuration, replace its `run` list with the above, retain
its input/scenario settings, and run the resulting file with
`uv run -m synpp config_matching.yml`. Both reports reuse one validation stage;
they do not run full-population V1/V2 matching or MATSim. Existing input caches
are read-only. The analysis still needs the sampled population and its work
locations to audit target covariates.

## The scientific question

Matching should reproduce the **conditional distribution of behaviour given
available person, household and residential characteristics**. National survey
activity-chain shares are not a population truth target: different demographic
composition, selection bias or survey weighting can change those shares.

The primary test is: **given covariates for an unseen survey respondent, how
well does an algorithm's predictive donor distribution describe that person's
withheld observed day?** A person can have many reasonable days, so predicting
one exact chain is not the sole objective. Predictions are distributions over
complete observed diaries, evaluated with proper scoring rules.

Population transport is a separate question. Population covariates are used to
measure support and estimate which held-out errors matter to that covariate
mix. There are no observed population diaries with which to establish absolute
population accuracy. Hidden survey selection depending on unmeasured behaviour
cannot be identified from these data alone.

## Protocol

1. Harmonize inputs with V2, recover household counts on the full population,
   and sample survey households reproducibly. Keep all respondents from a
   household together if a household ID exists; otherwise use the survey
   respondent ID. The current survey identifies its respondent using HHNR.
   The target-covariate universe excludes under-survey-age and cross-border
   people: their diaries are absent from this survey universe or replaced by
   separate border modelling. The manifest records this explicitly. This is an
   evaluation-universe definition, not another V2 inference mask.
2. Create fixed, shuffled outer household folds. Within each outer training
   partition, fit preprocessing and use another household split for early
   stopping. Outer test outcomes never train, scale or select the model, and
   outer test people are never donors. Models are repeated with different
   seeds on the **same outer folds** to measure training stability.
3. Call the **actual unchanged V1 stage** through a fold-local context containing
   only training respondents, their trips and appropriately coded pseudo-census
   test covariates. V1 retains its own bins, filters, priorities and fallback;
   it is not reimplemented as a superficially similar classifier. Inputs use
   V1's native codes, including the literal `None` PT-quality value, rather
   than V2's normalized categorical strings. Test people
   are treated as ordinary private-household respondents, not hypothetical
   collective residents. Call V2 using its actual encoder and donor sampler.
4. Include two controls: survey-weighted random donors within employment/student
   strata, and raw-covariate kNN with those same strata. A complicated neural
   representation should demonstrate gains over these controls, not just V1.
5. Produce equally sized donor ensembles for every algorithm. Source outcomes,
   indices, fold assignments, scores and neighbour IDs are saved to make donor
   leakage and unsupported predictions auditable.

The evaluation currently requires STATPOP exact-age semantics. ARE needs a
separate interval-aware V1 adapter before a faithful comparison is supported.

## Predictive outcomes and scores

Each real day is represented by trip count, total travel minutes, distance,
first departure and last arrival, purpose-specific participation/counts/dwell
minutes, and mode counts. Non-travellers have zero totals and a separate
mobility flag. First/last time is structurally zero for non-travellers, rather
than a claimed midnight observation. Travel/dwell totals ignore invalid
negative/missing durations. Validate source diary quality before interpreting
small timing differences.

The output includes:

- **Energy score:** multivariate predictive skill on numeric day summaries.
  Scales are set by outer training respondents only, using survey-weighted
  standard deviations; zero-variance dimensions use scale one.
- **Variogram score:** a complementary dependence-sensitive score for selected
  pairs among trip count, travel time, distance and first/last times. It is
  proper but not strictly proper for the entire joint distribution.
- **CRPS:** separate distributional errors in original units for trip counts,
  travel minutes, kilometres, timing, dwell and mode/purpose counts.
- **Brier scores and reliability diagrams:** mobile, work, education, car-driver,
  PT and leisure events. Reliability bins include household-bootstrap intervals
  and effective sample sizes.
- **Chain Brier score:** a per-person categorical score on the joint ordered
  purpose/mode chain, including the initial purpose. This tests a withheld
  person's predictive distribution; it is not a national chain-share chart.
- **Randomized ensemble ranks, interval coverage and widths:** diagnostics of
  predictive calibration. Randomized ties include the observed member in the
  rank denominator. Discrete outcomes and small ensembles affect coverage and
  rank resolution; descriptive plots are not independent hypothesis tests.

The energy/CRPS implementation scores each empirical ensemble using the
V-statistic convention. Equal draw counts are essential; larger draw counts
reduce Monte Carlo effects. It does not claim the unbiased finite-ensemble
U-statistic convention. Numeric summaries do not uniquely characterize all
possible diaries; the ordered-chain score complements them. Neither metric
checks final assigned spatial coordinates or post-mode-choice simulations.

No national activity-chain distribution is treated as a target.

## Bias-aware population transport and uncertainty

The comparison reports three estimands separately:

1. Unweighted survey-respondent predictive risk.
2. Survey-weighted predictive risk.
3. Estimated target-covariate risk: survey weights multiplied by clipped
   cross-fitted source-to-population density ratios.

The domain classifier uses covariates only and equalizes source/target prior
mass. Report its out-of-fold AUC, weight clipping, effective sample size,
category/missingness support and nearest-source covariate distances.
`covariate_balance.csv` reports source-to-target standardized numeric mean
differences and categorical total variation before/after transport. Large
remaining imbalance is evidence against trusting the transported risk; a high
AUC is a domain-separation diagnostic, not predictive travel skill. A regularized
logistic density-ratio model is an approximation and should be checked or
replaced if balance is poor.

Transported risk requires conditional exchangeability, adequate covariate
support and meaningful ratios. It **does not correct unobserved survey bias**,
prove future-year behaviour, or make survey day outcomes population truth.
Clipping deliberately trades variance against bias. The classifier's feature
encoding uses all covariates but no travel outcomes; behaviour models and outcome
scales remain outer-fold-local. Population covariates are themselves partly
imputed upstream. Holdouts with observed respondent covariates do not reproduce
all upstream measurement or imputation error.

Model repeats are averaged per respondent before household-cluster bootstrap.
V2-minus-V1/control intervals are paired on the same respondents, not separate
unpaired intervals. The main figure displays differences as percentages of
each reference score, with numerator and denominator resampled together;
raw-unit differences are also saved. Percentages are undefined if the reference
risk is zero. Employment/student mask-violation rates are exported explicitly.
Coverage is reported separately; paired score differences
apply only to respondents covered by both algorithms. Cohort risk tables cover
age/density quantiles, original income classes, household size/children, sex,
region and status. Their reporting bins never constrain matching.

Intervals condition on the fitted models and estimated transport weights. They
are not a full retraining bootstrap or a survey-design variance estimator with
unavailable PSU/stratum information. Percentile intervals can be degenerate for
small or all-zero event bins. A publication should predeclare its primary
outcomes, add sensitivity analyses and use enough households and model seeds.
Do not read dozens of secondary intervals as simultaneous significance tests.

## Embedding evidence beyond attractive projections

`analysis.matching.matching_embedding` exports these diagnostics:

- Original-space effective rank and behaviour-distance correlation within
  employment/student strata, preventing trivial status separation from
  masquerading as learned behavioural structure.
- Held-out-to-training neighbourhood behavioural distance at k=5 and k=20,
  with coverage and a raw-input neighbourhood reference. These are descriptive
  geometry measures; probabilistic predictive scores remain the main test.
- PCA or t-SNE illustrations of one held-out fold, coloured by exact age,
  original income class and log density. Projection trustworthiness measures
  how well the plotted local neighbourhood reflects the 16D representation.
  Separations in a 2D picture alone are not scientific evidence of clusters.
- Across-seed neighbourhood Jaccard overlap and pairwise-distance Spearman
  correlation on identical outer test people. These are invariant to arbitrary
  rotations of the latent axes. Embeddings from different folds are never
  concatenated as though their axes were aligned.
- Grouped held-out permutation contributions to the **actual donor-transfer
  energy score**, with paired household intervals. Joint household-count fields
  move together. Permutations occur within employment/student strata and use
  common random draws for baseline/perturbed donor sampling.

Age, income, household composition, density, residential context, mobility
resources, work context, personal context and border proximity have separate
contribution groups. These are **predictive associations**, not causal drivers.
Correlated groups can substitute for each other; permutation breaks dependence
between groups even while preserving it within a group. A zero contribution
can indicate insufficient training, correlated substitutes, weak data or
insensitive metrics, not absence of a real behavioural relationship.

## Configuration and artifacts

| Option | Default |
| --- | --- |
| `matching_validation_folds` | 3 |
| `matching_validation_repeats` | 2 |
| `matching_validation_max_persons` | 12000 |
| `matching_validation_population_sample` | 20000 |
| `matching_validation_draws` | 20 |
| `matching_validation_permutation_repeats` | 3 |
| `matching_validation_ratio_clip` | 20 |
| `matching_validation_bootstrap` | 500 |
| `matching_embedding_plot_method` | tsne |
| `matching_embedding_plot_sample_size` | 1500 |

Training settings come from V2, including its epochs, losses, inner validation
and learning rate. Defaults fit six models and perform repeated group
permutations; use a small smoke configuration before a substantive experiment.
The random/control and neural samplers use the same top-k/temperature where
applicable. V1 uses its existing minimum-observation setting.

The shared validation cache contains per-fold model artifacts, train/test
embeddings, neighbour IDs, day-summary arrays, out-of-fold scores, per-person
permutation deltas, feature availability, population support, covariate balance
and a manifest with seeds, splits, code/input hashes and settings. Helpers are
included in Synpp validation tokens to invalidate stale results after an edit.

Final reports live under:

```text
<analysis_path>/matching/comparison/report.html
<analysis_path>/matching/embedding/report.html
```

Figures are standalone PDF and SVG files. Tables, per-person evidence and a
manifest accompany the reports. No web service or publication step occurs.
The diagnostics contain respondent identifiers and individual covariates;
publication figures/tables should be reviewed and de-identified separately.

Local tests remain ignored as required by repository policy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_matching_v2.py tests/test_matching_analysis.py
```

## Research foundations

- [Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf): energy, CRPS and categorical predictive scoring.
- [Scheuerer and Hamill (2015), Variogram-Based Proper Scoring Rules](https://journals.ametsoc.org/doi/10.1175/MWR-D-14-00269.1): complementary sensitivity to multivariate dependence.
- [Sugiyama, Krauledat and Müller (2007), Covariate Shift Adaptation by Importance Weighted Cross Validation](https://www.jmlr.org/papers/v8/sugiyama07a.html): risk transport when covariates differ but the conditional outcome law is stable.
- [scikit-learn trustworthiness definition](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html): quantifying local projection distortion.
- [scikit-learn t-SNE documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html): nonlinear local-neighbourhood visualization and perplexity settings. An attractive separation is not evidence of prediction accuracy.

The combined workflow is a research-informed evaluation design for this
pipeline. It is not a claim of a new theorem, state-of-the-art accuracy, causal
identification, or guaranteed publication acceptance.
