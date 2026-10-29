# Secondary-location behavior comparison

This workflow reads the **completed spatial-assignment caches** of the three
configurations listed in `comparison.yml`. It does not schedule pipeline stages,
rerun MATSim, overwrite run caches, or wait for the final simulation stage.
Simulation input/output paths come from the supplied pipeline YAML files.
Where those files were edited after a run, completed-stage metadata determines
the algorithm and model settings. Production expert versions are selected by their nearest dependency path; deeper
cross-fitting checkpoints are excluded from replay. Dependency versions are resolved from the
latest completed full-spatial or secondary-assignment root rather than whichever
dependency pickle happens to be newest. When only secondary assignment has been
rerun, coordinates are reconstructed from its secondary locations and matching
home/work/education dependencies; older combined-location outputs are not used.

From the repository root:

```bash
uv run python "analysis/secondary locations/run.py" --phase status
uv run python "analysis/secondary locations/run.py"
```

To provide the source configurations explicitly, use `--configs`. Include the
RDA configuration to retain its comparison panels:

```bash
uv run python "analysis/secondary locations/run.py" --configs \
  config_dib_10_rda.yml \
  config_dib_10_nn_distance.yml \
  config_dib_10_nn_no_distance.yml
```

The same arguments work with `generate_report --slurm` to submit only the
report generation to Slurm. Supplying just the two NN configurations produces
a comparison without RDA. The command reads completed assignment caches and
does not rerun the population pipeline or train models.

The default command prepares data, performs survey replay using saved models,
and creates `results/report.html`, `results/findings.md`, figure captions,
machine-readable tables, and PDF/SVG/600-dpi PNG figures. Results stay in this
folder unless `--output` is supplied. The original pipeline configurations and
cached models are read only. Exact-coordinate intermediate Parquet tables are
local analysis artifacts; publication figures use aggregated spatial cells.

For a scheduled analysis job (4 CPUs, 8G **per CPU**, 32G total, four hours):

```bash
bash "analysis/secondary locations/submit.sh"
```

The submitter prints the Slurm job ID. Logs go to `logs/secondary_analysis_%j.log`.
It uses the same stack/GCC/Python modules as the experiment launcher. This job
only runs the analysis. No simulations are submitted.

## Outputs

1. **Distance profiles:** weighted ECDF, mean/median/tail distances, stationary
   and short-trip shares, purpose-specific mean distances and person-cluster
   uncertainty intervals.
2. **Behavioral gradients:** residential ÖV classes A–D/no-class, separately for
   car-access groups; additional age and municipality-type summaries.
3. **Composition-adjusted contrasts:** car access versus no access, and high
   versus low residential accessibility. All methods use the same microcensus
   covariate weights on common support. The exported table reports how much
   weighted support survives this restriction.
4. **Accessibility transitions:** home-to-destination ÖV-class probability
   matrices, split by car access, with a common 0–1 color scale.
5. **Person-level activity spaces:** radius of gyration of secondary destinations
   and repeated-visit shares in 250 m cells, among people with two or more
   secondary-arrival legs. These count people once rather than weighting frequent
   travelers more heavily simply because they have more trips.
6. **Matched survey replay:** the two saved NNs predict destinations for identical
   microcensus people/trips. Each prediction starts at the **observed trip origin**.
   Survey features, including daily-chain features, remain observed. Income
   imputations differ across run caches, so one common reference realization
   (the RDA cache by default) supplies every survey input to both models. Three draws
   per trip by default reveal stochastic variation; their weights sum to the
   original observed trip weight. Paired model differences average draws first.
7. **Geneva and Zürich maps**, separately for shopping and leisure: observed
   survey destinations / NN with distance input / NN with learned selector in
   the top row, and RDA / NN with distance / NN with learned selector for the
   synthetic population in the bottom row. Equal extents, 500 m grid cells,
   normalization, and color scales make the panels comparable. Out-of-frame
   destinations remain in the normalization denominator; the visible fraction
   is annotated. Backgrounds use local municipal boundaries, lakes and exported
   scenario roads, with no dependency on online map tiles. PDF/SVG contain vector
   annotations; the density layers are rasterized for manageable file sizes.

`results/prepared_manifest.json` and `results/replay_manifest.json` identify cache
roots, input artifacts, saved checkpoint files and their sizes/timestamps,
recorded settings, cohort counts, exclusions, random seeds and number of draws.
The report embeds interpretation notes and links all figures and tables.

## Reuse and controls

```bash
# Rebuild just the comparable data tables.
uv run python "analysis/secondary locations/run.py" --phase prepare

# Predict only, using prepared survey records and existing fitted models.
uv run python "analysis/secondary locations/run.py" --phase replay \
  --max-replay-people 2000 --survey-draws 10 --threads 4

# Rebuild statistics and figures from the prepared tables and matching replay.
uv run python "analysis/secondary locations/run.py" --phase figures

# Small end-to-end smoke run in another directory.
uv run python "analysis/secondary locations/run.py" \
  --output /tmp/secondary-comparison-smoke --max-replay-people 30 --survey-draws 1

uv run -m pytest "analysis/secondary locations/test_comparison.py" -q
```

`comparison.yml` defines the three source configs, cities/extents, map purposes,
random seed, thread limit, replay sample/draw counts and minimum common-support
stratum size. Source simulation settings are not duplicated there. Survey people
are sampled uniformly without replacement from the union of selected city-origin
cohorts. All eligible trips of selected people are retained, and both NNs use
exactly that sample. Both cities use square 20 km extents by default. The main
population summaries always use the complete extracted population, not the
replay sample. Increasing replay draws does not increase the number of
independent observed people.

## Scientific interpretation

- The microcensus is a **behavioral reference, not a marginal-distribution target**.
  Focus on the direction and magnitude of conditional contrasts, accessibility
  gradients, destination concentration and activity spaces. Differences in raw
  means can arise from composition as well as destination-choice behavior.
- Statistical cohorts contain weekday, domestic secondary-arrival legs of people
  aged at least six. Synthetic border-crossing people are excluded consistently
  across methods; international legs are excluded geographically. Survey records
  are restricted to domestic endpoints. Audit counts document these exclusions.
- Distances are Euclidean in EPSG:2056. A secondary-arrival leg goes from its
  actual preceding activity to a secondary activity. It is neither a routed
  MATSim distance nor the entire daily travel budget.
- Car availability is harmonized explicitly: survey 0/1 means always/sometimes,
  survey 2 means never; synthetic 1/0 means access/no access. Access is **not**
  ownership or a driving licence. The home accessibility label is computed at the
  actual household home, including for diaries starting away from home.
- Synthetic trip weights are one per sampled agent-trip. Survey trip summaries
  use person weights for each trip; activity-space summaries use each person’s
  weight once. The uncertainty intervals cluster repeated trips by person.
  Household clustering and the full complex-survey design are not represented.
- Standardized car contrasts condition on purpose, age group and home-access
  tier; accessibility contrasts condition on purpose, age group and car access.
  Each common stratum must contain at least ten distinct people in both exposure
  groups of **every** dataset. They estimate descriptive associations on retained
  support, not causal effects. Interval calculations treat reference stratum
  weights as fixed. Sparse-support exclusions must be read alongside the effects.
- The three populations are not assumed to be identical matched agents merely
  because they use the same random seed. Person IDs are not paired across
  populations. Survey replay is the paired comparison.
- The saved NNs were trained in separate runs with different survey income
  imputations. Shared replay inputs isolate the fitted-model predictions from
  input differences at evaluation time, but do not remove training-data
  differences. This is a comparison of fitted pipelines, not a controlled
  causal estimate of removing the target-distance feature alone.
- Existing survey records may have been used in model estimation. **The replay
  is not independent held-out validation.** It isolates observed-origin
  destination choice and cannot establish predictive generalization or reproduce
  sequential rollout error. A paper claiming generalization would need new
  person/household-separated fitting and testing outside this read-only analysis. The cross-fitted gate
  stages now save that separate person-held-out assessment in their validation artifacts.
- “Learned selector” describes the flag-false experiment accurately: target
  distance is excluded from neural inputs and selection, but it still affects
  training distance losses and stationary-trip handling.
  It is not an experiment with every influence of observed distance removed.
- Only one fitted run/seed per method is available. Person-cluster intervals
  describe sampling variation across people, **not** between-seed variability,
  training uncertainty or total model error. The report does not rank models
  with a survey goodness-of-fit score.
- Maps deliberately compare **shares**, not raw population-versus-survey point
  counts. The top row shares people, trip origins, income imputations and all other inputs; the bottom row
  uses each synthetic population’s own city-origin trips. All out-of-view mass
  stays in the denominator. The map caption states these different conditioning
  sets rather than suggesting the rows contain identical people.

For probabilistic gates, replay records both the sampled expert and its local
probability for each draw. Selection-diagnostic lines average fitted probabilities
when available; the CSV table counts actual weighted draws. A trip can therefore
contribute to both experts in that table. Historical deterministic outputs fall
back to averaging their recorded decisions.

## Distance-loss weight experiment

Run one sequential job for weights **0.2, 0.3 and 0.5**, followed by a comparison
against the completed **0.1** distance-enabled run:

```bash
bash "analysis/secondary locations/submit_loss_sweep.sh"
```

The default source is `config_dib_10_nn_distance.yml`. The submitter first pins
its completed assignment, model checkpoints and upstream artifact identities.
The original YAML, pipeline metadata, fitted 0.1 models and input caches remain
read only. A private snapshot is written under the ignored results directory;
it can contain private configuration values and must not be published.

Each weight retrains the existing regional, subregional, local and short-range
expert stages, in that order, and reruns secondary-location assignment. The
changed weight is `secondary_nn_distribution_loss_weight`, multiplying the sum
of origin-distance CRPS, home-distance CRPS and spatial energy. It applies to
all four experts in the distance-enabled experiment. Population synthesis,
primary locations, destinations, OSM, H3 and detour factors are read from the
same frozen completed run. The job does not build or simulate MATSim. The
no-distance experiment is unaffected.

The runner executes the normal stage `configure`/`execute` functions through a
restricted context which can only read the pinned upstream stages and results
of the preceding expert stages. This avoids invalidating unrelated caches when
other pipeline code changes. Stage completion records support restarting a
failed job: completed stages are reused after checking their output identities;
partial stages run again. Changes to frozen input files, model code or trial
settings cause a clear failure. One experiment directory cannot run twice
concurrently. Trial processes run sequentially and release memory between trials.

Slurm requests 24 CPUs, 11,947 MiB per CPU (approximately 280 GiB total), and 48
hours. Pipeline stages use 12 threads. Logs are written to
`logs/secondary_loss_weights_<job-id>.log`.

The final comparison is written to:

```text
analysis/secondary locations/results/loss_weight_sweep/report.html
```

It includes population ECDFs, short-distance detail, errors around the 1.5 km
expert switch, car/ÖV-class gradients, standardized behavioral contrasts, and
paired error improvements over weight 0.1 with person-cluster intervals.
Matched replay uses the same 1,000 city-origin survey people, three draws per
trip, and one input realization from the frozen distance-enabled run. The
baseline is replayed using its existing models; it is not retrained. Each
weight's population locations and replay predictions are retained. Four figures
are exported as PDF, SVG and 600-dpi PNG, with CSV statistics and JSON provenance.
The replay is an observed-origin fitting-data diagnostic, not independent
held-out validation. Compare physical errors and behavior rather than training
loss values, whose weights differ between trials.

To inspect or prepare the experiment without requesting a job:

```bash
uv run python "analysis/secondary locations/loss_sweep.py" --phase plan
```

Optional `--config`, `--output`, `--weights`, `--threads`, `--replay-people` and
`--draws` arguments can be passed to the submitter. Use a new output directory
when changing an existing experiment's settings. Resubmit the same command to
resume after a failure.

The completed experiment's [findings and short-range diagnosis](loss_weight_findings.md)
explain why 0.1 was retained. To reproduce the geographic choice-set check:

```bash
uv run python "analysis/secondary locations/diagnose_short_range.py"
```

This check reads the cached H3 representative points and common survey cohort;
its counts describe occupied cells before purpose filtering. It writes
`results/loss_weight_sweep/short_range_diagnosis.json` without editing run caches.

## Short-range-only hybrid trial

To change only the distance-enabled short expert, reuse the completed sweep:

```bash
bash "analysis/secondary locations/submit_short_loss_trial.sh" \
  --distribution-weight 0.05 --relative-weight 0.05 \
  --output "analysis/secondary locations/results/short_loss_0.05_0.05"
```

The pair means **(short distribution coefficient, short relative-distance Huber
coefficient)**. The hierarchy keeps its three saved 0.1 expert models. The runner
fits only the short expert and reruns its secondary assignment, then predicts the
same survey people and draws as the original study. All four reference
populations and replay-error tables are reused. The new report compares five
cases, with a separate settings table that distinguishes the short-only pair
from the earlier global 0.1/0.2/0.3/0.5 coefficients.

The job requests 24 CPUs and 11,947 MiB per CPU, uses 12 pipeline threads, and has
a 12-hour limit. It does not execute the full Synpp graph. Its plan records the
updated model source and checks the unchanged hierarchy stage identities and
frozen reference artifacts. The original sweep plan and checkpoints are kept
read only. The new coefficient pair and output directory must match a saved
plan when resuming. New short models and predictions remain in that experiment's
`trial/` folder; `report.html` and the five-case tables/figures are written at its
root. Figures include the fifth case and distinguish global versus short-only
coefficients in their legends. As before, the observed-origin replay is a
fitting-data diagnostic rather than independent held-out validation.

The corresponding pipeline controls are
`secondary_nn_short_distribution_loss_weight` and
`secondary_nn_short_relative_distance_loss_weight`. Both default to 0.05 for the
distance-enabled short expert. Neither adds a short-specific penalty to the
no-distance experiment; its distribution coefficient remains the shared value.

## Original loss with independent fold validation

The legacy-only short trial uses zero distribution scoring and the original
0.07 relative-distance coefficient:

```bash
bash "analysis/secondary locations/submit_short_loss_trial.sh" \
  --output "analysis/secondary locations/results/short_loss_legacy_cv5" \
  --distribution-weight 0 --relative-weight 0.07 --loss-kind relative \
  --validation-strategy cross_validation --min-epochs 20 --folds 5 \
  --reference-focused "analysis/secondary locations/results/short_loss_0.05_0.05"
```

For `relative`, the 0.07 coefficient multiplies the original Huber penalties on
expected distance from both origin and home. CRPS, spatial energy, and the
separate auxiliary origin penalty are disabled. Target distance remains an
input, the hierarchy remains its fitted 0.1 version, and routing still switches
experts at 1.5 km.

Five independent person-fold models validate after every epoch; each permanently
excludes its own validation people from training. They explore at least 20
epochs before stopping, then their weighted aggregate validation curve selects
the duration of a fresh full-data production refit. Earlier epochs remain
eligible if later exploration does not improve the score. This avoids the
leakage from rotating folds through one continuously trained network.

The report reuses the four global-weight references and the saved hybrid
(0.05, 0.05) case, adding the new legacy/CV case as a sixth model configuration.
It records the original-versus-auxiliary relative-loss scope, validation strategy,
minimum exploration period and actual selected/refit epochs, and includes the
five-fold learning curves. The new loss and training procedure change together;
its difference from earlier trials cannot be attributed to the loss alone.
Upstream feature transforms stay frozen, and the observed-origin replay remains
a fitting-data diagnostic, not independent held-out validation.

`--reference-focused` can be repeated for additional completed focused cases
with the identical common-input fingerprint. Their source files are pinned and
read only. Resume with the same arguments; changing loss kind, validation
settings or references requires a new output directory. No full Synpp graph is
executed by these focused jobs.

### Controlled comparison of short-expert losses with five-fold training

Run `bash "analysis/secondary locations/submit_cv_loss_sweep.sh"` to train the
short expert sequentially with proper-score weights 0.1, 0.2, 0.3 and 0.5, then
the (0.05, 0.05) hybrid. Each uses five independent person folds, validation
every epoch, at least 20 exploration epochs before early stopping, and a final
full-data refit for the selected duration. The completed legacy (0, 0.07) CV5
trial is reused. All three hierarchy experts remain the original 0.1 models,
which isolates changes to the short expert. Inputs, survey trips and draw seeds
remain frozen. This is a short-expert experiment, not a new global weight sweep.

The sequential job requests 24 CPUs and 11947M per CPU, uses 12 model threads,
and has a 24-hour time limit. Completed stages are resumable. Outputs go to
`analysis/secondary locations/results/short_loss_cv_sweep/report.html`.
The report includes six CV cases, the original fixed-validation 0.1 baseline,
training curves and duration selections, and paired before/after short-trip
errors with person-cluster intervals. Historical 0.2/0.3/0.5 runs used different
hierarchies; their before/after comparisons therefore cover only trips at or
below 1.5 km. The report also records endpoint errors and car/OVGK patterns.
Replay remains a fitting-record diagnostic; the independent folds select the
training duration but do not make this replay an independent test set.

Use `uv run python "analysis/secondary locations/cv_loss_sweep.py" --phase compare`
to regenerate the combined report from completed predictions, without training.

## Regenerate the three-way report

From the repository root, run:

```bash
"analysis/secondary locations/generate_report"
```

From `analysis/secondary locations`, run `./generate_report`. This runs the
existing analysis on the latest completed RDA, distance-input NN, and
distance-free NN assignments, and writes `results/report.html`. All normal
`run.py` arguments are forwarded. Use `./generate_report --slurm` inside this
folder to submit the analysis with its existing 4-CPU, 8G-per-CPU launcher.
To regenerate only after a successful pipeline job:

```bash
"analysis/secondary locations/generate_report" --slurm --after-job JOB_ID
```

The dependency must be a numeric Slurm job ID. The analysis never retrains models
or starts a simulation. Completed assignment metadata determines short-trip
sampling policy; editing the YAML after a run cannot silently change replay.

`diagnose_distance_short.py` records the distance-input short-trip investigation
using preserved pre-correction artifacts in `results/distance_short_investigation`.
It exports the completed training history, candidate-radius coverage, and a paired
counterfactual using distance-only facility sampling on identical observed origins,
purposes and draw seeds. Hierarchy predictions above 1.5 km remain identical in
that diagnostic. The counterfactual is not a sequential synthetic rollout or an
independent held-out evaluation.

The report dependency command also accepts several successful prerequisite jobs:

```bash
"analysis/secondary locations/generate_report" --slurm --after-job DISTANCE_JOB:NO_DISTANCE_JOB
```

Replace both placeholders with numeric job IDs. Slurm starts the analysis only
when both runs succeed. The completed NN assignment records
`secondary_nn_no_car_no_pt_weight`; report replay uses that value, while older
assignments default to their historical policy. The report notes that the soft
accessibility rule is an imposed prediction rule, rather than an independently
learned OVGK effect.
