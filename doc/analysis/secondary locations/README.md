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
