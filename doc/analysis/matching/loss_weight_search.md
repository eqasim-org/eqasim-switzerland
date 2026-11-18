# Behavioural loss-weight search

The analysis-only stages `analysis.matching.optimize_weights` and
`analysis.matching.optimized_report` leave the synthesis implementation and paper
unchanged. The search uses existing prepared input caches from `config_dib_10.yml`.

The seven behavioural weights vary independently between 0.25 and 4 times their
current configured values, on a log scale. Twenty candidates include the current
weights plus 19 seeded Latin-hypercube candidates (SciPy; no new dependency).
Other model and regularisation settings stay fixed. This is a finite search,
not a guarantee of globally optimal weights.

Households are split approximately 60/20/20 into fitting, selection and independent
testing groups. Trial models fit only the first group, retaining their usual
internal 10% early-stopping validation and minimum ten epochs. Selection minimises
survey-weighted multivariate energy score with 50 donor draws. All candidates
must achieve complete matching coverage; outcome scales, seeds and donor pools
are held constant. Weighted validation losses cannot be compared across weights.

After selection, the current and selected BSLM models are refitted on the 80%
development group and compared against RHDM and OHDM on untouched test households,
with the same donors and 50 draws. Charts use common matched recipients; coverage
is saved explicitly. Test outcomes never select weights. The selected weights
are then refitted on all eligible survey people for a separate three-group,
50-draw donor-reconstruction experiment and ten synthetic-population assignments.
That application retains the established report's limitations: reconstruction is
not independent testing, and population distributions are descriptive sensitivity
results, not known population truth.

Outputs are isolated under `<analysis_path>/matching/loss_weight_search/`:
`trials.csv`, `selection.json`, `membership.csv`, `test_scores.csv`, model checkpoints,
`report.html`, and `application/matching/repeated_shared_v1_v2_v3/report.html`.
The original reports and upstream caches are preserved. Trial model checkpoints
can be reused after interruption. Keep the configuration unchanged when resuming;
use a new analysis path for a different search.

Run from the repository root:

```bash
sbatch analysis/matching/submit_weight_search.slurm --config config_dib_10.yml
sbatch --dependency=afterok:SEARCH_JOB_ID --job-name=matching_loss_report \
  analysis/matching/submit_weight_search.slurm --config config_dib_10.yml --report
```

Both jobs use the project `.venv` through `uv run --no-sync`, verify dependencies
at startup, request eight CPUs and 64 GB, and reuse upstream caches in debug mode.
The report job loads the cached optimization result rather than selecting weights
again. The two stage options `matching_hpo_trials` (20) and `matching_hpo_draws`
(50) can be changed in a separate configuration. Standard repeated-experiment
options control application repetitions (current defaults 100 survey / 20 population).
