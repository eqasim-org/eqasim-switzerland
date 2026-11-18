# Expanded matching replay and population comparison

V2 retrieval defaults are now 200 neighbours and sampling temperature 0.1.
The encoder architecture and behavioural-loss weights are unchanged. The
repeated experiment defaults are 100 Microcensus diary draws per respondent
and 20 full-population assignments per method, comparing RHDM, BSLM and OHDM.
Three household donor-exclusion groups are retained, with one shared V2 encoder
and its internal 90/10 fit/early-stopping split. This remains reconstruction
under donor exclusion, not an independent encoder generalisation experiment.
More draws reduce donor-sampling noise; they do not remove survey uncertainty.

The expanded launcher explicitly pins the retrieval settings and repetition
counts, writes a temporary configuration derived from `config_dib_10.yml`, and
forces execution of the shared model stage and repeated experiment. Input stages
reuse existing caches in debug mode. The temporary config is removed once Synpp
has configured the pipeline. Upstream datasets are not rebuilt. Packages are
checked in the same project environment used by the job.

Outputs use a separate directory:
`<analysis_path>/matching/repeated_k200_t01_mz100_pop20/`.
Existing reports and the paper are preserved. `manifest.json` records actual
retrieval options, draw counts and training history. Per-draw checkpoints allow
resuming. Ensemble scoring is batched to bound quadratic pair-distance memory.

```bash
sbatch analysis/matching/submit_expanded.slurm --config config_dib_10.yml
sbatch --dependency=afterok:EXPERIMENT_JOB --job-name=matching_expanded_report \
  analysis/matching/submit_expanded.slurm --config config_dib_10.yml --report
```

The first job saves assignments/scores without plotting. The dependent report
job reads the completed experiment and creates `report.html`, including replay,
national/canton and residential-density population comparisons. No matching or
training is done by report generation when the completed stage cache is present.
