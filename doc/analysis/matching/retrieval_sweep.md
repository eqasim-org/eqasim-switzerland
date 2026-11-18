# Fixed-encoder matching retrieval experiment

`analysis.matching.retrieval_experiment` reuses the three completed Current
encoders from the architecture ablation. It changes only neighbour count
(10, 20, 50, 100, 200) and sampling temperature (0.01, 0.1, 0.5, 2.0).
Each combination uses the same household splits and 50 donor draws.
The baseline is 20 neighbours and temperature 0.5. No neural model is refitted.
Settings minimise the mean shopping/leisure event Brier score on the selection
group; the test group reports probability, chain, distance and count outcomes.
This follows previous inspection of the test split: results are exploratory,
not a new independent confirmation. Brier assesses overall probability accuracy,
including both agreement and discrimination. Broader pools may trade behaviour
similarity for less extreme probabilities. Saved per-person scores checkpoint
matching calls and support report regeneration without rerunning matching.

Outputs: `<analysis_path>/matching/retrieval_sweep/report.html`, `summary.csv`,
`selected_test_summary.csv`, `probability_bins.csv`, configuration manifests and
per-candidate/per-seed compressed scores. The report averages prediction
probabilities across seeds before binning for its curve, whereas its score table
averages scores computed separately per seed. Full comparison coverage is saved.

```bash
sbatch --job-name=matching_retrieval analysis/matching/submit_ablation.slurm \
  --config config_dib_10.yml --retrieval
sbatch --dependency=afterok:EXPERIMENT_JOB --job-name=matching_retrieval_report \
  analysis/matching/submit_ablation.slurm --config config_dib_10.yml --retrieval --report
```

## Paper probability figure

`analysis.matching.probability_figure.generate` independently reconstructs diary
events from saved donor draws and verifies the source-trip fingerprint. Shared
bins retain boundaries 0, .1, .2, .3, .5, .75, 1. Upper-panel positions are actual
survey-weighted predicted means and observed frequencies. Vertical pointwise
95% intervals resample households within bins, holding the fitted encoder,
donor draws and bin membership fixed. Lower panels show weighted shares per bin,
not density; heights sum to 100% per method despite unequal bin widths.
The paper update is marked using a dedicated red macro and red figure border;
method colours are retained. It uses the original completed replay, independently
of the retrieval experiment. No synthesis or paper empirical results are replaced
with the new sweep's outcomes.

## Counts-only encoder at the selected retrieval settings

The `counts_retrieval` stage reuses both Current and Counts encoder checkpoints
from the completed architecture ablation, with no new neural fits. Both use
200 neighbours, temperature 0.1, three fitted encoders and 100 donor draws per
recipient, in both selection and test phases. The Counts architecture retains
the original feature reconstruction and adds the single five-output trip-count
head. Equal ensemble sizes allow comparison with the baseline at these settings.
The output is isolated in `matching/counts_retrieval_k200_t01/`; the larger
national replay/population experiment is unaffected. `counts_retrieval_report`
compares event Brier, count CRPS, chain Brier, joint energy and distance errors,
with paired household-bootstrap intervals and probability curves. The same
previously examined household split is reused: results are exploratory.

Launch `submit_ablation.slurm --counts-retrieval`, adding `--report` for the
separate dependent reporting job. Configuration is derived from the working
`config_dib_10.yml`; existing stage caches are reused in debug mode.
