# Matching architecture ablation

`analysis.matching.ablation_experiment` compares Current, Counts, Features, and
Counts + features, without modifying the synthesis package or paper. The model,
loss, store and trainer adaptations live exclusively in `analysis/matching/`.
`ablation_training.py` is an analysis-local adaptation of the production trainer;
it retains its optimiser, internal validation, minimum ten epochs and stopping.

Counts uses one head with five outputs: total, shopping, leisure, education and
work trip counts. Targets cover full diaries, including secondary purposes and
zeros, independently of sequence truncation. Each target uses the existing
log-count scaling. The count objective is the total-trip error plus the mean
of the four purpose errors, multiplied by the existing trip-count weight.
Features replaces projected-input reconstruction with direct reconstruction of
selected mobility covariates. Numerical features use smooth-L1 and categorical
features use cross-entropy; valid attribute losses are averaged per person.
Missing/unseen categorical values and missing numerical measurements are masked.
This changes the reconstruction target and loss formulation as one experimental
package; the reconstruction weight remains unchanged.

All four variants use the configured original behavioural weights, not the
weights selected in the previous search. Three shared fitting seeds test whether
an apparent improvement depends on initialization. Household-disjoint groups
are approximately 60% fit, 20% selection and 20% test, using a new fixed split.
Selection-stage models train on 60%; test-stage models on the 80% development
group. Every fit reserves its usual internal 10% validation group. Fifty donor
draws score each recipient. Selection energy chooses the preferred variant;
test scores evaluate all four. Follow-up design was informed by earlier results:
this is not an external validation cohort. No automatic production integration
occurs. This experiment evaluates survey matching, not population forecasts.

The report has eight outcomes, individual fitting-seed points and paired
household-bootstrap differences against Current. Bootstrap intervals condition
on the fitted models and split, average across seeds, and have no multiple-test
adjustment. Outputs are under `<analysis_path>/matching/architecture_ablation/`.
Each fit has a fingerprinted checkpoint, donor rows and training history.

```bash
sbatch analysis/matching/submit_ablation.slurm --config config_dib_10.yml
sbatch --dependency=afterok:EXPERIMENT_JOB --job-name=matching_ablation_report \
  analysis/matching/submit_ablation.slurm --config config_dib_10.yml --report
```

The report-only job loads saved scores. Dependencies are verified in the same
project environment used for execution. The default is 24 small neural fits
(four variants, three seeds, two phases); it does not retrain upstream stages
or run transport simulation.

## Doubled-count follow-up

`ablation_double_counts` fits only Counts + features with count weight 3.0
instead of 1.5. It preserves every other option, the three seeds, household split
and 50 draws. It checks saved membership, seeds and weights before appending the
four previous controls to a separate `architecture_ablation_double_counts`
output. `ablation_double_report` renders all five variants. This is exploratory
reuse of a previously examined test split. Launch the same Slurm script with
`--double-counts`, and add `--report` for the dependent report job.
