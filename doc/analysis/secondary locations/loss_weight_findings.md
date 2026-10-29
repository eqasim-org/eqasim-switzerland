# Distribution-loss weights and the remaining short-trip bias

The experiment completed in Slurm job 17988449. Retain **0.1** as the shared
distribution-loss default. The table reports weighted mean absolute distance
error per replay draw, using identical inputs and draws across fitted models:

| Distribution weight | Overall distance MAE | MAE for observed trips of 10 m–1.5 km | Short-range checkpoint epoch |
|---|---:|---:|---:|
| **0.1** | **929 m** | **125 m** | 8 |
| 0.2 | 1,011 m | 303 m | 2 |
| 0.3 | 1,021 m | 281 m | 2 |
| 0.5 | 955 m | 157 m | 4 |

The paired increase in short-trip MAE relative to 0.1 was 178 m at 0.2,
156 m at 0.3 and 32 m at 0.5. Their approximate person-cluster 95% intervals
were 163–194 m, 142–171 m and 25–39 m, respectively. Overall uncertainty is
larger, especially for long trips; this is a working-default choice from one
seed and an observed-origin diagnostic, not an independent generalization claim.
Weight 0.2 gave a marginally larger population mean, but mean matching alone
would conceal the worse conditional errors.

At weight 0.1, the 1–1.5 km replay band still averaged 1,003 m against an observed
1,231 m. Immediately above the expert switch, the 1.5–2 km band averaged
1,873 m against 1,736 m. The routing rule remains target distance <=1,500 m
for the short-range expert; longer trips use the hierarchy.

## What explains the remaining problem

- **Training and prediction compare different alternatives.** Short-range
  training uses the occupied leaves in the observed destination's middle H3
  branch. Prediction searches the origin's neighborhood, initially within
  1,800 m, expanding when eligible supply is sparse. Among 1,027 short replay
  trips with valid training branch labels, every origin neighborhood included
  cells outside its training branch. The median outside fraction was 59.1%;
  the median search contained 548 occupied cells, versus 241 in the observed
  branch. These are geographic counts before purpose filtering, not counts of
  the final eligible company pools. They demonstrate a structural mismatch;
  a controlled retraining with aligned alternatives is needed to establish its
  contribution to predictive error.
- **A correct-radius facility must be in the selected cell.** Final Gaussian
  distance weighting operates inside that cell. It cannot repair a cell whose
  entire facility pool lies at the wrong radius. Fine-cell representative points
  also differ from actual facilities, which matters proportionally more for
  very short trips.
- **Increasing the distribution weight changes a combined objective.** Origin
  CRPS, home-distance CRPS and spatial energy all receive the larger coefficient.
  The 0.2/0.3 short-range models selected epoch 2 rather than epoch 8. Their
  poorer distance errors show that stronger combined scoring did not solve the
  problem; the stopping histories alone do not identify its cause.

## Targeted auxiliary loss

Keep distribution scoring at 0.1 and use a short-specific **0.05** distribution coefficient and a **0.05** relative
expected-trip-distance Huber coefficient for the distance-enabled short expert:

```text
L_short = cross_entropy
        + 0.05 * (CRPS_origin + CRPS_home + spatial_energy) / 1000 m
        + 0.05 * Huber((E[predicted_trip_distance] - supplied_trip_distance)
                       / (supplied_trip_distance + 100 m))
```

The option is `secondary_nn_short_relative_distance_loss_weight`; zero disables
the addition. It targets origin-to-destination distance only, and does not repeat
the home-distance term. It applies in training and validation, including the
stopping criterion, with the same survey weights and purpose masks. Hierarchical
experts, the distance-free experts/selector and the legacy `relative` objective
receive no auxiliary term.

The focused trial uses the requested (0.05, 0.05) pair; its three hierarchy
experts retain their fitted 0.1 weights. The earlier 0.02 auxiliary proposal was
not a result from the completed sweep. The completed weight sweep did **not** evaluate
this new combined loss. It requires retraining and a fresh comparison before
claiming improvement. It also does not remove the choice-set mismatch. Aligning
the origin-based alternatives in training and prediction is the next structural
correction if the local bias persists.

The generated evidence is under
`analysis/secondary locations/results/loss_weight_sweep/`: the paired-improvement
and error-band CSVs, fitted-stage histories, and `short_range_diagnosis.json`.
