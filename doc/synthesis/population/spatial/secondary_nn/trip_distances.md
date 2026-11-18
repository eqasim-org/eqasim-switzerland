# Loop and coincident-coordinate distance policy

`trip_distances.py` supplies one reference rule for survey-derived NN targets and
survey distance plots:

| Observation | Reference distance | NN destination assignment |
|---|---|---|
| Finite, distinct endpoints | Euclidean displacement | Predict normally |
| Coincident endpoints, `_loop` mode | Zero endpoint displacement | Preserve the current place |
| Coincident endpoints, other mode | Reported travelled distance / 1.3 | Predict normally |
| Missing endpoints, positive reported distance | Reported travelled distance / 1.3 | Use the fallback target |
| No usable geometry or reported distance | Unavailable | Raise a preparation error |

There is no one-meter stationary shortcut. Positive submeter coordinate distances
remain observations. Purpose equality does not identify a loop; the recorded
`_loop` mode and coincident endpoints do. The distance-free model receives the loop
flag as a diary rule but never receives the individual corrected target distance.
The recently introduced probabilistic stay model is removed.

All coincident/nonfinite coordinate records are excluded before preparing any
expert's training tensors, validation partitions or person scalers. Full diary
rows remain in `mz_chains`: their corrected references feed remaining-chain
budgets and retain trip identifiers and ordering. Unresolved loops contribute zero
displacement to these location budgets, not zero travelled distance to transport
statistics. Original reported distances remain available.

The microcensus fix is deliberately small: if an original home loop is split into
two legs, halve its `w_rdist` in each leg, conserving the original total. The
original stage-processing code did not retain stage distances, so equal allocation
is an approximation. Unsplit records are unchanged. No further division by two
is applied by the NN fallback. The fixed detour factor 1.3 is also an assumption,
not a measured Euclidean distance.

The report takes the corrected survey diary from the distance-enabled NN run as
its common reference, while retaining existing RDA population results. It stores
coordinate distance, reported distance, reference distance and source flags.
Predicted distances always come from actual assigned coordinates; the fallback
is never applied to predictions to hide their errors. Endpoint errors and observed
map destinations exclude unreliable reference coordinates; proxy references still
enter distance-distribution comparisons and are identified in the report caveats.

`scripts/submit_secondary_comparison.py` refreshes survey preparation in each NN
cache before the focused expert/assignment run. Keeping this separate from the
population graph prevents an upstream survey change from rebuilding the synthetic
population. It reuses cached spatial inputs, whose coordinates and trip IDs are
unchanged by the distance-allocation fix. Both NN jobs must succeed before the
comparison report starts. Feature version 3 requires full expert retraining.
