# Investigation of the compressed-OSM secondary models

This audit uses the report generated on 14 November 2026 from training jobs
20885811 and 20885818, followed by report job 20885820. It does not modify the
assignment algorithms or retrain models. Report manifests identify the exact
cached inputs and checkpoints. Run the audit from the repository root:

```bash
uv run python "analysis/secondary locations/investigate_results.py"
```

The optional `--results` argument selects a report directory. CSV diagnostics
are written to its `investigation/` directory. Upstream caches are read only.
The script follows recorded dependencies for survey-chain targets rather than
selecting the latest available survey stage.

## Stationary trips have two distinct causes

Distance-enabled population: 36,243 of 1,165,183 trips are below 10 m (3.11%).
Of those, 26,559 have supplied target distances below 10 m and are explicitly
kept at the original location by `secondary_nn.locations`. This accounts for
2.28% of all trips, almost exactly the observed survey stationary share (2.28%).
The remaining 9,684 stationary trips have positive targets of at least 10 m;
they add 0.83 percentage points. The positive-target leisure-to-leisure group
contributes 3,403 of these, with a stationary rate of 6.58% in that group.
These numbers identify where to inspect destination reuse, but do not prove
that every near-zero trip reused the same facility ID.

The no-distance assignment has no target-based stationary shortcut, as intended.
Only 734 of 26,561 trips with survey targets below 10 m remain below 10 m;
their mean assigned distance is 4.05 km. This conditional comparison is a
diagnostic, not a proposal to reveal those targets to inference.

Priority: retain a legitimate stay/return alternative with probability learned
from non-distance inputs, such as origin purpose, activity duration, preceding
activity, and repeat-visit history. For distance-enabled trips, inspect positive
target facility sampling first. Do not remove all stationary assignments.

## Population transfer is harder than observed-origin replay

With distance input, mean absolute radius error for supplied population targets
of 10–100 m is 122 m; for 100–250 m it is 99 m; for 250–700 m it is 74 m.
Survey replay gives considerably smaller errors. Synthetic origins, facility
availability and chained prediction can therefore matter beyond fitted loss.
These cohorts are not identical, so this is evidence to investigate transfer,
not a causal estimate of rollout error.

At the expert boundary, population radius error rises from 80 m for targets
of 700–1500 m to 301 m for 1500–1600 m. Positive distance bias rises from 15 m
to 180 m. There is a real difference in precision between experts near the
switch. This does not establish that the threshold comparison itself is wrong.

The tail also deserves a separate feasibility audit: for population targets
above 20 km, mean target distance is 44.20 km and mean prediction is 35.60 km.
Inspect purpose-eligible domestic supply and origin displacement before simply
increasing the distance-loss weight for these trips.

The short loss uses cell representative coordinates and a 75 m allowance;
the final choice uses individual facility coordinates. At a 50 m target, that
allowance is large. Test facility-aware cell scores or a target-scaled allowance
before increasing every loss weight. Preserve the neural attractiveness model
and the origin-centred choice set. No hierarchy-proposed radius should enter
the distance-free short model.

## Routing is an important source of distance-free error

Across individual survey replay draws, 24.2% of weighted draws select the expert
opposite the observed <=1.5 km category. Those draws account for 37.4% of total
absolute distance error. Among observed-short draws sent to the hierarchy,
mean observed distance is 0.57 km and predicted distance is 6.79 km. Among
observed-long draws sent to the short expert, the corresponding means are
6.82 km and 0.76 km. The routing threshold is 1.6 km, while this diagnostic
uses the 1.5 km training cutoff: disagreement near the boundary is expected.
These are error descriptions, not an oracle estimate of gains obtainable by
changing experts; both experts can be wrong on the same trip.

Keep hierarchy-first routing. Improve the hierarchy's nearby probability mass
using origin-neighbourhood supply and destination-versus-origin attractiveness
features. Candidate dynamic inputs currently contain distances and detour,
but no explicit origin-neighbourhood OSM/offer density vector. Origin ÖVGK
is already present. Nearby supply summaries can be precomputed and compressed
without introducing a selector or rerunning spatial queries for every epoch.
Evaluate distance-band calibration of hierarchy predictions on reserved people.

## Training diagnostics do not show a general overfitting failure

Best fixed-holdout epochs for distance-enabled regional/subregional/local/short
experts are 49/32/31/36. For distance-free experts they are 50/49/48/42.
The distance-free regional expert improves from validation loss 2.454 at epoch
31 to 2.359 at epoch 50, with its best checkpoint at the maximum epoch.
Subregional validation also improves to almost the last epoch. Test a higher
maximum epoch limit with the existing minimum-epoch rotation and early stopping;
there is no evidence here that a larger network should be the first change.
Train/validation loss levels are affected by training dropout and different
cohorts and cannot alone diagnose model capacity.

The common replay contains just TWO trips whose persons were reserved from
gradient training in ALL four experts. Expert-specific validation sets are
drawn from different eligible populations. These people were also used for
early stopping, so even their intersection is not an untouched test set.
Establish a shared person-level test partition before fitting any expert,
scaler, or hyperparameter. Report results on that partition and keep the
existing replay maps as descriptive diagnostics.

## Accessibility and destination quality

The inference policy distinguishes any purpose-eligible A–D facility from none;
it does not grade A versus B versus C versus D. Large cells with a single
accessible facility already receive full support. This is a structural limit
of the policy, even though ordered home/origin/destination quality features
are supplied to the networks.

Test a graded, purpose-specific accessibility factor using the eligible
facility quality distribution, with explicit interactions for car access,
licence and PT subscription. Use a weighted distribution or shares, rather
than only the existence of one facility. Compare calibration separately for
A/B/C/D/none and driver/non-driver groups; preserve positive support.
Do not infer causal feature sensitivity from population cohort gradients:
run matched-input counterfactual predictions on the shared test partition.

Distance accuracy alone does not establish destination accuracy. For observed
700–1500 m survey trips, distance-enabled mean endpoint error is 946 m, even
though radius error is about 56 m. Stochastic alternatives need not reproduce
individual destinations, so endpoint error is not by itself a failure criterion.
Add held-out cell log likelihood and spatial distribution/visitation metrics,
alongside distance calibration, to assess learned attractiveness.

## Suggested experiment order

1. Establish the shared untouched test people and report per-draw as well as
   mean-of-draw errors. The existing paired report averages three distances
   before taking absolute error; audit CSVs instead score individual draws.
2. Extend maximum training epochs with early stopping, keeping architecture
   and loss fixed, to isolate incomplete convergence.
3. Audit facility sampling for positive-target stationary returns and make
   short-model distance scoring reflect facility geometry.
4. Add compact origin-neighbourhood supply and purpose-specific graded
   accessibility interactions; test each separately before combining them.

Use the same survey preprocessing, population, seeds and test partition across
ablations. Current fitted runs differ in training income imputations; their
comparison does not isolate the causal effect of distance input or compression.
