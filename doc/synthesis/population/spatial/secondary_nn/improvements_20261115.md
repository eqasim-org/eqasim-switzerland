# Secondary model improvements, November 2026

All four experts reserve the same deterministic 10% of survey people before
fitting person scalers or model weights. The partition uses the configured seed
and person identifier, independently of expert eligibility. Existing rotating
validation and early stopping operate within the development population. The
maximum increases to 120 epochs; the minimum remains 30.

Four origin inputs summarize shop supply, leisure supply, OSM context and total
destination supply within 1.5 km. Summaries are cached on a 200 m grid. Each OSM
quantity is log transformed before aggregation, avoiding direct addition of areas,
lengths and counts. The regional person scaler saves the lookup; all experts use
it during training and prediction. Existing destination OSM compression remains.

Accessibility weights grade the mean purpose-eligible facility quality through
the coordinate hierarchy. The A/B/C/D/none scores are exp(-0.65*quality), affinely
mapped onto the configured positive floor and one. Final facility sampling uses
the same grading. Weight exponents are 0.15 for licensed drivers with a car,
1.25 for non-drivers with a PT subscription and 1 otherwise. These policy
strengths are modeling assumptions and should be assessed on the reserved test
people; they are not estimated behavioral coefficients.

For distance-enabled short trips, radius scoring uses actual facilities inside
each candidate cell and the target-centered kernel used in sampling. Individual
facility penalties prevent opposite errors from cancelling. The distance allowance
is min(configured tolerance, max(1 m, 0.1*target)). Log-space normalization preserves
radial preference in sparse pools. The distance-free short expert receives no
radius or direction preference from the hierarchy.

The stationary and distance rules have subsequently been replaced by the
[explicit loop and coincident-coordinate policy](trip_distances.md). There is no
one-meter shortcut or learned stay model in the current pipeline.

The report adds shared-test-person tables for errors of mean repeated distances
and individual-draw radius/endpoint errors. Existing maps and population summaries
remain descriptive. Report stationary shares now count zero displacement; routing diagnostics exclude zero-displacement loop references.

Run both focused expert-training/assignment jobs and their dependent report:

```bash
uv run python scripts/submit_secondary_comparison.py
```

The submitter copies `config_dib_10_sNN.yml` and `config_dib_10_sNNnD.yml` into an
ignored log directory, retrains all four experts, reuses upstream data and schedules
the RDA comparison report after both jobs succeed. Slurm uses `--mem-per-cpu`.
Feature schema version 3 prevents short-only runs from mixing incompatible experts.
