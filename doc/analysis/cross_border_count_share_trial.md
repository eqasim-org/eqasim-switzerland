# Group O: observed-count share experiment

This is an offline experiment in `analysis.cross_border.count_share_trial`, not a
production stage or a new assignment default. It never rewrites source inputs,
production caches or population assignments.

The 5 November 2026 experiment uses the completed foreign demand and Swiss
cross-border model, including actual Swiss home coordinates and assigned travel
directions. Its inputs include the current 1.5 road eligibility run and the
latest importance edits. The Swiss reference pool restores foreign endpoint
coordinates from its survey audit by donor ID; it is not counted as additional
population. Observed counts are the existing nine local Group O references.

## Allocation

1. Retain Group O demand, residence categories, purposes and all OD endpoints.
   Normalise the observed crossing counts to shares of the modelled crossing
   total; do not inflate the cross-border population to all-vehicle road counts.
2. Reserve known transit exits, including exits of people interviewed elsewhere.
   Preserve entries of transit people whose other exit cannot be located.
3. Compute one-to-many directed free-flow shortest paths from each crossing and
   in the reversed graph for opposite-direction travel. The graph is restricted
   to Swiss roads. The approximation ignores turn restrictions and congestion.
4. Jointly allocate all movable categories to the crossing shares. Candidate
   support can limit added Swiss travel time versus the fastest Group O crossing.
   For returning foreign visitors, check inward and outward legs separately.
5. Prefer short Swiss road travel and a short foreign approach. The latter uses
   straight-line distance at 60 km/h, not foreign road routing. A five-minute
   entropy/smoothing scale produces probabilities rather than deterministic ties.

`allocate` solves the column-price dual with only one variable per crossing.
With an unbounded price it fits feasible margins; bounded prices implement soft
count targets. Every OD retains its mass. This is a joint constraint across
categories, not an observed category-by-crossing target (such counts are absent).

`feasible` first aggregates identical candidate masks. With nine crossings there
are at most 511 nonempty patterns, avoiding a degenerate LP with thousands of
identical constraints. `integer_draw` vectorises singleton choices and uses
multinomial draws for repeated OD groups. Foreign visitor entry/return events
stay together as one car tour.

## Findings and limits

On this snapshot, the 1.5 straight-line constraint cannot reproduce the observed
shares. A 15-minute extra Swiss-road-time limit is also infeasible; 20 minutes is
feasible. Count shares plus the 20-minute Swiss-side limit reduce mean Swiss
road time from about 29.6 to 27.8 minutes while matching the selected expected
margins. The foreign approach proxy becomes longer, so this is not a uniform
improvement in entire-journey geography.

These are calibration results, not validation against independent traffic data.
References mix years, weekday/all-day periods, counter offsets and all-vehicle
coverage. Ordinary Swiss/French traffic and freight are absent. Unlocated transit
exits outside this reference cannot be invented or allocated to Group O. Literal
observed traffic counts should therefore not become hard cross-border totals.

No production implementation has been adopted. A candidate implementation should
preserve surveyed individual crossings where known, pair car passengers with
their vehicle's choice, keep PT/freight separate, report infeasibility, and inspect
foreign approaches and repeated crossings after routing. A subsequent population
sample can disturb site margins even when full-population expectations fit.

## Replaying the snapshot

The dated analysis output contains the report, aggregate comparisons and an
`inputs/` snapshot. That input snapshot contains internal geographic data; keep it
with the analysis data rather than committing it to Git.

```bash
.venv/bin/python -m analysis.cross_border.count_share_trial \
  --input /path/to/cross_border_count_share_trial_20261105/inputs \
  --output /path/to/new_trial_results
```

Inputs: `od.csv` (grouped demand), `sites.csv`, `references.json`,
`baseline_allocation.npy`, `air_costs.npz`, `road_costs.npz`, `road_timings.json`,
`extraction.json`. The script writes scenario matrices, crossing/category/time
comparisons and diagnostics. Archived preparation/report scripts document this
specific snapshot; they are not reusable production stages.

Local regression tests cover margin/OD preservation, forbidden alternatives,
infeasibility, whole-tour drawing and seeded reproducibility. The experiment also
checked category and direction totals, fixed transit events, twenty integer draws,
and reference/smoothing sensitivities. No full population build or MATSim run was
launched.

## Production successor

The approved method is now implemented in
`data.cross_border.count_assignment`; see the
[count-guided assignment guide](../data/cross_border/README.md#count-guided-physical-crossing-shares).
The trial runner remains an offline comparison tool. The production version
additionally excludes a transit's reported exit before measuring the fastest
valid entry, integrates Swiss facility aliases, and retains unsupported groups
with explicit diagnostics. Its seeded integer results need not match the trial
draw exactly.
