# Shared diary-matching workflow

V1, V2 and V3 now use
[`workflow.py`](../../../../synthesis/population/matching/workflow.py) for input
preparation, recipient eligibility and final output assembly. Donor-search
algorithms, random seeds, sampling rules and V2's fitted model are unchanged.

Each stage follows the same sequence:

```python
prepared = workflow.preprocess(context, "v1")  # or "v2" / "v3"
# Run this algorithm's donor search using prepared.source / prepared.population.
mapping, removed = ...
return workflow.postprocess(prepared, mapping, removed)
```

## Loading and shared rules

`load_population` copies the full sampled roster and checks unique person IDs.
`load_survey` retains the existing cleaning of inconsistent primary-work diaries
and usual commute preparation. `survey_for_day` applies the same domestic/day
selection, interpreting boolean strings explicitly rather than using Python
truthiness. Genuine non-travel survey days remain donors.

`eligible_mask` is the single definition of recipient eligibility:

- STATPOP: age at least `MZ_AGE_THRESHOLD` (currently six).
- ARE: a positive age class.
- Both: exclude `home_only_mask`, the explicit home-only scenario bypass.

Those bypass residents remain in the returned roster with `mz_person_id = -1`.
Collective-residence occupants who are actually matched to home-only donors are
eligible recipients; the algorithm's existing donor restrictions still apply.
Preprocessing checks that every adapter preserves exactly the common eligible
ID set. Population frames are not filtered prematurely: V2 needs full household
membership when recovering child and adult counts.

## Algorithm-specific preparation in the same module

- `prepare_v1` retains recursive hot-deck categories, clipping, distance bins and
  type conversions.
- `prepare_v2` retains exact age, ordered income, household counts, continuous
  density/distance features, missing-value semantics and observed diary targets.
- `prepare_v3` retains its existing categories and encodings for the profile
  search. Profile definitions and hard constraints stay in `V3.inputs`.

Residence opening-hours regions are attached by `preprocess` after preparation.
The compatibility facades `V2.inputs.prepare_inputs` and
`V3.inputs.prepare_inputs` delegate to this module without region attachment,
preserving their previous contract for training and analysis callers that attach
regions explicitly. The public `matched_v1.get_mz_persons` helper delegates to
`load_survey` so existing callers retain the same cleaning implementation.

## Output handling

`initial_mapping` creates a consistent placeholder mapping. `postprocess` checks
IDs, preserves original roster order and household IDs, restores explicit
home-only bypasses, normalises donor IDs and returns `(mapping, removed_ids)`.
Below-age residents retain the no-donor placeholder unless their household is
removed under the shared collective-residence failure rule.

For all three algorithms, an unmatched eligible collective-residence occupant
removes their entire household when a household ID is available. This includes
already-matched members, below-age children and home-only bypass members in that
household. Explicit whole-household removal takes precedence over restoring an
individual bypass. Ordinary unmatched residents are removed individually. The
algorithms report their failed recipients; household expansion happens only in
`postprocess`. Without a household identifier, the failed person is removed
without guessing which other residents belong with them.

All omitted active residents are reported in `removed_ids`, as are eligible
recipients without a valid donor. This closes a previous V1 accounting gap where
an inner merge could omit a resident without including their ID in the reported
removal set. Algorithms may still fail on different people because their donor searches
differ; a common starting cohort does not force a method to invent a donor.

Changes to this module invalidate all three matching-stage implementation tokens.
No configuration options or upstream stage contracts were added. Retrieval
settings are unchanged by this refactor. Upstream cache frames are never edited.

## Extending the workflow

Add a population-wide eligibility rule to `eligible_mask` or shared survey
cleaning to `load_survey`/`survey_for_day`. Add feature conversions only to the
appropriate preparation adapter. New preparation must retain the eligible ID
set; an assertion catches accidental population changes. Keep donor-search
objectives and sampling inside their existing algorithm modules. Run the local
workflow regression checks and affected dependency graphs after changes.

V2 diagnostics report both `unmatched_persons` (failed donor searches) and
`removed_persons` (all residents removed, including household members).
