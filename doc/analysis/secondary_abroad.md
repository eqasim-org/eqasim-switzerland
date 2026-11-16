# Swiss residents' secondary activities abroad

`analysis.cross_border.secondary_abroad` reads completed prepared plans and the
exact upstream dependencies recorded in `pipeline.json`. It executes no Synpp
stages and treats the production caches as read-only.

```bash
.venv/bin/python -m analysis.cross_border.secondary_abroad \
  --config config_dib_10.yml \
  --output output/secondary_abroad_20261109
```

The latest completed preparation in the configured working directory is used.
Check `provenance.json` for the chosen file and cache key. Add `--previous` with
an exact `matsim.simulation.prepare__...` key for a previous snapshot. Each
snapshot uses its own sampling fraction; before/after differences are not a
controlled algorithm experiment when sampling or upstream inputs differ.

The analysis identifies Swiss residents by membership in that preparation's
sampled Swiss population, irrespective of nationality. It counts selected-plan
main activities, excluding interaction stages. Ordinary foreign secondary
activities and explicit `border` terminals remain separate. An explicit terminal
represents one crossing even when its coordinates fall inside Switzerland.
Ordinary crossing events require endpoints on opposite sides of the Swiss
polygon; incidental or repeated crossings along a route are outside this report.

Comparisons use residents aged 6+ and workday/weekend MZ person weights normalized
to the current prepared eligible population. Original MZ purposes and observed
trip origins are retained. A second activity comparison removes diaries spent
entirely abroad without changing the population denominator. National MZ all-mode
comparisons include long-distance travel and modes absent from the local road
controls; compare road modes and geographic scopes separately.

Explicit event purposes resolve through the raw survey pool plus crossing-specific
`swiss_facilities` donor aliases from count assignment. Missing explicit references
raise an error rather than silently dropping their purpose.

Cross-border survey controls come from the cached expanded reference pool and
the full-population event-assignment ledger. Road rows are incoming events;
equal entry and exit targets imply two events per reference row. The legacy PT
control is not doubled. The configured overlap region uses the production
journey-level exclusion rule with endpoint evidence. Ordinary prepared and MZ
journeys are classified by whether both actual endpoints lie inside this region.
Vehicle drivers and passenger-person events are reported separately.

Outputs include aggregate counts by purpose, canton, mode, geographic scope,
home distance and donor geography; survey targets and exclusions; exact cache
provenance; and validation totals. The 100 m border sensitivity identifies
near-boundary classifications. Distances are straight-line home-to-activity
distances, not routed journey distances. Home-border bands are recomputed with
the same polygon for both snapshots rather than comparing historical cached
distance definitions.

The output directory also holds extraction pickles containing individual plan
data for efficient reruns. Keep those local and ignored; distribute only the
aggregate reports. The dated 9 November 2026 output includes a snapshot-specific
HTML report and chart. Its renderer and interpretations are not automatically
reused for future runs.

Dated reports describe their own completed snapshots. Recompute the comparison
after changes to overlap exclusions, crossing assignment or destination calibration.
