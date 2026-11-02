# Road-user map: activity purposes

The local road-user viewer shows actual vehicle passages and exported population
attributes. Clicking a directed link opens **Activity purposes on this link**.
Select a traveller group and either **Heading to** (destination activity) or
**Coming from** (origin activity). The table gives sample passage counts and
percentages within that group. All purpose categories are shown.

Groups reuse the map's existing classification: Swiss residents, external-France
residents, other cross-border demand, freight, LCV, Swiss cross-border and unknown.
These are population-source groups, not nationalities. Home/border activities
remain explicit rather than being reassigned to an inferred tour purpose.

## Counting and coverage

- Count actual `entered link` events in `0 <= time < 86400`. A vehicle passing
  twice contributes twice. Initial placements, transit vehicles and bicycles are
  excluded. This differs from the map's distinct-agent profile counts.
- Match an event journey to its selected-plan network leg by person, sequence,
  vehicle and mode. Ignore interaction activities when finding its origin and
  destination. An unmatched leg remains **Unknown**, included in the denominator.
- Destination purposes describe intended activities, including journeys that
  later abort or remain unfinished. They do not prove arrival at the activity.
- Percentages require no population scaling. Counts are unscaled sample passages;
  they need not equal the separately scaled daily-count CSV, whose midnight
  convention and vehicle coverage differ.
- The earlier station-corridor journey examples remain a separate subset. The
  new purpose table covers all indexed links, regardless of station membership.

## Updating an existing map

Run from the repository root, using the existing viewer's output directory:

```bash
uv run -m analysis.counts.road_users purposes --output /path/to/map-output
```

This reuses `persons.tsv`, `trips.tsv`, `person_features.npz` and the link index,
then streams the original events file once. It checks source file identities
against `manifest.json`, so replaced exports cannot silently be mixed with old
indexes. It does not run the population pipeline or simulation.

The command requires a C++17 compiler, Expat and zlib. `--compiler` and `--sdk`
allow an explicit local compiler/macOS SDK without changing system settings.
`--destination` can build the new artifacts in another directory for inspection.

`purposes.sqlite` stores sparse counts indexed by link. `purpose_validation.json`
records provenance and verifies passage totals against the existing event index
for **every link**. Publication happens only after validation. The viewer reads
this database on demand, without loading all purpose counts into the browser.
Restart an older viewer process after updating the Python code, then refresh the
browser. Existing road filters and overlapping-link selection are unchanged.
