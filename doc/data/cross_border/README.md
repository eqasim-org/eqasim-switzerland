# Road through-trips and survey crossings

The 2021 A+GQPV road-border survey interviews travellers **entering Switzerland**.
The blank `FAHRTRICHTUNGGU` field does not mean unknown road direction: that field
describes entry/exit for rail interviews only. See the BFS
[2023 road-transport methods report](https://dam-api.bfs.admin.ch/hub/api/dam/assets/29270695/master#page=10),
page 10, footnote 5. This rule applies to road **border** interviews, not Alpine
interviews.

## Inputs and stages

`data.cross_border.generate_od` retains the original interview ID and the
`GUAUSREISESTRASSE` reported exit code for through-trips. It decodes exit names
from `crossborder/AuGQPV_2021/AGQPV21_finale_Auswertungsdatenbank_Codebuch.xlsx`,
which accompanies the survey CSV. Blank codes and `9999` (don't know) are missing
exits. If the codebook is unavailable, codes remain available and a warning
explains why they cannot be mapped by name.

`data.cross_border.through_crossings` runs on the sampled road through-population,
before destination facilities are created. It uses the final
`matsim.scenario.network.pt` car network and the same directional link matcher as
MATSim preparation. It keeps the surveyed crossing as entry for every road
through-agent. The former network-cost and straight-line entry/exit direction
selector has been removed.

`data.cross_border.road_exit_points` first keeps the existing exact-name matches
against road locations in `crossborder/border_interview_places_road_and_rail.gpkg`,
ignoring only accents, punctuation, spacing and case. It can also read the optional,
locally collected `crossborder/crossings_1.gpkg`. That file has individual names in
`candidate_label`, a survey area in `interview_place`, and transport modes in
`modes`; it does **not** contain the survey exit codes.

Explicit candidate aliases in `road_exit_points.EXIT_LABELS` connect the official
2021 codebook names to those collected names. An alias is used only when:

- Its official codebook name still agrees with the reviewed name for that code.
- Exactly one car-eligible collected record has the individual label. Rail rows
  are excluded before matching.
- Its coordinate is not shared by a differently named road candidate.
- The coordinate is within 500 m of the national boundary, and exactly one
  canonical road point in its interview group is within 500 m.
- There is no explicit unresolved identity issue, such as the supplied
  Schaanwald label describing FL–Austria while its point is on the Swiss border.

The 500 m check uses the first search radius of the existing directional link
matcher; it was not fitted to survey counts. The selected canonical point keeps
its original ID and geometry. Entry sampling and its crossing weights are not
changed. These are locally checked aliases, not an official geocoded crosswalk.
A group of smaller crossings is never treated as a single crossing location.

The reference stage writes `road_exit_reference.csv` with each code, its candidate
label, acceptance/rejection reason, nearest-point distance and chosen canonical
ID. Missing, ambiguous or rejected codes remain unmapped for route inference.
Without `crossings_1.gpkg`, local aliases are unavailable; exact-name matches and
the official supplement below can still be used. Changes to the optional inputs
or the codebook invalidate the reference cache.

### Versioned local inputs

`cross_border_reference_path` selects the crossing-reference directory for both
`data.cross_border.interview_places` and `data.cross_border.road_exit_points`.
Its default is `null`, which retains the original `data_path/crossborder/`
layout and its optional exit supplements. A relative path is resolved against
`data_path`; absolute paths are also supported. For the reviewed release:

```yaml
config:
    cross_border_reference_path: crossborder/references/2021_reviewed_20261023
```

An explicitly selected directory must contain all four runtime inputs below.
A missing file raises an error instead of silently reverting to legacy inputs.
The files remain in the data directory, outside Git.

| File | Role in the reviewed release |
| --- | --- |
| `border_interview_places_road_and_rail.gpkg` | Unmodified copy of the shared entry-sampling layer. Preserve row order, coordinates, labels and importance weights: row order determines canonical `BCP_` IDs. |
| `crossings_1.gpkg` | Unmodified copy of the original collected labels, used only as a conservative fallback. Known contradictory aliases remain blocked in code; official matches take precedence. |
| `road_exit_points_official.gpkg` | Corrected smaller-exit identification using dated official locations, survey-code associations and source identities. |
| `road_exit_points_reviewed.json` | Source-bound approval of the 12 visually checked office coordinates, including the declared boundary scope for Liechtenstein. |

The accompanying `road_exit_points_official.sources.json` and `.matching.csv`
retain collection provenance and the name-match audit. `reference_manifest.json`
records the release name, file hashes, source roles and unchanged survey inputs;
it is for reproducibility, not an additional model input. Runtime cache validation
tracks the selected entry layer and exit inputs, plus the original survey codebook.

The survey CSVs and codebooks stay under `crossborder/AuGQPV_2021/` and
`crossborder/AuGQPV_2015/`. They are not copied into the reference release or
modified by this selection. The `2021` in the release name is the source edition;
`20261023` is the review date, not a new survey year.

For a future update, create a new release directory, preserve the entry layer
and fallback layer unless separately audited, and collect the official input
with `--output` pointing into that directory. Copy the collector's provenance
and audit alongside it. Recheck any changed official identities/coordinates before
creating the matching review JSON; an old review must not be transferred to a
changed office. Select the new directory in the configuration after validating
its exit mappings and outward network links. Keep the previous release available
for reproducibility and rollback.

### Official online reference

The optional `crossborder/road_exit_points_official.gpkg` adds a dated official
reference. Its `road_exits` layer associates the 2021 survey names with individual
locations from **swisstopo swissTLMRegio 2021, CustomsOffice**. Its
`source_customs_offices` layer retains all 134 original office records. The
[official product page](https://www.swisstopo.admin.ch/en/landscape-model-swisstlmregio)
documents its generalised geometry (20–60 m positional accuracy); the
[2021 catalogue item](https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swisstlmregio/items/swisstlmregio_2021)
provides the dated archive and checksum.

The runtime stage reads this local file only; it never calls an online service.
It preserves the original major-crossing matches and gives official office
locations priority over collected smaller-crossing aliases. An official record
must retain its survey name, source UUID, reference year and URL. Its name must
agree with the current codebook. The nearest canonical road point must be within
500 m, with no exact-distance tie, and the office must be within 500 m of the
Swiss national boundary. All original IDs and canonical geometries are retained.
Unlike an unverified collected label, an independently named official office can
identify a site among several nearby canonical lane points. The audit records
the nearest and second-nearest distances and source coordinates.

An official office can be inland or on the FL–Austria customs border, so its
presence in the source does not establish a Swiss national-road crossing. Such
records remain unresolved under these checks. Do not increase the threshold
solely to include them. Known contradictory local aliases are blocked even when
the optional official file is absent.

To reproduce or update the input, run this **input-preparation utility**, using
the pipeline data directory from your configuration:

```sh
uv run -m data.cross_border.collect_road_exit_points \
    --data-path "/path/to/pipeline" --reference-year 2021
```

This downloads the dated official archive, verifies its published SHA-256
checksum, extracts only the CustomsOffice shapefile, and matches complete names
after stripping parenthesised customs numbers. Explicitly reviewed alternative
names are listed in `collect_road_exit_points.OFFICE_ALIASES` and checked against
the expected codebook names. There is no fuzzy name matching or address geocoding.
For example, `Ramsen-Dorf` is not silently assigned to `Ramsen`.

The utility writes the GeoPackage, a `.sources.json` manifest and a `.matching.csv`
audit into the data directory. It keeps the large download temporary. Use
`--archive` to reuse a downloaded shapefile ZIP; it still verifies the official
catalogue checksum. Use `--output` for a proposed replacement: existing files
are never overwritten. Inspect the resulting mappings before replacing the
active input. The default layout needs no YAML change; select a complete versioned
directory with `cross_border_reference_path` as above. Share the inputs with other
pipeline users alongside the code; they are not stored in Git.

### Visually reviewed offices

`crossborder/road_exit_points_reviewed.json` records the user's visual confirmation
of the 12 yellow office locations (13 survey codes) on 23 October 2026. Each
review binds a source UUID, name, edition, coordinates and list of survey codes
to a reviewer/date and `border_scope`. A changed official identity, edition or
coordinate requires a new review; old confirmation is not silently transferred.
This optional input is stored beside the GeoPackage and should be shared with it.

Reviewed offices get stable `BCP_OFFICIAL_<source UUID>` exit-only IDs at their
actual official coordinates. They do not enter the interview sampling layer or
change its weights. The 500 m guard remains for unreviewed records. Every reviewed
exit must have an outward car link in the current network, checked separately
from the confirmation of its coordinate.

For Schaanwald, Ruggell and Mauren, `border_scope: swiss_liechtenstein` uses the
outer Switzerland-plus-Liechtenstein boundary to select the LI–Austria exit link.
Other crossing matching and Swiss national-exit diagnostics retain the Swiss
polygon. `reported_exit_is_final` is undefined for these customs exits because
they are distinct from the last Swiss national-border crossing.

The selected links are carried by `data.cross_border.destinations` as
`reviewed_exit_link_id`. MATSim preparation validates that those car links still
exist and preserves them when patching both facilities and population XML.
It raises an error if a checked link is absent instead of snapping an inland
office, tunnel or LI–Austria location onto a different crossing. Counts include
the new exit IDs, so their recorded border activities are retained.

## Foreign destination and route

The original foreign destination determines the destination side of the network.
For a destination beyond `border_offset` (default 50,000 m), the destination-to-exit
line intersects the outer Swiss-border buffer. Where an exit cannot be mapped,
the line instead uses the known entry as its anchor. Nearby foreign road nodes
are considered using the existing endpoint and motorway candidate limits. For
closer destinations, candidates are near the original coordinate. All candidate
nodes must be outside Switzerland.

Candidates are evaluated in the travel direction, from the inside end of the
known entry link towards the foreign endpoint. A mapped reported exit constrains
the route through its outgoing link. For distant destinations, nodes nearest the
original destination and its motorway access are also considered. The existing
`data.cross_border.travel_times` car model estimates the unmodelled onward journey
from a proxy to the original destination. A displacement cost from the candidate node
to the ideal endpoint uses the existing 50 km/h endpoint access speed scale; this
prevents selection solely because a route stops farther short of its endpoint.
The network costs are free-flow approximations, without congestion, turn
restrictions or calibrated MATSim penalties. MATSim performs final routing later.

For missing/unmapped exits, the last inside-to-outside transition on the route
is the provisional exit. Re-entries are counted explicitly. The fallback is not
restricted to crossings adjacent to the destination country. Unreachable
entry/exit/destination combinations raise an error rather than dropping agents.

## Activities, facilities and diagnostics

- A road through-agent has a foreign origin, a one-second entry activity and a
  distinct foreign destination facility.
- A mapped reported exit adds a one-second activity at its own outgoing facility.
- A route-inferred exit adds no extra activity; it is part of the onward route.
- PT through-trips keep their direct endpoint trip and get no road border stops.
- Ordinary cross-border activity chains retain their existing construction.

The stage writes `through_crossings.csv` and `directional_links.csv` in its cache.
`exit_source` distinguishes `survey` (mapped reported exit), `network` (missing
exit) and `reported_unmapped_network` (reported code retained but not located).
For mapped exits, `exit_mapping_source` distinguishes `exact_name`,
`collected_label`, `official_office` and `reviewed_official_office`, preserving their provenance. The
reference audit also includes the official name, UUID, year, URL and coordinates.
`exit_border_scope` distinguishes national and LI–Austria customs-exit semantics.
The diagnostics include the route's last exit link, re-entry count, foreign
endpoint, proxy displacement and whether the constrained reported exit is also
the final exit. `reported_exit_equals_entry` flags respondents who report the
same crossing for entry and exit; those reports are retained, rather than
silently replaced. These are pre-routing diagnostics, not simulation events.

Crossing counts preserve the actual entry and reported-exit locations separately.
Missing exits are not counted again at the interview entry. Final route/event
analysis is needed for complete traffic counts including inferred exits.

## Validation and updates

When validating the fallback, hide reported exits in records whose exit locations
are known, route without those constraints, and compare predicted exits with the
reported crossings. Report vehicle-weighted agreement, spatial tolerance and
coverage by interview area and destination. This validates the located subset;
missing and unmapped exits can have different characteristics.

The initial read-only check on 23 October 2026 used 2,449 valid weekday road
through-observations for foreign residents from the 2021 survey. Of these, 776
reported exits could be located, 1,034 reported codes were unmapped and 639 exits
were missing. Hiding the 776 located exits gave 44.5% agreement within 2 km,
or 34.1% with vehicle weights. The fallback therefore remains approximate and
must not be interpreted as recovering the reported route reliably. Extending
the individual crossing reference is preferable to fitting special-case routing
rules to these records.

The initial collected-label extension located 990 exits. Comparison against the
official 2021 source subsequently exposed seven mislabeled associations that the
local proximity check could not detect: Riehen–Lörracherstrasse, Diepoldsau,
Brusata, Camedo, La Croix-de-Rozon, Crassier and Allschwil. Those aliases are now
blocked in the local-only reference and corrected using official locations.

With the official input, 1,645 reported exits can be located: 655 more than the
initial local extension. A further 165 reported exits remain unresolved, and
639 observations have no reported exit. Coverage is 67.2% of all 2,449
observations and 90.9% of the 1,810 observations reporting an exit. Of the located
observations, 839 use official smaller-crossing locations, 776 retain the exact
major-name mappings, and 30 retain three locally checked aliases not covered by
the official layer. This increases use of reported locations; it does not
improve or revalidate the routing fallback itself.
Vehicle-weighted located coverage among all observations rises from 37.7% with
the initial local extension to 62.2% with the official reference.

The independent network check matched all 83 accepted official sites directly
from their original office coordinates. Each selected the same outbound link as
its chosen canonical crossing point, using the production directional matcher
and cached scenario network. The source archive's checksum also matched the
published catalogue checksum. These checks establish reference and directional
consistency, not the respondent's full route or the meaning of a final exit.
The full 2,449-observation route check completed without dropping observations:
655 newly constrained routes and 70 corrected earlier assignments changed;
the other 1,724 retain their routing choices. Of the 1,645 mapped exits, 295
are followed by a later exit from Switzerland in the provisional route. These
remain flagged, pending clarification of the survey question and route review.

After the user's review, all 12 offices are attached to their appropriate road
links, adding 60 located observations. Coverage becomes 1,705 of 1,810 reported
exits (94.2%), or 63.5% of all observations with vehicle weights. There remain
105 reported but unresolved exits and 639 missing exits. The other 2,389 route
choices are unchanged. Two newly located observations use LI–Austria exits;
their last Swiss national exit remains a separate diagnostic.

The input audit also found conflicting coordinates: for example, the Romanshorn
ferry shares a point with Kreuzlingen motorway, and Perly shares a point with
Bardonnex motorway. Perly and Bardonnex are separately named crossings in the
[Geneva authorities' crossing list](https://www.ge.ch/document/g7-faq).
These conflicting aliases are not accepted. Other labels, including Inzlingen
and La Cure, have no canonical point within 500 m; the audit retains their
distances for review instead of silently attaching them to a distant point.

For future inputs, keep the survey CSV, accompanying codebook and crossing layers
consistent. Add or correct individual labels in the data directory, review their
codebook-name association in `EXIT_LABELS`, and inspect `road_exit_reference.csv`
before rerunning through-trip assignment. Do not widen the radius merely to
obtain more matches: inspect the collected and canonical coordinates first.
Adding independently identified smaller crossings improves reported-exit
coverage without fitting routing rules to individual observations. The optional
`cross_border_reference_path` selects an input release; it does not change the
direction-inference method or its thresholds.
