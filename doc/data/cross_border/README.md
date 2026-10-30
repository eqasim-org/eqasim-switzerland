# Cross-border survey demand and crossings

The 2021 A+GQPV road-border survey interviews travellers **entering Switzerland**.
The blank `FAHRTRICHTUNGGU` field does not mean unknown road direction: that field
describes entry/exit for rail interviews only. See the BFS
[2023 road-transport methods report](https://dam-api.bfs.admin.ch/hub/api/dam/assets/29270695/master#page=10),
page 10, footnote 5. This rule applies to road **border** interviews, not Alpine
interviews.

## Choosing a road within an interview place

`data.cross_border.road_candidates` supplies both foreign and Swiss OD sampling
with **capacity × importance** weights. The existing reference `importance` is
retained; there is no additional road-class multiplier and no new required config
parameter. Survey vehicle/person weights and interview-group controls are not
changed. The existing foreign-OD detour eligibility test and random seed remain
in use. The Swiss reference pool retains its existing within-group eligibility.
PT continues to use its original importance/gateway rules.

The capacity comes from the final `matsim.scenario.network.pt` road network,
before artificial border terminals are added. MATSim's link capacity is already
for all lanes: **do not multiply it by `permlanes` again**. Since the road survey
observes inbound traffic, the selection weight uses the inbound capacity, not
the sum of entry and exit capacities. This is an attractiveness assumption,
not a measured traffic share or a hard traffic quota.

Before applying these weights, the directional matcher identifies the road under
the reference point from detailed geometry. It looks for an anchor within 100 m,
allows nearby parallel links of the same OSM road class within 25 m of the closest
anchor, and follows directed continuity of that class for up to 2 km to find the
border link. If a point is already beyond the crossing in travel direction, it
searches backwards. These bounds are geometric safeguards, not fitted traffic
parameters. It uses the minimum capacity along this short anchor-to-border
corridor, so a wide customs-plaza link cannot inflate a narrower approach's
weight. It does not perform agent routing or replace turn restrictions. If the
road cannot be followed, the existing directional matcher remains the fallback;
its match type is recorded. Explicit reviewed exit links remain authoritative.
The same matcher is used by through-trip construction and scenario preparation.

Points sharing the same entry/exit link pair **within one interview group** count
as one physical choice. Their strongest original importance represents that
choice; duplicate importance is not added. All original point coordinates stay
available to the detour test. After filtering, the physical choice's weight is
divided over its eligible aliases, so extra reference points cannot gain extra
traffic. Different interview groups are never combined. If a group's capacity
is missing or invalid, that entire group uses importance-only weights and emits
a warning; all-zero importance uses equal physical-road weights.

The stage writes:

- `road_candidate_weights.csv`: original points, canonical aliases, entry/exit
  links, inbound capacities, original importance, physical selection weights,
  and expected alias shares before the detour filter.
- `directional_links.csv`: anchors, matched road classes, corridor distances,
  direction checks and fallback types for every road reference.

For the reviewed St-Louis points and the network checked on 2026-10-25, the
motorway approach has capacity 2,000 and the side road 500 vehicles/hour. Both
points have importance 5, giving an **80% / 20%** split when both are eligible.
The motorway now matches entry `2233381` and exit `2075581`; the side road keeps
`121456` / `121455`. These are candidate probabilities, not results from a rebuilt
population. Rerun the pipeline through `matsim.simulation.prepare` to inspect the
realised sampled flows. Existing inputs/caches were only read during validation.

## Vehicle demand calibration

`data.cross_border.generate_od` builds the foreign population and
`data.cross_border.swiss_residents_od` builds the Swiss reference pool. Both use
the shared `survey_calibration` helpers. The 2015 survey supplies road-vehicle
volume controls; the 2021 survey supplies the conditional purposes, countries,
locations and reported exits. Each stage controls its own residence group.

For `car` (the existing category includes motorcycles), the factor is:

```
2015 vehicle weight at interview-place ID / 2021 vehicle weight at that ID
```

Coordinates are recovered without removing observations from that denominator.
Occupants are not used to calculate vehicle factors. This replaces the national
mode × purpose × origin-country × destination-country person-weight ratios,
which could transfer a large national flow onto one surviving geocoded record.
The 2015 car record with occupancy 954 therefore contributes its vehicle weight
only. Invalid 2015 occupancies remain visible in the diagnostic output.

A local control requires at least ten positive-weight 2021 observations and an
effective sample size of five (`sum(w)**2 / sum(w**2)`). These are broad support
checks, not fitted crossing targets. Unsupported groups share a factor from the
remaining 2015 and 2021 vehicle totals. A group with no 2021 observations is
explicitly reported: pooling cannot manufacture its geographic support. A new
2021 group with no remaining reference control uses the national ratio, flagged
as such; it is not interpreted as an observed zero. In the configured weekday
inputs, all 29 road groups meet the checks for both residence populations.

Within each supported group, 2021 relative weights and journey patterns remain
unchanged. Rail and coach retain the former person-based national OD/purpose
calibration method. Invalid 2021 car occupancies (noninteger, less than one, or
above nine; above two for motorcycles) are sampled from valid 2021 peers of the
same vehicle type and, when supported, purpose, using the configured seed and
original 2021 vehicle weights. This only repairs passenger expansion. It never
changes a vehicle weight or removes its driver.

## Sampling the foreign population

`data.cross_border.sample` samples the expanded foreign population within each
**interview place × visitor/transit category (`label`) × travel mode**. Drivers
and passengers remain separate, so passenger counts cannot alter vehicle demand.
This replaces independent Bernoulli retention, which could noticeably change
local crossing volumes in a small population sample even with correct survey
weights.

Each stratum receives either the floor or ceiling of
`input_downsampling × full stratum population`. Largest fractional remainders
receive the remaining places, preserving the nearest integer to the requested
national sample size. Ties and random selection within strata use `random_seed`;
selection is without replacement and independent of input row order. Small
strata may round to zero. All rows of a selected person retain their original
identifiers, values and input order. A fraction of one retains everyone.

The stage writes `sampling_counts.csv` with full counts, fractional sample
targets, selected counts and expanded sample counts. It requires no new config
parameter and does not change survey weights, Swiss household sampling or
destination/routing rules. It preserves incoming demand proportions, not exact
two-direction traffic at each crossing: transit exits and routes can still add
passages at other interview places.

## Missing endpoints and geographic scope

The OD stages now also depend on the existing `data.spatial.nuts` and
`data.cross_border.road_exit_points` stages. No new configuration option or
external download is needed. Endpoint recovery uses the following evidence:

1. A Swiss municipality code, including the existing historical-ID mapping,
   or a NUTS identifier present in the spatial data and consistent with the
   endpoint's reported country.
2. For a missing foreign endpoint, a representative point of its country
   polygon as an explicitly labelled **directional proxy**, not a town/address.
3. If no country geometry exists, a missing road origin can start at its
   surveyed crossing group. It is not assigned an invented foreign residence.
4. A missing road destination without country geometry needs a mapped reported
   exit. An outward routing proxy is constructed from that exit; the downstream
   route still has to visit the reported crossing. Without this evidence, the
   observation remains unresolved and its weight is reported separately.

Already located Swiss endpoints are kept. If the point cannot be assigned to
a Swiss municipality but a valid Swiss municipality code exists, that code
supplies a representative point. No Swiss destination is inferred from another
respondent. Remaining geographic failures are exported before expansion.

`start_location_source` and `end_location_source` accompany the generated OD
records. In visitor and Swiss reference outputs they follow the reordered
endpoints, so `start` means the foreign end for visitors and the Swiss end for
Swiss residents. Country/crossing/exit proxies are excluded from residence-based
MMT filtering; crossing-only origins also bypass perimeter routing, which needs
a meaningful geographic direction. Legacy `residence_x/y` column names continue
to serve the downstream routing contract and must not be interpreted as known
addresses when their source is a proxy. Geographic exclusions still apply to
located origins, and overlapping exclusion polygons cannot duplicate records.

The stages write these diagnostics into their own cache directories:

| File | Meaning |
| --- | --- |
| `vehicle_calibration.csv` | 2015/2021/calibrated vehicle entries, raw and effective sample sizes, factors and support status by interview group |
| `survey_demand_audit.csv` | Interview-level weights, endpoint sources, occupancy repairs, unresolved locations and supported/unsupported trip categories |
| `survey_scope_counts.csv` | Weighted demand by group, mode, travel category and assignment scope |
| `reference_occupancy_flags.csv` | Suspicious 2015 car occupancy records; their vehicle weight is retained |
| `unlocated_swiss_end.csv` | Remaining Swiss endpoints that could not be placed, with their weights |
| `assignment_counts.csv` | Expanded and retained foreign agents, or Swiss reference-pool agents, by crossing and mode |

These are generated analysis files, not repository input CSVs. The stages retain
the existing return types, identifiers and downstream assignment structure.

## Counting directions and validating a rerun

For a weekday, road weights are annual **inbound vehicle counts divided by 260**.
Twice that number is a balanced two-direction benchmark. It is not a count of
actual generated crossing events, and no factor of two is applied during this
vehicle calibration.

Foreign visitors may produce an entry and a return; transit produces an entry
at the surveyed group and an exit elsewhere. Swiss residents receive one
explicit directional event per selected resident, using the controls below. Swiss `Within CH`
records (Swiss-to-Swiss journeys via abroad) and Swiss through-records are not
handled by that reference-pool assignment. They remain in calibration and are
explicitly flagged as outside assignment scope rather than redistributed to
visitors. Supporting these journey types is a separate behavioural change;
a successful calibration does not establish final all-person
traffic agreement. The separately generated French population and actual
through-exits also have to be included in the final event comparison.

### Swiss directional event allocation

`synthesis.population.models.cross_border` converts each retained incoming
road-driver reference into one entry event and one balancing exit event.
Expanded car-passenger references receive the same treatment, separately from
drivers. PT retains its historical total, split as evenly as possible by
canton/interview group/mode; an odd residual direction is seeded. This change
does not reinterpret PT survey weights or change the survey-calibration stage.

With the existing `cross_border_use_spatial_calibration: true` default, the fitted
model and raking supply selection weights. Event quotas set the volumes by
origin canton, interview group, mode and direction. Residents are selected
without replacement, first in the reference municipality and then, if needed,
from the remaining eligible residents of the same canton. Municipality IDs are
normalized across census and survey inputs. Driver events are allocated to
adults first. Unsupported demand remains a reported shortfall; it is not moved
to another canton, assigned to an underage driver or turned into an extra trip.
Setting spatial calibration to false retains unconstrained Bernoulli selection
and reference matching; exact event quotas are then not enforced.

This is an **event-accounting assumption**, not a survey estimate of distinct
travellers. For example, 1,000 retained incoming car references require 1,000
entries and 1,000 exits. The existing one-trip design represents these with
2,000 distinct selected residents, replacing their ordinary diary chains.
No residents are added to the population and no same-day return is inferred.
Inferring complete cross-border tours would require a separate behavioural model.

`synthesis.population.trips` consumes the matched direction instead of drawing
entry/exit again. Existing directional MZ departure-time sampling is retained.
The original OD IDs and directional facility IDs are reused, so event expansion
does not duplicate the facility inventory. New matched attributes are
`border_crossing_direction` and `border_interview_place`.

Overlap prevention remains based on residence: the foreign survey reader
excludes Swiss residents; the Swiss model removes residents and reference
origins inside `cross_border_exclude_shapefiles`, which are covered by the
external population. The polygon exclusion in both the foreign generator and
the Swiss model is applied only when `include_external_population: true`.
With `false`, retain the configured polygon if reusing a network cache: it
does not remove survey demand, Swiss training observations or event targets.
Changing the polygon itself can still invalidate network/geography stages
because they also declare this setting. MZ border-crossing donors are excluded from ordinary
matching. Incidental crossings of domestic routes are not another explicitly
assigned survey event and must be examined separately in routed-link counts.

Additional diagnostics in the model cache are:

| File | Meaning |
| --- | --- |
| `crossing_event_scope.csv` | Road entry/exit targets before and after geographic exclusion; unchanged legacy PT total |
| `crossing_event_assignment.csv` | Full-population targets, assigned events and shortfalls by canton/group/mode/direction |
| `crossing_origin_matching.csv` | Municipality matches versus canton fallbacks for those cells |

The trips cache also contains `swiss_crossing_trip_counts.csv`: selected sampled
events, mutually exclusive truck-driver/outside-Switzerland exclusions, and
generated events by group/mode/direction. Household downsampling, optional
population scaling, matching losses and later exclusions can change final totals.
In particular, a 5% sample need not contain exactly 5% of each small directional
cell. Compare the model ledger, sampled trips and routed passages separately.

No new configuration parameter is required. Rerun from the changed model through
`matsim.simulation.prepare`, allowing Synpp to rebuild downstream stages. When
testing the Java final-turn/length fixes as well, select the corrected JAR with
`eqasim_path`; the default older JAR does not include those fixes.

To rebuild only the two OD outputs with the working configuration:

```bash
uv run -m synpp config_dib_10.yml \
  --run data.cross_border.generate_od \
  --run data.cross_border.swiss_residents_od
```

The explicit `--run` arguments also avoid requesting education-only stages when
working on the separate cross-border branch. For a subsequent full scenario,
rerun the downstream stages normally. Never mix new OD outputs with old sampled
agents when comparing counts.

An isolated validation on the configured weekday inputs matched all 58 road
group/residence controls. Before geographic/model-scope exclusions, the balanced
benchmarks are 40,444.18 vehicles/day for Group O and 14,353.30 for Vallorbe road
(ID 108; collected crossing FID 89, label 544). These include motorcycles, exclude
car passengers, and should not be presented as final simulated crossings. Five
foreign car observations remain unlocated, representing 117.43 inbound
vehicles/day nationally (about 0.02% of the combined car control).

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

## Boundary endpoints for foreign car visitors and transit

`matsim.simulation.prepare` replaces the foreign approaches of
**all survey-generated From-To car visitors and Through car agents**, regardless
of origin distance, with short directional terminal links.
`cross_border_boundary_endpoints` defaults to `true` when cross-border demand is
included; set it to `false` to keep the original foreign origins and approaches.
Car passengers, PT, the separate French
population and Swiss residents are unchanged.

Transit uses its surveyed entry and mapped reported exit. When no mapped exit
facility exists, its existing route-inferred exit link supplies the sink instead;
the diagnostics retain `exit_source` and `inferred_exit_link` so this is distinguishable
from a reported exit. The default return-crossing field on destinations is never
used as a transit exit fallback. Missing or unsuitable exit links retain the original
plan with a diagnostic reason. Explicit entry/exit activities and their times stay
fixed. For inferred exits there is no explicit planned exit time: the final activity
keeps its original start time, while routing and QSim determine actual arrival.
Terminals do not prevent intermediate exits/re-entries on the route through CH.

The change runs after directional crossing assignment and before population
routing. A source link joins the exterior node of the assigned entry link; a
separate sink leaves the exterior node of the assigned exit link. They normally
extend 100 m outside Switzerland, with shorter/rotated alternatives where the
border bends. These are artificial injection/removal links, not real roads or
residential locations. Both connectors must remain wholly outside Switzerland;
ambiguous directional matches retain their original treatment and are reported.
Existing links, restrictions, facilities and Swiss destinations are preserved.
New terminal facilities keep the relocated activities' coordinates and link IDs
consistent with eqasim's scenario validator; they offer the original endpoint
activity types.
Connector speeds use the corresponding crossing link. Artificial terminals have
at least 1,000,000 vehicles/hour of flow capacity and 75,000 lane-metres of storage
(about 10,000 cars at the usual 7.5 m cell size), before QSim's sampling factors.
The lane count is a technical storage provision, not a physical road width:
it increases for shorter fallback connectors without changing their geometry or
travel time. At 5% sampling these provisions give about 13.9 vehicles/second and
500 cars of storage. Existing roads keep their original capacities. These finite
buffers avoid imposing an additional artificial bottleneck; congestion on the
real network can still delay admission. Connectors carry no toll or intersection
attributes and are classified as outside for network calibration. Their geometry
is appended to the local prepared detailed-network CSV with matching directed IDs.

A standalone QSim regression using the scenario's MATSim 2026.0 JAR compared
old/new terminals in six fixtures (12 runs): clustered injection at 1%, 5% and
10% sampling, sink receiving capacity, a 1 m fallback, and a real-road storage
bottleneck. All 1,240 simulated trips completed. With 100 simultaneous departures
at 5% and sufficient receiving-road storage, maximum source injection delay fell
from 14,256 s to 7 s; the real-road bottleneck kept the final arrival times
identical. In the sink fixture, maximum additional upstream waiting fell from
7 s to zero. The narrow-road fixture still backpressured the source, confirming
that generous terminals do not bypass congestion on real links. These deliberately
clustered stress tests are not forecasts of national travel-time savings. They
use standard queue dynamics and a long stuck-time threshold so forced movements
cannot hide blocked links; nationwide performance still requires simulation.

The first departure is moved to the existing planned entry time minus the
short approach's free-flow time. Entry, Swiss-activity and exit times stay fixed;
the final external activity starts after the short return leg. This preserves a
**nominal** entry time, not a guarantee of realised QSim arrival times. Queueing
at injection and actual crossing counts still require simulation validation.

Preparation writes `border_endpoints_persons.csv` (patched, unsupported or absent
persons, reasons and timing) and `border_endpoints_links.csv` (terminal geometry
and reference crossing links). All modifications affect prepared copies only.
Before adding terminals, it checks directed connectivity separately for every
configured network-routing mode and writes `border_endpoints_connectivity.csv`.
An unrelated disconnected road component raises an error. Once terminals are
added, the prepared config sets `routing.networkRouteConsistencyCheck=disable`:
MATSim's default check removes intentional one-way sources/sinks and then aborts.
The setting remains in the exported simulation config; no manual override is
needed. Without added terminals, preparation leaves the setting unchanged.
Regenerate preparation and its network dependencies after changing the road
buffer; existing scenario outputs do not acquire these changes automatically.
The minor-road retention buffer is now 5 km around the modelled region,
registered as `minor_road_border_buffer=5000.0` (metres) in
`matsim.scenario.network.convert_osm` so changes invalidate the network cache.
A local
turn-aware check retained both BCP_190 directions with 5 km; they were removed
with 2 km. This does not guarantee that every border crossing is preserved.

### Repairing detours within grouped interview places

`cross_border_route_repair` defaults to `true` in `matsim.simulation.prepare`.
Set it to `false` to retain the original capacity/importance-weighted crossing
choices without this preparation-time repair. No additional input file is needed.
The repair operates **after** directional facility matching, when the actual Swiss
activity links are known, and **before** boundary terminals and final population
routing. Upstream OD tables, sampling weights, people, interview groups and Swiss
activity locations are unchanged. Consequently, upstream assignment tables show
the original sampled crossings; the prepared plans and repair audit show final
crossings.

Only car drivers assigned to an explicit `Gruppe ...` interview place with more
than one eligible physical road crossing are considered. Individually named
interview offices, passengers, PT, the separate French population and ordinary
Swiss travellers without a survey-assigned border activity are not redistributed.
Capacity and road importance still determine the initial sampling. Duplicate
reference points for the same directed road pair remain one physical candidate.

1. Route the existing domestic journey with the configured eqasim JAR's
   free-flow, link-to-link SpeedyALT router, retaining car turn restrictions.
   Count crossings using the geographic Swiss border, including roads absent
   from the crossing reference. Keep valid existing assignments.
2. For a problematic assignment, compare candidates **within the same group**.
   A Swiss resident's one-way leg must have exactly the expected entry or exit;
   a visitor must have one entry on arrival and one exit on return; transit must
   have one entry and one exit. A route with a forbidden car turn is ineligible.
   The real domestic network is not restricted to Swiss roads or otherwise edited.
3. For visitors, prefer the fastest supported whole return journey: foreign
   approach, Swiss visit and foreign return. Use unprojected survey coordinates
   with `reported` provenance, retaining their original precision from the sample.
   These can represent towns/regions rather than exact addresses. Attach to the
   nearest car-road detailed geometry within 1 km, keeping tied directed links.
   Approach/return times use the existing directed Pandana model on a temporary
   exterior-only graph, so they cannot enter Switzerland at a different crossing
   first. These are approximate ranking costs without turn restrictions; the
   domestic route checks in steps 1–2 remain turn-aware. Crossing links shared between
   query segments are counted only once in whole-journey time. If no candidate
   has supported foreign legs, visitors fall back to the valid domestic journey.
4. Transit changes require a mapped **survey** exit and supported foreign origin
   and destination. Keep that exit fixed; only the ambiguous grouped entry may
   change. Exclude the same physical entry/exit pair from inferred group choices,
   rather than interpreting a very short turnaround as a successful transit trip.
   Unmapped or network-inferred exits remain unchanged and explicitly flagged.
   This does not override an individually observed entry office.
5. Select the fastest valid candidate, with stable point-ID tie-breaking. Keep
   the original plan when no supported candidate exists. No traveller is dropped
   and there is no national/cantonal crossing-count target in the selection rule.

Repaired people receive local `borderRouteChoice:...` facilities. Shared original
facilities are not moved. The local visitor destination copy supplies matching
entry/exit IDs to the existing terminal writer; mapped transit exits stay intact.
Activity times are retained, followed by the usual terminal departure adjustment.
When `cross_border_boundary_endpoints: false`, repair is limited to Swiss residents:
foreign approaches would otherwise remain attached to their old crossing-dependent
projections.

The preparation cache's `border_route_choice/` directory contains:

- `persons.csv`: original/selected crossing, category and interview place, outcome
  (`kept_valid`, `repaired`, or an explicit unresolved reason), crossing counts,
  domestic travel times, choice basis and available whole-journey travel time.
- `candidates.csv`: feasible alternatives and their domestic/whole-journey times
  for problematic assignments only.
- `foreign_attachments.csv`: coordinates, attached directed links and snap distance.
- `*_queries.csv` / `*_routes.csv.gz`: deduplicated routing queries and metrics.
  Domestic queries include turn-violation checks. Exterior results identify
  `routing_method=pandana_exterior` and leave distance/turn-violation metrics blank
  because those queries estimate time only. `inside_nodes.txt` records geographic
  classification.

`BorderRouteQueries.java` is a small JDK source-launch helper using the pipeline's
existing JAR; it does not build or modify a second eqasim version. Domestic queries
run with one routing worker and the configured Java heap. Preparation
therefore adds routing work, primarily for problematic grouped assignments. The
selection is based on free-flow routes; realised QSim/replanning crossings still
need validation after rebuilding and running the scenario.

Both routing paths reject topologically unreachable pairs before routing. They
collapse the directed graph into strongly connected components and test reachability
between components; distinct components can still be connected in one direction.
The check ignores turn bans and therefore rejects only paths that are impossible
even without those bans. Reachable domestic pairs still require SpeedyALT. For
foreign approaches, Pandana computes the remaining approximate travel times in a
batch. Start/end link traversal is included, with no intermediate Swiss roads or
border links. Equal/adjacent endpoints do not require an intermediate routing query.
The original scenario network is never cleaned or modified by these checks.

Initial implementation validation (27 October 2026, before replacing exterior
ranking costs with Pandana) used copies and saved routing batches
from the Group A experiment, not a new national preparation. The production
selector repaired the same 792 Swiss/visitor assignments and the same 80 supported
transit cases. For the 2,301 Swiss/visitor test people, geographic crossing passages
fell from 5,012 to 2,996; 19 unsupported cases retained their original assignment.
The 80 repaired transit cases fell from 268 to 160 geographic passages while
retaining their survey exits. All 2,610 fixture people and their non-border
activities/times were preserved. Correcting double-counted border-link time in
whole-journey comparisons changed four visitor candidate choices relative to the
initial experiment, without changing the repaired cohort or crossing totals.
The JDK source helper reproduced 80 full-network and 40 exterior route metrics
exactly; 32 local regression checks passed, including compatibility with the
existing terminal writer. A read-only dependency check found no cycles. Nationwide
results and congested simulation performance remain to be checked after rerunning.

The subsequent performance check used all 3,356 exterior queries from the first
national rerun. Pandana completed them in 57 seconds including network reading
and graph construction: 3,061 had an approximate path and 295 were unavailable.
For 261 pairs shared with the earlier exact exterior test, none of the previously
reachable pairs became unreachable; the median time reduction was 0.88 seconds,
but 98 pairs differed by more than a minute. These costs are therefore explicitly
approximations and can change rankings. Domestic turn-aware eligibility is retained.
The directed reachability implementations were checked against independent BFS
on 145,200 Java pairs and 64,000 Python pairs, including one-way component bridges;
34 local regression tests passed. No active pipeline process is hot-patched by
these code changes; restarting preparation is required to use them.


## Cross-border activity timing

`data.cross_border.activities` adapts sampled microcensus chains to cross-border
locations. `retime_with_travel_times` estimates travel times from distance and
mode, works backwards from the first Swiss activity and forwards from the last,
and keeps inserted border stops at one second.

If inbound travel would require departure before the initial activity starts,
the Swiss activity block is delayed by the minimum necessary offset. Its
activity durations and relative times are preserved. The initial activity keeps
its start, and the return journey follows the delayed last Swiss activity.
The final activity retains its original end unless the return arrives later.
Other chains retain their Swiss activity times. Through traffic keeps its
original first departure and is scheduled forwards.

The stage logs the number of delayed chains and the maximum delay. This rule
applies to both MATSim and SUMO inputs. Rerunning the pipeline detects the stage
change and rebuilds dependent caches; existing cache files need no manual edits.
