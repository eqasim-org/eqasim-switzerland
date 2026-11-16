# Geneva airport regional passenger demand

See the [airport workflow](README.md) for the package overview. To sample
coordinates, modes and departure times, use [data.airports.trips](trips.md).

`data.airports.geneva` returns a pandas DataFrame of passenger ground-access
movements between GVA and Swiss municipalities, in both directions. Its nine
observed regional groups remain constraints and audit outputs. Foreign groups
remain explicit unresolved rows. See the [allocation model](model.md). It is a
standalone input stage; it does not yet insert airport travellers into the
synthetic population or MATSim plans. Counts represent the full population by
default, irrespective of the population sampling fraction elsewhere in the pipeline.

## Inputs and evidence

The default PDF is `<data_path>/airports/geneva/geneva_airport.pdf` (the actual supplied
filename). The 32-page 2025 accessibility presentation has no extractable text
layer: pypdf and PyMuPDF both returned no text. After rendering the pages for
visual review, the relevant percentages were transcribed into
`<data_path>/airports/geneva/geneva_survey.csv`. The sibling JSON records the PDF SHA-256,
survey year and extraction notes. No PDF parser or OCR dependency is required at
runtime. Page references in the CSV are physical PDF pages, starting at one.

The reviewed tables cover profile shares (p. 4), resident catchments (pp. 5–6),
visitor/circuit catchments (pp. 8–9), combined catchments (pp. 11–12), and access
modes (p. 15). Mode shares are retained as evidence; this stage does not fabricate
a region-by-mode joint distribution from separate marginals.

Public data are saved in
`<data_path>/airports/geneva/data_from_websites/geneva_airport.json`, including observation year,
retrieval date, field provenance, source URLs and stable refresh-page URLs:

- [Airport annual report 2025, printed p. 141 / PDF p. 143](https://2025.gva.report/wp-content/uploads/2026/03/Rapport-annuel_2025.pdf): final total, airside counts and directional traffic.
- [Airport 2025 monthly statistics, PDF p. 1](https://www.gva.ch/content/dam/gva-aem/documents/airport/statistiques-a%C3%A9roportuaires/statistiques_pax_geneve-aeroport_2025_manuel.pdf): provisional monthly seasonality weights.
- Refresh through the [statistics](https://www.gva.ch/fr/aeroport/publications/statistiques) and [reports](https://www.gva.ch/fr/aeroport/publications/rapports) pages.

All Geneva datasets live under `<data_path>/airports/geneva/`, while the Python
stage stays at `data.airports.geneva`. Other airports can have sibling folders.
Runtime execution is offline and never refreshes sources implicitly. There are
no bundled dataset fallbacks in the source package.

```text
airports/geneva/
  geneva_airport.pdf              # supplied accessibility survey
  geneva_survey.csv               # reviewed survey figures
  geneva_survey.json              # PDF checksum and review metadata
  data_from_websites/
    geneva_airport.json           # traffic statistics and source metadata
    geneva_flight_times.json      # flight-board snapshot and timing assumptions
    annual_report_2025.pdf        # local copy of the annual report used
    monthly_statistics_2025.pdf   # local copy of the monthly statistics used
  geneva_ods.parquet
  geneva_ods.*                    # CSV exports, summary and cache signature
```

The public-data JSON records each archived PDF's URL, local path relative to
`airports/geneva/`, download date and SHA-256. The stage reads the reviewed
figures in the JSON; changing an archived PDF alone does not re-extract or
update those numbers. On refresh, update both the reviewed figures and their
archive metadata.

## Passenger accounting and allocation

Regions are CH-GE, CH-VD, CH-VS, other Swiss cantons, FR-74, FR-01, FR-38-73,
other French départements, and other countries. Codes and group membership are
explicit columns. Groups describe complete administrative areas, without
implying that demand is uniformly distributed inside them. Other Swiss cantons
remain one observed constraint; their municipal split comes from the allocation
model. The grouped Savoie/Isère and other foreign constraints remain unresolved.

1. Reconcile each table's rounded regional bars within countries to that table's
   normalized country totals. The combined country target is 63% CH, 35% FR and
   2% other countries. For example, GE's pooled probability is `0.63 * 33/62`,
   rather than treating the individually rounded bars as exact percentages.
2. Retain resident (`outgoing`) and visitor/circuit (`incoming_other`) priors.
   Their land-access profile weights are `57/96` and `(35+4)/96`. Survey labels
   describe the traveller's journey profile, not the ground-trip direction.
3. Fit the joint region/profile table by iterative proportional fitting to the
   observed pooled regional and profile margins. This reconciles independently
   rounded charts without losing the observed overall distribution. Priors and
   fitted probabilities are exported for inspection. Small unknown-location
   shares are redistributed within their known country; unknown countries fall
   into OTHER. Raw unknown observations remain in the reviewed CSV.
4. Estimate annual land-access passenger movements as
   `annual_passengers - direct_transit_passengers - connecting_multiplier * connecting_passengers`.
   **The default multiplier of two is a modelling assumption:** each connecting
   traveller uses an arriving and a departing flight without ground access.
   The report does not establish the counting convention unambiguously. Confirm
   with the airport; set it to one if the connecting count already includes both
   movements. Both total scenarios are included in the summary. Do not also
   remove the survey's 3% transit share: that would exclude airside demand twice.
5. Allocate to ground directions using the published arrival/departure ratio,
   reconciling its two-passenger discrepancy with the official annual total.
   Both profiles occur in both directions. Identical catchments in both
   directions are an annual approximation; no directional survey is available.
6. Divide by calendar days for an annual-average day, or apply normalized monthly
   traffic weights and divide by that month's days. Monthly totals reconcile to
   the final annual total despite the provisional source differing by one.

With the default inputs and multiplier two, the estimate is **17,366,723 annual
passenger ground trips**, or **47,580.06 per average day**. Multiplier one gives
17,576,098 annual trips. These are modelled ground counts, not measured road
traffic. Vehicle occupancy, escorts' return legs, employee commuting and freight
are excluded. There is no observed weekday, hourly, purpose or region-specific
seasonal distribution; a `workday` scenario elsewhere does not silently change
these counts. Survey-year/traffic-year mismatches emit a warning.

## Running and outputs

Use a dedicated Synpp config to run just this stage:

```yaml
working_directory: /path/to/airport_stage_cache
run:
  - data.airports.geneva
config:
  data_path: /path/to/input_data
  # geneva_airport_month: 2  # null by default: annual-average day
  # geneva_airport_demand_factor: 1.0
```

Run `uv run -m synpp <this-config.yml>` from the repository root. The airport stage now depends on the shared municipality allocation model,
which reuses STATPOP, STATENT and spatial inputs. It needs no synthetic-population
or Java simulation stages. `geneva_airport_demand_factor` explicitly
scales the totals, e.g. 0.1 for a later 10% scenario; it defaults to 1.0.

The default persistent output is `<data_path>/airports/geneva/geneva_ods.parquet`:

| Artifact suffix | Content |
| --- | --- |
| `.parquet`, `.csv` | Swiss municipality rows × profiles × ground directions, plus unresolved foreign groups |
| `.municipality_od.csv` | Municipality OD totals with profiles summed; unresolved foreign groups retained |
| `.regional_od.csv` | 18 directional regional ODs, profiles summed |
| `.probabilities.csv` | Reviewed priors and fitted region/profile probabilities |
| `.summary.json` | Totals, source links, assumptions and alternative airside total |
| `.time_distribution.csv` | Fifteen-minute flight/ground-event probabilities by direction and weekday |
| `.signature.json` | Input/model signature and output checksums |

`annual_trips`, `period_trips`, and `daily_trips` are fractional expectations;
corresponding `_integer` columns use deterministic largest-remainder allocation
and preserve the rounded overall total. No random sampling occurs. `month=0`
means the entire year; `period_days` gives the denominator. Preserve `profile`
when using the detailed table or use the aggregated regional file directly.
`GVA` is an airport node, separate from the Genève canton polygon.

All `geneva_airport_pdf`, `geneva_airport_survey_csv`, `geneva_airport_web_data`,
`geneva_airport_time_data` and `geneva_airport_cache` overrides accept absolute paths or paths relative to
`data_path`. A custom survey CSV requires a sibling `.json` with `pdf_sha256`
and `survey_year`. The local `airports/geneva/geneva_survey.csv` is selected by default.

## Refreshing and cache reuse

SHA-256 covers the PDF contents, reviewed CSV and its metadata, selected public
JSON, flight snapshot, stage/timing code and demand settings. Synpp's `validate`
hook also checks the exported files, so editing external inputs or deleting
exports invalidates Synpp's result as well as the persistent Parquet.
Unchanged inputs reload without fitting again. Output
checksums detect missing or corrupted artifacts, and a manifest is written last
after atomic file replacement. `override_geneva_ods: true` forces rebuilding on
each pipeline invocation; it cannot bypass PDF review.

For new traffic data, update the selected public JSON, its year, counts, links
and retrieval date. Keep monthly, directional and annual counts from compatible
periods. Review the connecting counting convention. To use a new PDF, review and
update the CSV percentages and page references, then update the sibling metadata's
PDF digest and survey year. A changed PDF fails explicitly until this is done;
stale survey numbers are never silently applied to a replacement PDF.

Focused local validation is in `tests/test_geneva_airport.py`: margin and total
conservation, monthly reconciliation, scaling, cache invalidation, malformed
sources, PDF review checks and reuse without fitting. Tests remain untracked.

## Departure-time distributions and random sampling

The airport-supplied PDF does **not** provide hourly passenger volumes. Its p. 27
shows the mode share conditional on flight-departure hour; it cannot identify
how many people travel in each hour. Pages 28–29 likewise show conditional mode
shares, rather than the prevalence of each arrival lead time.

The added public snapshot
`<data_path>/airports/geneva/data_from_websites/geneva_flight_times.json` records the official
[GVA flight board](https://www.gva.ch/fr/passagers/vols-et-destinations), retrieved
on 13 November 2026 for **13–19 November 2026**. It contains 2,096 operating-flight
records before excluding cancelled/deleted/diverted/rerouted services. Nested
codeshare marketing numbers are not counted as separate flights. The snapshot
stores UTC schedules, source/endpoint/query, local timezone, statuses and all
modelling assumptions. Refresh is explicit and offline runs never call the API:

```bash
uv run -m data.airports.geneva_times --output /path/to/new_geneva_flight_times.json
```

Select another snapshot with `geneva_airport_time_data` (absolute or relative to
`data_path`). A snapshot change invalidates the OD cache, and the generated
summary identifies the selected file and its dates. The public board currently
exposes today plus six days. Capture other seasons over time if an annual
profile is needed; the November week is **a seasonal flight-count proxy**, not
an observed 2025 or annual passenger distribution. Flight size, load factors,
airside connections by hour and region-specific timings are unknown. Every
included operating flight receives equal weight within its date. Dates then
receive equal weight in a grouped average. Counts remain based on 2025;
this extra snapshot supplies only timing shapes.

The stage exports `geneva_ods.time_distribution.csv`, with 15-minute bins and
probabilities summing to one separately for each direction/event/weekday:

| Direction | Event | Interpretation |
| --- | --- | --- |
| `to_airport` | `scheduled_flight` | Scheduled aircraft departure time |
| `to_airport` | `airport_arrival_target` | Desired arrival at GVA, before the flight |
| `from_airport` | `scheduled_flight` | Scheduled aircraft arrival time |
| `from_airport` | `ground_departure` | Departure from GVA after clearance |

The `day` column uses `monday` through `sunday`, plus `weekday` (Monday–Friday),
`weekend` (Saturday–Sunday) and `all` (all seven days). Each grouped probability
curve is the arithmetic mean of its daily normalized curves. This avoids
weighting a busier day more heavily. If a future snapshot spans multiple weeks,
each calendar date receives equal weight. These are timing averages; daily trip
volumes are not changed by selecting a group.

For compatibility, the CSV also retains integer `weekday` values: 0–6 for
Monday–Sunday, -1 for `all`, -2 for `weekday`, and -3 for `weekend`.
Bin edges are local wall-clock seconds from midnight on the **flight day**.
Values above 86,400 preserve next-day departures; do not truncate them. For
towards-airport trips, there cannot be a single regional-origin departure
distribution until travel times are supplied.

Ground timing is derived using explicit assumptions:

- Before departure: triangular lead time of 90/120/180 minutes
  (minimum/mode/maximum). [SWISS advises roughly two hours for short-haul and
  three for long-haul](https://www.swiss.com/fr/fr/fly/check-in/previous-day-check-in.html).
  This advice informs the range, but does not establish its shape or the
  90-minute minimum. All three parameters are modelling assumptions; they are
  saved in the snapshot and can be replaced.
- After arrival: triangular clearance time of 15/30/60 minutes. This entire
  distribution is an **uncalibrated assumption**, pending airport passenger-exit
  observations. It includes deplaning, baggage and border processing in aggregate.
- Early airport arrival targets are clipped to 04:00, when the
  [terminal reopens](https://www.gva.ch/en/passengers/travel-preparation/faq).
  This creates an explicit opening-time mass. It is an approximation to early
  passenger behaviour; the sampler then subtracts the supplied ground travel time.

Use the sampler when generating trips. It samples an operating flight from the
selected direction/weekday, draws the corresponding buffer and returns ground
**departure** times:

```python
from data.airports.geneva_times import get_time_distribution, sample_departure_times

data_path = "/path/to/input_data"

# Region -> GVA: supply actual/estimated travel times, scalar or one per trip.
to_airport = sample_departure_times(
    "to_airport", 1000, seed=518745, day="monday", data_path=data_path,
    travel_time_seconds=45 * 60,
)

# GVA -> region: ground departure = scheduled arrival + clearance.
from_airport = sample_departure_times(
    "from_airport", 1000, seed=518746, day="weekend", data_path=data_path,
)

# Retrieve the average weekday curve without sampling trips.
weekday_curve = get_time_distribution(
    "weekday", direction="from_airport", data_path=data_path,
)
```

Pass the pipeline's `context.config("random_seed")` or a reproducibly derived
seed in downstream generation. The function requires an explicit seed and either `data_path` or an explicit
`snapshot_path`. Names are case-insensitive. `weekday=0` remains supported as
an alias for `day="monday"`; do not pass both arguments together. For a custom
timing source, pass `snapshot_path=...` to the sampler or distribution getter.
`departure_seconds` retains negative and next-day values;
`departure_day_offset` and `departure_time_of_day` separate day and clock time.
`airport_event_seconds`, source flight/date and flight seconds remain available
for audit. The dates identify the source schedule, not a simulated traveller's
calendar date. Calculations use local wall-clock time, without attempting to
simulate elapsed time across daylight-saving transitions.

Timing tests verify probability conservation, cancellation/deduplication rules,
seed reproducibility, travel-time shifts, correspondence between sampled times
and exported probabilities, preservation of previous/next-day trips, named-day equivalence, and equal-day
averaging for weekday/weekend groups.

The timing engine also supports explicitly marked partial snapshots for other airports.
Geneva retains its complete seven-day requirement. The exported `covered_days` and
`complete_day_group` columns expose coverage; all Geneva day groups are complete.

The shared model preserves the original regional/profile expected and integer
totals when splitting Swiss groups into municipalities. The reported 47,580.06
daily trips are unchanged. Foreign municipality IDs remain null; see `spatial_resolution`.
