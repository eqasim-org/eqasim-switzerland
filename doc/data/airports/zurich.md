# Zurich airport passenger demand

See the [airport workflow](README.md) for the package overview. To sample
coordinates, modes and departure times, use [data.airports.trips](trips.md).

`data.airports.zurich` builds passenger ground-access counts from the available
public evidence. All inputs, archived sources and outputs live under
`<data_path>/airports/zurich/`. Runtime execution is offline. This standalone
stage does not yet create people, assign coordinates or add MATSim plans.

The current estimate covers **23,226,058 local scheduled/charter passenger
movements in 2025**, or **63,633.04 per calendar-average day**. This is already
arrivals plus departures: do not multiply it by two. Transfers, transit and general
aviation are excluded, as are employees, escorts, freight and empty vehicle legs.

## Available sources

| Source and local archive | Evidence used |
| --- | --- |
| [Statistics Report 2025](https://newsroom.flughafen-zuerich.ch/download/57e5442b-9ffe-4503-b491-866b87ca994f/statistikbericht_2025.pdf), `data_from_websites/statistics_report_2025.pdf` | Physical PDF pp. 25–27: monthly local, transfer, transit and directional terminal counts. Page 26 explicitly states that transfer is counted twice. |
| [Integrated Report 2025](https://report.flughafen-zuerich.ch/2025/ar/app/uploads/Integrated_Report_2025_FZAG.pdf), `data_from_websites/annual_report_2025.pdf` | Annual passenger accounting cross-check, including 23,226,058 local passengers. |
| [ARE/HSLU Definition touristische Verkehre, 2024](https://www.are.admin.ch/dam/de/sd-web/kU22CgrcOCsT/ARE(2024)-Definition_touristischer_Verkehr.pdf), `data_from_websites/are_tourist_traffic_2024.pdf` | Printed p. 76 / physical PDF p. 86: historical 88% Switzerland / 12% abroad ground-access origins, citing Interface & EBP 2018. The underlying survey year is not established in this source. |
| [Cantonal airport report 2024](https://www.zh.ch/content/dam/zhweb/bilder-dokumente/themen/mobilitaet/luftverkehr/flughafenberichte/Flughafenbericht%202024.pdf), `data_from_websites/cantonal_airport_report_2024.pdf` | Context only: p. 12 gives Zurich airport's market capture in each canton, not the share of Zurich passengers coming from each canton. It cannot be normalized into an airport OD distribution. |
| [Airport modal-split release, 2024 survey](https://newsroom.flughafen-zuerich.ch/en/modal-split-continues-to-develop-positively/), `data_from_websites/modal_split_2024.html` | Context only: 46% public transport covers all airport users and excludes walking/cycling. It is not a passenger-specific mode share and is not used to assign modes. |
| [Official departure board](https://www.flughafen-zuerich.ch/en/passengers/fly/flightinformation/departures) and [arrival board](https://www.flughafen-zuerich.ch/en/passengers/fly/flightinformation/arrivals), `data_from_websites/flight_board_raw.json` | Scheduled operating-flight timing for 13–17 November 2026, retrieved from the board's public `/flights` endpoint. Passenger loads are unavailable. |

`data_from_websites/zurich_airport.json` records URLs, refresh links, local archive
paths, retrieval dates, checksums, reviewed table identities and known gaps.
`zurich_monthly_2025.csv` contains the extracted observations;
`zurich_regions.csv` contains the two supported catchment weights.

## What the regional OD means

The observed spatial evidence is deliberately coarse: **Switzerland (CH)** and
**outside Switzerland (FOREIGN)**. The 88/12 historical access-origin benchmark
is applied to 2025 traffic and to both ground directions as an explicit modelling
assumption. It includes Swiss residents and visitors staying in Switzerland; it
is not a residence or nationality split. There is no contemporary observed canton-level OD or foreign-country split.
The [shared allocation model](model.md) now distributes the Swiss component to
municipalities using population, hotel counts and Euclidean distance. It transfers
its coefficients and Swiss resident/visitor mixture from Geneva. Those municipal
shares are modelled, while the source passenger totals stay fixed.

Monthly local traffic is observed directly. To estimate its directional split,
the stage subtracts half the twice-counted transfer total from each directional
terminal total. Each connecting traveller has an arrival and a departure; transit
is reported separately. Monthly checks enforce:

```text
local = terminal arrivals + terminal departures - transfer movements
```

Annual local direction totals are 11,563,111 towards ZRH and 11,662,947 away from
ZRH. These remain four regional audit rows. The main output now expands Swiss
rows to municipalities, with airport node `ZRH` separate from geographic regions.
The two foreign directional rows remain explicitly unresolved. Region shares sum to one, so no passenger demand is
lost when the spatial evidence is coarse.

A configured month uses that month's actual local and directional counts and
calendar days. The default divides annual demand by 365 (366 in a leap year).
The stage does not infer workday volumes from the partial flight-board snapshot.
`zurich_airport_demand_factor` explicitly scales demand; population sampling
settings elsewhere do not silently change this full-population input.

## Timing and coverage

`zurich_times.py` uses the same tested probability and sampling engine as Geneva,
with Zurich-specific source data and paths. The normalized snapshot is
`data_from_websites/zurich_flight_times.json`. The capture has five dates,
13–17 November 2026, with 2,962 commercial flight records before status filtering.
Noncommercial services and recognizable cargo/freight labels are excluded;
codeshare names nested within a flight are not counted again. Unlabelled cargo
may remain. This is a seasonal flight-count proxy, not a 2025 passenger-time
observation; connecting passengers at this hub may have a different hourly shape
from local passengers.

Available named days are Friday, Saturday, Sunday, Monday and Tuesday.
**Wednesday and Thursday are unavailable**, and requesting them raises an error.
`weekend` averages the complete Saturday/Sunday pair equally. `weekday` currently
averages only the three covered weekdays (Monday, Tuesday and Friday), so it is
an incomplete weekday proxy. `all` averages all five dates equally. Every
selected date has equal total weight; operating flights within a date have equal
weight. The CSV includes `covered_days` and `complete_day_group` so missing days
cannot be mistaken for observed data. Names are case-insensitive; integer days
0–6 and the older `weekday=` argument are also supported.

Ground-time assumptions are separate from the observed schedules:

- Region → ZRH: scheduled departure minus triangular check-in lead time
  90/120/180 minutes, then minus the supplied ground travel time. Airport arrival
  targets are clipped to 04:00, following [Zurich check-in opening](https://www.flughafen-zuerich.ch/de/passagiere/fliegen/rund-um-den-flug/check-in).
- ZRH → region: scheduled arrival plus triangular clearance time 15/30/60 minutes.

The triangular parameters are **uncalibrated assumptions**, not measured
passenger distributions. [SWISS advice](https://www.swiss.com/ch/de/fly/check-in/previous-day-check-in.html)
informs the two-to-three-hour check-in range, without establishing its statistical
shape or minimum. The airport-event distribution towards ZRH gives an arrival
target; it cannot be an origin departure distribution before travel time is supplied.
Negative and next-day departure seconds are preserved, with explicit day offsets.

```python
from data.airports.zurich_times import get_time_distribution, sample_departure_times

# data_path is your configured input-data root.
weekend = get_time_distribution("weekend", direction="from_airport", data_path=data_path)
times = sample_departure_times(
    "to_airport", 1000, day="monday", seed=518745,
    travel_time_seconds=45 * 60, data_path=data_path,
)
```

Capture other days/seasons explicitly over time. The refresh tool preserves
reviewed timing assumptions and recomputes coverage from the new board:

```bash
uv run -m data.airports.zurich_times --output <data_path>/airports/zurich/data_from_websites/zurich_flight_times.json
```

It updates the raw board archive and its source checksum. Newly available named
days are exported automatically. Historical timing captures should be copied to
separate dated files before replacing them if they are needed for later analysis.

## Running and outputs

Use a dedicated Synpp config:

```yaml
working_directory: /path/to/zurich_stage_cache
run:
  - data.airports.zurich
config:
  data_path: /path/to/input_data
  # zurich_airport_folder: airports/zurich
  # zurich_airport_month: 7
  # zurich_airport_demand_factor: 1.0
  # override_zurich_ods: false
```

Run `uv run -m synpp <this-config.yml>` from the repository root. No synthetic-population build or Java simulation is needed. The shared model
reuses full census and spatial stages, which can come from existing caches. The folder override accepts
an absolute path or a path relative to `data_path`.

| Output in `airports/zurich/` | Content |
| --- | --- |
| `zurich_ods.parquet`, `zurich_ods.csv`, `zurich_ods.regional_od.csv` | Main Parquet/CSV: Swiss municipal ODs plus unresolved foreign demand. Regional CSV: four original audit ODs. Annual/period/daily counts and integer allocations |
| `zurich_ods.municipality_od.csv` | Municipality OD table, retaining explicitly unresolved foreign rows |
| `zurich_demand_monthly.csv` | Unscaled observed monthly counts, derived local directions and calendar-average daily local totals |
| `zurich_ods.time_distribution.csv` | Fifteen-minute scheduled-flight and ground-event probabilities by day/group, direction and event, with coverage flags |
| `zurich_ods.summary.json` | Source links, available data, totals, assumptions and missing evidence |
| `zurich_ods.signature.json` | Input/model signature and output checksums |

Integer counts use deterministic largest-remainder allocation and preserve each
rounded overall total. The probability table and sampler normalize per direction,
event and selected day group.

## Updating evidence

The 46-page statistics PDF has a usable text layer. The maintenance extractor
reads just pp. 25 and 27 with the optional `pypdf` package, checks all monthly
accounting identities, and reproduces every saved monthly value. It does not
require OCR. Normal Synpp execution reads the verified CSV and needs no PDF
package. To re-extract after reviewing the archived report and annual metadata:

```bash
python -m data.airports.zurich_sources --folder <data_path>/airports/zurich
```

The helper needs `pypdf` installed in the selected Python environment. For a new
layout or reporting year, review page references and annual metadata first; a
failed reconciliation stops extraction. `monthly_file` and `regions_file` in the
metadata select the reviewed CSVs. A new regional survey can replace the
coarse weights once its geographic definitions and denominator are reviewed.

SHA-256 covers reviewed CSVs, metadata, archived evidence, timing snapshot,
relevant model/helper code and settings. The Synpp validation hook also checks
exported files, so missing or modified exports invalidate its cached result.
Sources and reviewed CSVs are checked against their recorded digests;
a changed PDF requires review rather than silently retaining old extracted
figures. Output checksums detect missing or corrupted exports. Unchanged inputs
reuse the Parquet; `override_zurich_ods: true` forces regeneration.

Focused local tests cover transfer accounting, count conservation, monthly
seasonality, scaling, cache invalidation, source review, timing coverage and
reproducible sampling. Geneva regression tests also cover the shared helpers.
