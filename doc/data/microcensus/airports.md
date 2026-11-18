# Airport passenger activity-chain donors

`data.microcensus.airports` returns a combined pandas DataFrame of airport
passenger diaries from **2015 and 2021**, with **one row per connected Swiss airport-related segment**. A respondent
with separate departure/return portions can have multiple linked rows.
It is independent of ordinary microcensus cleaning and population synthesis.
The stage reads the raw journeys and stages so it can retain air journeys with
car-passenger access that the cleaned main-mode classification can obscure.

## Sources and detection

| Edition | Default folder under `data_path` | Delimiter | Flight stage code |
| --- | --- | --- | ---: |
| 2015 | `microcensus` | comma | `f51300 = 17` |
| 2021 | `microcensus/21` | semicolon | `f51300 = 19` |

Each folder must contain `zielpersonen.csv`, `haushalte.csv`, `wege.csv` and
`etappen.csv`. Mode, employment and student codes follow the supplied database
structure workbooks inside `MZMV2015_mit_Geo.zip` and `MZMV2021_mit_Geo.zip`;
see the [source inspection](../airports/population_integration.md).

Source columns are renamed immediately on reading. The extraction and nested
JSON stage records use descriptive names throughout:

| Source column | Name used in processing |
| --- | --- |
| `HHNR`, `WEGNR`, `ETNR` | `source_person_id`, `trip_id`, `stage_id` |
| `alter`, `gesl`, `WP` | `age`, `sex_code`, `person_weight` |
| `f51300`, `f51310a` | `mode_code`, `car_ownership_code` |
| `wzweck1`, `wzweck2` | `purpose_code`, `return_home_code` |
| `f52900`, `f52950` | `stage_purpose_code`, `previous_activity_code` |
| `rdist`, `f51100`, `f51400` | `distance_km`, `departure_minutes`, `arrival_minutes` |
| `S_LND`, `Z_LND` | `origin_country_code`, `destination_country_code` |

Projected coordinates are `origin_x`, `origin_y`, `destination_x` and
`destination_y`; the source coordinates have an explicit `_source_` qualifier.
The six activity-status responses have descriptive `employed_activity_status_*`
and `nonemployed_activity_status_*` names. Raw codes appear only in input mappings.

The stage identifies a flight endpoint within `microcensus_airports_radius_m`
(default **3,000 m**) of a configured Geneva/Zurich airport point. Ground stages
before the flight give an access event; ground stages afterward give an egress
event. Each block must be observed within the same journey, contain no other
flight, and have its other ground endpoint in Switzerland (`S_LND`/`Z_LND =
8100). This selects passengers with evidence of an air journey, rather than
airport employees or people only dropping someone off.

Airport points use the existing `airport_model_airports` option; the default is
GVA and ZRH. Detection is approximate: the search circle is not a terminal
polygon, and survey coordinates can be imprecise. `airport_match_distance_m`
in each event records how far its reported flight endpoint is from that point.

After identifying a respondent, the stage keeps only ground stages whose
origin and destination countries are both Switzerland (8100). Flights, foreign
stages and cross-border stages break the diary into separate segments. Only
segments connected to an observed airport access/egress event are returned.
A crossing stage is excluded, not geometrically clipped at a national boundary.

Departure segments stop at the departing airport; arrival segments start at the
arriving airport. Foreign activities are not retained, even in the nested JSON.
The last Swiss access activity / first Swiss egress activity has purpose
`airport`; the code GVA/ZRH is stored separately, not embedded in that purpose.

Purposes share the same mapping as the ordinary microcensus readers:
`work`, `education`, `work_secondary`, `shop`, `leisure`, `other`, `home`,
`interaction`, `border` and `unknown`, plus the new `airport` boundary purpose.
Business travel becomes `work_secondary`; errands and escort activities become
`other`; return-home flags become `home`. The existing 2021 reader's treatment
of raw purpose code 12 as `other` is preserved (2015 uses `unknown`). Unidentified
initial purposes fall back to `other`, as in the ordinary reader.

`activity_chain` uses the **same hyphen encoding** as
`data.microcensus.activity_chains`, e.g. `home-work-airport` or
`airport-home-leisure-home`. `mode_chain` has one broad mode per retained trip,
using that stage's vocabulary (`car`, `car_passenger`, `pt`, `bike`, `walk`,
`unknown`). Detailed taxi/ride-hail/rail/car distinctions remain in the other
mode columns and JSON. There are always one more activities than trip modes.

## Returned columns

| Columns | Meaning |
| --- | --- |
| `person_id` | Respondent key such as `mz2015_<source ID>`; shared by that person's separate Swiss segments |
| `chain_id`, `segment_index` | Unique segment key and its index within the respondent |
| `source_person_id`, `survey_year` | Original respondent ID and survey edition |
| `age`, `sex` | Reported age; sex 0=male, 1=female, matching pipeline convention |
| `employed`, `is_student` | Nullable booleans from source employment and activity-status codes |
| `home_x`, `home_y` | Reported household location in EPSG:2056, independent of the diary starting point |
| `home_airport_distance_m` | Euclidean home-to-airport distance when there is one distinct airport; null for multiple airports |
| `home_airport_distances_m` | JSON mapping from each visited airport to its home distance |
| `airport`, `direction` | Airport codes and `to_airport`/`from_airport`; multiple values joined by `|` |
| `mode`, `mode_detailed` | Airport access/egress mode summaries; multiple event modes joined by `|` |
| `airport_event_count`, `airport_events` | Number of airport events and a JSON array containing each event's linked airport, direction, mode, stage-mode sequence, times and distance |
| `activity_chain`, `mode_chain` | Hyphen-separated Swiss purposes and broad trip modes |
| `activity_chain_json` | Swiss activities, retained trips and Swiss ground stages with coordinates, modes, times and purpose codes |
| `diary_trip_count`, `source_diary_trip_count` | Retained segment trip count and original full-diary journey count |
| `diary_date`, `person_weight` | Date and original respondent weight |

Separate Swiss segments are separate rows, so use `chain_id` as the unique
row key and `person_id` to link them. They are never joined by an invented trip
across an omitted flight. Use `airport_events` to match each detailed
mode/direction to its airport rather than pairing joined summary strings.
The respondent weight repeats across their segments; count distinct `person_id`
values when computing respondent-level totals.

The primary access mode is the ground mode with the greatest reported distance,
preferring motorized modes over access walking/cycling where present. `mode`
combines rail/bus/tram/ship as `pt`, while `mode_detailed` retains the specific
mode. Other distinctions include `own_car`, `company_car`, `rental_car`,
`carshare`, `car_driver`, `car_passenger`, `taxi`, and `ridehail`. Household-car
ownership is asserted only when `f51310a = 1`; a driver with unknown ownership
remains `car_driver`. Every stage's detailed mode remains in the JSON. The 2021
ride-hail category is taxi-like services such as Uber; 2015 has no separate
ride-hail code, so those users cannot be distinguished from the reported modes.

Employment uses the derived `f40800_01` status: codes 1–5 are employed; adult
`-99` means not employed. Student status uses a reported education activity
(code 32 in `f41000*`/`f41001*`) or apprenticeship (`f40800_01 = 5`). A reported
zero/other status without education is false. Unknown/skipped answers remain
null; children are not automatically assigned adult employment/student answers.

Times are seconds relative to the diary day and may exceed 86,400. A departing
segment ends with arrival at the airport, without a following flight leg or
airport departure time. An arriving segment begins at the airport before its
first observed egress departure. The 2015 source dates are month/day/year; 2021
dates are day.month.year. `survey_year` retains the edition even for a recorded
January date in the following year.

## Run and inspect the returned data

Add to an existing Synpp configuration:

```yaml
run:
  - data.microcensus.airports
config:
  data_path: /path/to/data
  microcensus_airports_radius_m: 3000
```

Options `microcensus_airports_2015_folder` and
`microcensus_airports_2021_folder` accept absolute paths or paths relative to
`data_path`. The stage reads these source files and returns the DataFrame directly.
It does not write Parquet, CSV, summaries or cache manifests into the data folder.
The former `microcensus_airports_folder` output option has been removed.

This is a normal Synpp stage: Synpp decides whether to reuse its cached result
or call `execute()`. When executed, the stage reads the two surveys and returns
the newly extracted DataFrame. There is no custom validation hook or separate
extraction cache. Provenance remains available in `result.attrs['summary']`.
No random sampling or population downsampling is applied to the donor library.

For example:

```python
import json
chains = context.stage('data.microcensus.airports')
donor = chains.iloc[0]
daily_stages = json.loads(donor['activity_chain_json'])['stages']
airport_events = json.loads(donor['airport_events'])
```

The current extraction contains **280 Swiss segments from 273 respondents** (206 segments from 2015,
74 from 2021), containing 280 airport ground events. These are unweighted
observations, mostly arrivals, and are not airport demand totals. They provide a
small resident donor library; the household survey is not representative of
foreign hotel visitors. Population integration remains separate work.
