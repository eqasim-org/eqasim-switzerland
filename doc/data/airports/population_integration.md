# Airport population integration and microcensus evidence

Airport plans are integrated through `synthesis.prepared.population`,
`synthesis.prepared.vehicles` and `synthesis.prepared.facilities`. Airport
integration is enabled by default (`use_airports: true`); set `use_airports: false`
to disable it. The example and working configurations both enable airports. Matching, ordinary activity chains, location
assignment and mode choice keep their existing implementations and caches.

## Population replacement

`synthesis.airports.trips.inject_airport_trips_in_population(population, trips)`
returns a new prepared `Population` without modifying its inputs. It:

1. Removes each selected resident's ordinary person, activity and owned-vehicle
   rows, then provides an airport agent with the **same person ID**. Residents
   were selected once, without replacement, from `synthesis.population.sampled`
   before matching. Selected residents missing from the final ordinary plans
   are restored as airport travellers; the number is logged.
2. Adds visitor agents with deterministic `airport_visitor_<trip_id>` IDs.
   IDs are checked against normal agents, selected residents and serialized
   freight/LCV IDs; collisions get deterministic numeric suffixes.
3. Supplies separate `airport_trips` and `airport_vehicles` tables, like the
   existing freight/LCV tables. Every movement has one plan, one ground leg and
   two activities. Resident purposes are `home` and `airport`; visitor purposes
   are `secondary_home` and `airport`, ordered by direction. An arrival and
   departure are independent passengers; no same-day return is invented.
4. Creates only the vehicle required by each road/bike plan and retains the
   preassigned mode. Passenger fleet exports remove replaced residents' old
   vehicles and add the new airport resources. Freight/LCV owner numbers have
   separate namespaces and are not removed when their numbers match residents.
5. Provides explicit facilities for both endpoints at their sampled coordinates.
   Java preparation requires activity facilities; it can therefore assign links
   without relocating airport activities to unrelated freight facilities.

`synthesis.prepared.households` stays unchanged: resident membership uses the
original ID, and temporary visitors have no invented resident household.
The minimal airport XML includes `isAirport`, `isVisitor`, airport code,
`subpopulation=airport` and the person's mode-to-vehicle mapping. Resident age
is retained because the Swiss Java configurator counts adults in households.
Full survey attributes are not required for these fixed airport plans.

`matsim.config.config` registers the airport subpopulation with `ReRoute` and
`KeepLastSelected`. It does not run discrete mode choice or activity rescheduling
on airport agents. Airport travellers use the ordinary default scoring parameters;
the pipeline removes any airport-specific scoring set left by older configs.

Negative departure seconds and times after midnight retain their offsets.
The initial activity has no start bound; the final activity has no end bound.
There is no artificial final activity end earlier than a late airport arrival.
Simulation time-window choices still determine which movements run.

## Limits and donor evidence

The model uses direct home/accommodation-airport chains. It does not add escorts'
return/empty legs, multi-day away states or intermediate activities. Drop-off,
taxi and ride-hail retain their observed access labels in the sampled data but
use the configured simulation mode mapping. See the [trip guide](../../synthesis/airports/trips.md).

[`data.microcensus.airports`](../microcensus/airports.md) supplies pooled
resident selection probabilities. The [chain analysis](activity_chains.md)
distinguishes transfer stops from later home-based outings; stochastic chain
extensions remain deferred. The following source inspection is historical
context, before the current cleaned donor filters.

## What the raw microcensus contains

Read-only inspection on 14 November 2026 covered the available 2015 files under
`<data_path>/microcensus/` and 2021 files under `microcensus/21/`. The former are
comma-separated, the latter semicolon-separated. Relevant files are:

| Table | Useful fields |
| --- | --- |
| `wege.csv` | `HHNR`, `WEGNR`, main mode, purposes, departure/arrival times; full journey |
| `etappen.csv` | `HHNR`, `WEGNR`, `ETNR`, `f51300`, stage purposes, coordinates, countries and times; access/flight/egress stages |
| `zielpersonen.csv` | Diary date `USTag`, respondent weight `WP`, demographics and home-related information |
| `wegeinland.csv` | Swiss domestic portions of journeys; relevant for cutting air journeys to their Swiss access legs |

The source codebooks were read directly from the local archives:

- `mz/MZMV2015_mit_Geo.zip`, workbook
  `2017-11-23-Datenbankstruktur_MZMV2015_mit_Geodaten_mit_Korrekturen.xlsx`.
- `mz/MZMV2021_mit_Geo.zip`, workbook
  `2023-06-05-Datenbankstruktur_MZMV2021_mit_Geodaten.xlsx`.

Both workbooks' `Etappen` sheets define stage modes and purposes. For 2015,
`f51300=17` is air travel; for 2021 it is `19`. Main trip mode `wmittel=1`
(2015) / `wmittel1=1` (2021) identifies an air-main journey, but that is not the
mode code used in the stage table. Stage `f52900=1` means transfer/change of
mode/parking, and `11` means return home or to accommodation. There is **no
separate airport activity-purpose code** in this purpose classification.

The current raw data contain:

| Survey edition | Air-main journeys | Respondents with air-main journeys | Air stages |
| --- | ---: | ---: | ---: |
| 2015 | 294 | 285 | 321 |
| 2021 | 106 | 103 | 113 |

These are unweighted diary observations, not annual passenger counts. Recorded
diary dates also include the following January (2016/2022); the edition label
alone should not be used as a strict calendar-date filter.

## Airport access chains found

An exploratory check used flight-stage endpoints within 3 km of the existing
configured airport reference points, in EPSG:2056. Within the same
`(HHNR, WEGNR)` journey, stages before that flight form a candidate access chain,
or stages after it form a candidate egress chain. Cases whose selected block
still contained another flight were excluded. The other ground endpoint's
country code was checked to be Switzerland (`8100`). No individual records,
addresses or respondent IDs are published here.

| Edition | Airport | Ground chains | To airport | From airport |
| --- | --- | ---: | ---: | ---: |
| 2015 | Geneva | 73 | 4 | 69 |
| 2015 | Zurich | 133 | 14 | 119 |
| 2021 | Geneva | 32 | 6 | 26 |
| 2021 | Zurich | 42 | 1 | 41 |

Examples include direct car-passenger access/egress, car-driver segments,
rail followed by bus/walk, and taxi access/egress. One to six ground stages occur
in the selected blocks. Respondents' surrounding daily journeys also remain
available, so this is evidence for both access sequences and daily-plan donors.

These are **candidate chains**, not a validated airport trip classifier:

- A circular search around the approximate airport reference point is not a
  terminal boundary. Swiss geocoding quality and municipality-centre imputation
  require review. A 1 km search misses many Zurich records; the number of
  raw flight endpoints grows substantially between 2 and 3 km there.
- The pronounced arrival/departure imbalance makes these records unsuitable for
  setting airport direction totals or an hourly demand curve. They can provide
  donor structures while the airport reports and flight snapshots retain control
  of totals and airport event times.
- These counts are before the pipeline's respondent/unknown-purpose filters.
  Not all candidates will be valid donors after cleaning and temporal checks.
- The household-resident survey is not a representative sample of foreign hotel
  visitors. Visitor activity chains need their own evidence or explicit proxies.
- The 2021 file has fewer observed air journeys. Pooling editions requires
  harmonizing mode codes, purposes and weights, not copying codes unchanged.

## Why the cleaned activity chains are insufficient by themselves

`data.microcensus.trips` maps main air mode to broad `pt`, initially retaining
`mode_detailed=plane`. It then marks a journey as `car_passenger` when any
contained stage is a car-passenger stage. Thus `mode_detailed == plane` alone
can miss air journeys with car-passenger access. The reader also collapses the
stage sequence into one main trip and maps airport transfers to generic purposes.
`data.microcensus.activity_chains` constructs its chains from that cleaned table;
it does not restore an airport stop.

The separate **`data.microcensus.airports` donor extraction stage** reads the raw
stage table and retains flight-stage evidence before these mappings. It records
airport access/egress events, retains the connected Swiss activities and ground transport sequence, excluding
flights and foreign diary portions, and provides resident departure/return donors. Matching
those templates to selected agents and anchoring the airport event to the
existing flight snapshot would avoid changing ordinary microcensus cleaning.

The immediate answer is: **yes, we have relevant observations and can build a
small resident airport-day donor library**, especially from 2015. We should not
expect a ready-made, representative visitor-chain library or derive demand
volumes from these few records. Donor extraction is now implemented in `data.microcensus.airports`; population
integration is implemented through the prepared stages described above.
