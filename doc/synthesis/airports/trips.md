# Sampled airport access trips

See the [airport workflow](../../data/airports/README.md) for the complete dependency graph, source
inventory, cache behaviour and refresh checklist.

`synthesis.airports.trips` returns one combined pandas DataFrame of daily passenger
movements for Geneva (`GVA`) and Zurich (`ZRH`), restricted to Swiss ground
endpoints. Foreign ground access is excluded before expansion and rounding.
One endpoint is the airport;
the other is sampled within the municipality assigned by the existing airport
OD stages. Coordinates use EPSG:2056. These are individual passenger movements,
not vehicle trips. With `use_airports: true`, prepared stages replace selected
residents' ordinary plans and add visitors as new agents.

## Inputs and evidence

| Input | Use |
| --- | --- |
| `data.airports.ods` | Combined Geneva/Zurich municipal counts, already scaled by the population's `input_downsampling`; see [scaling](../../data/airports/ods.md) |
| `data.airports.geneva.model` | Airport coordinates and the transferred Zurich resident/visitor mixture |
| `synthesis.airports.locations` | Eligible synthetic resident and accommodation pools, cached separately from trip settings |
| `synthesis.population.sampled` | Final pre-matching synthetic roster, including IDs, attributes and actual homes; already at the simulation sampling rate |
| `data.microcensus.airports` | Pooled 2015/2021 airport respondent attributes for coarse resident-selection preferences |
| `data.spatial.municipalities` | Historical ID mapping and polygons; sampled points must lie in their assigned municipality |
| `data.spatial.ovgk` | Current accessibility polygons used to classify home/hotel locations |
| `data.statent.statent` | NOGA 551 hotels and 559 other accommodation/dormitories, including student residences |
| `airports/geneva/geneva_trip_sampling.csv` and sibling JSON | Reviewed Swiss-endpoint mode shares, tied to the supplied Geneva PDF by checksums |
| `airports/geneva/geneva_regional_mode_shares.csv` and sibling JSON | Reviewed page-23 Geneva/Vaud/other-Swiss mode shares used to fit distance preferences |
| Airport flight snapshots | Scheduled flight times and existing check-in/clearance assumptions |

The Geneva airport-supplied 2025 survey, physical page 21, reports the following
shares for Switzerland as the ground endpoint country:

| Access mode | Share | Default effective speed (km/h) |
| --- | ---: | ---: |
| Drop-off / pick-up | 20% | 60 |
| Private car | 10% | 60 |
| Rail | 28% | 70 |
| Bus / tram / Aérobus | 23% | 30 |
| Taxi | 6% | 60 |
| Ride-hail | 6% | 60 |
| Other grouped modes | 7% | 60 |

These country-specific shares differ from the pooled mode shares on page 15.
Both airports use the page-21 Swiss shares, as requested. Applying Geneva's
shares to Zurich is an assumption. `mode_detailed` retains the seven categories;
`access_mode` is retained as an identical compatibility column. Travel-time
estimation continues to use these detailed categories and speeds.

`MODE_MAP` in `synthesis.airports.trips` converts them to the simulation's five-mode
vocabulary (`car`, `car_passenger`, `pt`, `bike`, `walk`):

| Detailed mode | Simulation `mode` |
| --- | --- |
| `private_car` | `car` |
| `rail`, `bus_tram_aerobus` | `pt` |
| `dropoff`, `taxi`, `ridehail` | `car` |
| `other` | `car_passenger` (assumed fallback) |

The 7% `other` category has no usable breakdown, so its passenger mapping is an
explicit assumption, not an observed share. No walk/bike share is invented.
Taxi, ride-hail and drop-off travellers take a car and generate road traffic.
Since separate taxi and ride-hail simulation modes are unavailable, these trips
use `car`. Escort return legs and taxi/ride-hail repositioning are not generated.
All speeds are configurable modelling
assumptions, not measurements from the report. The summary includes `mode_map`
and both detailed `access_modes` and grouped `simulation_modes` counts.

### Distance preferences and fixed totals

`synthesis.airports.modes` fits `preference = a + b * Euclidean_distance_km`
separately for each detailed mode. The three observations are the normalized
regional mode shares from Geneva survey page 23. Each region's representative
distance is its full-population Geneva-OD-weighted municipal-centroid distance
to GVA. If a region has zero source demand, the allocation model's regional
weights supply a fallback. Survey sample sizes and individual distances are
unavailable, so the three regional observations have equal weight. This is an
ecological approximation, not a passenger-level choice-model estimate.

The same coefficients apply to Zurich by assumption. Predictions clamp distance
to the three-region calibration range and preferences to [0.01, 1], avoiding
negative weights or unbounded extrapolation. They guide allocation only; page-21
Switzerland-wide shares determine each airport's integer mode targets using
largest-remainder rounding. There is no new demand scaling.

Allocation fills each airport's PT target from movements with A/B/C locations
first. Only if that capacity is insufficient does it use movements with D-only
locations, and only the required number of those. Eligibility is checked separately
for resident homes and visitor accommodation within each municipality. PT mode
quotas are split between A/B/C and D support, with weighted draws without replacement
inside each tier. Remaining road quotas are assigned across all remaining movements.
A municipality with no PT support may therefore use either `car` or `car_passenger`.
Municipal, directional and resident/visitor trip counts remain fixed.

When A/B/C/D support is sufficient, all seven airport-wide mode totals match
their rounded targets exactly. Otherwise, the PT shortfall transfers to all five
road categories in proportion to their source shares, including `other` mapped to
`car_passenger`. If the input has no road share, the explicit fallback is
`private_car`. Missing PT accessibility alone never discards demand. Visitor locations use replacement. Residents are distinct selected people;
people sharing a household can naturally share its home coordinate.

`summary.mode_distance_model` records coefficients, regional mean distances,
normalized observations, clipping rules and source metadata. Each airport's
`mode_assignment` includes `requested`, `feasible_target`, `assigned`,
`pt_eligible_trips_abc`, `pt_eligible_trips_d_only`, `pt_trips_using_d`,
`pt_target_shortfall`, `additional_road_trips`, and separate additional/forced car
and car-passenger counts. `margins_preserved` says whether every original mode
quota was met. These concern all sampled movements; `access_modes` describes
located movements, so completely missing accommodation may still cause a difference.
Each trip includes `pt_location_available`, `pt_access_tier` (`ABC`, `D`, `none`),
and `mode_assignment_distance_km` (municipal centroid distance).

The microcensus is not used for the mode-distance fit: its sparse airport sample
and arrival/departure imbalance add uncertainty, while the Geneva report supplies
the requested modal evidence directly. Its demographics are now used separately for resident selection below. Saved gravity-model coefficients remain unchanged. The small
linear fit is reused through Synpp and the signed trip exports; it is calculated
again only when the trip stage needs to regenerate its outputs.

## Direct activity chains

Residents use `home-airport` or `airport-home`; visitors use
`secondary_home-airport` or `airport-secondary_home`. The output columns
`origin_purpose`, `destination_purpose` and `activity_chain` make this explicit.
For visitors, `secondary_home` denotes the sampled hotel or dormitory;
`is_visitor` and `location_source` retain the residence profile and source. No intermediate stops or later outings are sampled,
and the two directional passenger totals do not imply linked same-day returns.

## Residents, visitors and location sampling

Endpoint country, residence and nationality are different concepts. A Swiss
endpoint can belong to a Swiss resident or an overseas visitor. Passport
nationality is unavailable and is not inferred.

`is_visitor` is a boolean **residence-profile proxy** requested for location sampling:

- `True`: visitor/incoming-other profile; sample a hotel or dormitory.
- `False`: resident/outgoing profile; sample a household/home location.

It does not establish Swiss citizenship. `traveller_type` retains the explicit
`resident`/`visitor` label, and `traveller_type_source` records its provenance.
The definition is also exported as `summary.is_visitor_definition`.

Among Geneva movements with Swiss ground endpoints, the source-derived shares
are **57.5606% resident (`is_visitor=False`)** and **42.4394% visitor
(`is_visitor=True`)**. They come from the supplied 2025 accessibility survey,
physical pages 4–12: profile totals (p. 4), resident and visitor regional/country
tables (pp. 5–6 and 8–9), and pooled margins (pp. 11–12). After reconciling these
rounded margins, the joint masses are 0.3626315084 for residents with Swiss
endpoints and 0.2673684916 for visitors with Swiss endpoints. Dividing by their
sum, 0.63, gives the conditional shares above. These are not nationality shares
or the unconditional 57/96 and 39/96 journey-profile fractions.

Zurich transfers this 57.5606%/42.4394% Swiss-endpoint mixture from the saved model;
it has no observed local resident/visitor split. Its municipality-specific
probabilities can differ from the nationwide mixture. These source shares
precede integer rounding and hotel-capacity constraints; realized
located-trip shares may differ.

The model first determines municipal OD counts, and `data.airports.ods` applies the
population sample fraction. The trip stage then assigns the resident/visitor
labels **after** those municipal counts are fixed:

1. Round the airport visitor target once from the source visitor share above,
   then apportion it between directions in proportion to their trip counts.
2. Distribute that target across municipalities in proportion to their number
   of usable NOGA 551 hotels plus 559 accommodation/dormitories.
3. Cap a municipality's visitors at its total OD count, redistributing any
   excess to other municipalities with hotels.
4. Mark all remaining movements as residents. A municipality with no usable
   hotel always contains residents only.

The weighting is accommodation establishment count alone at this step; distance and population have
already determined the municipality OD totals. Integer quotas are deterministic,
then individual visitor labels are sampled reproducibly within each municipality.
`visitor_probability` is the resulting municipal visitor quota divided by its
trip count. `municipality_accommodations` records this weight; `municipality_hotels` is retained as a compatibility alias for the same combined count.

This intentionally replaces the former municipality/profile posterior rule.
`source_profile` preserves the upstream survey/model profile for audit;
`profile`, `traveller_type` and `is_visitor` describe the new assignment. The
municipality, direction and airport trip totals do not change. The old observed
municipal profile split is not imposed on the generated labels.

`summary.by_airport.<airport>.visitor_assignment` records the target share,
target/assigned visitor counts and any unfilled target. If municipalities with
hotels cannot accommodate the requested visitor count within their fixed OD
counts, the attainable visitor count is used and the remainder is resident.

Residents are selected from **`synthesis.population.sampled`**, the roster consumed
immediately by every matching algorithm. It already includes population scaling,
sampling, age, sex, employment, household ID, licence and car availability. The
airport branch does not depend on matching or change its input. Swiss residence
is determined by valid Swiss municipal home coordinates, not citizenship.
Explicit home-only scenario residents and people already assigned a road
cross-border movement are excluded from the airport candidate pool.

`locations` retains one row per eligible synthetic person, validates their home
against the current municipality, and classifies its OVGK. There is no raw
STATPOP-coordinate draw and no aggregation of people sharing a home.

### Pooled demographic preferences

Deduplicate airport survey rows by `(survey_year, person_id)` so multiple Swiss
segments do not count one respondent more than once. Pool 2015 and 2021 with one
unweighted observation per respondent, without airport, direction or regional
models. Estimate separate marginal preferences for:

- Age groups `0-17`, `18-34`, `35-54`, `55-74`, `75+`.
- Sex (`0=male`, `1=female`, harmonized by the survey stage).
- Employed / not employed. Synthetic codes are `1=employed`, `2=unemployed`,
  `3=inactive`; the last two become false, never `bool(2)` or `bool(3)`.

Missing survey answers are ignored separately per attribute. Smooth each
marginal towards the eligible synthetic population's distribution with
`airport_resident_demographic_prior_strength: 20.0` pseudo-observations. Each
candidate receives a weight from raking the small national age×sex×employment
population cells to those three smoothed marginals. This uses the population's
joint support without fitting a sparse joint survey model, and avoids counting
age/employment correlations twice. Every supported candidate retains positive
weight. Missing population attributes form an explicit unknown category whose
baseline mass is preserved. Convergence and marginal error are reported;
structural zeros or fixed municipal quotas can prevent exact demographic matches.

### Selecting people and coordinates

After municipal counts and visitor labels are fixed, select resident agents
without replacement **across both airports and both directions** within their
municipality, using the demographic weights. Randomize trip slots within each
municipality to avoid favouring Geneva, Zurich or one direction when capacity
is scarce. Each selected person contributes their `person_id`, `household_id`,
`age`, `sex`, harmonized nullable `employed`, original `resident_employment_code`,
`is_student`, `driving_license`, `car_availability` and `statpop_person_id` when
available. Their actual home coordinate is retained in the trip endpoint.

One resident can receive at most one independent airport movement in this
stage. Too few residents produces
`insufficient_synthetic_residents_in_municipality` in the unresolved audit;
people are not reused and demand is not moved to a different municipality.
The original OD total still equals located plus unresolved movements.

Resident selection precedes mode allocation. PT availability uses **the selected
person's own home OVGK**, rather than the best location anywhere in the
municipality. A/B/C is used first, followed by D as needed and road fallback.
Mode allocation never relocates the selected resident to a different home.

Visitors continue to sample uniformly among NOGA 551 hotels and 559 other
accommodation/dormitories, including student residences. Their `person_id` and
resident attributes remain null: visitor agents are not invented or borrowed
from the resident population. The selected municipality, mode eligibility and
home/hotel/dormitory tags are preserved. Visitor locations may be reused.

`summary.resident_sampling` records survey respondent counts, smoothed marginal
targets, population frequencies, marginal ratios and raking convergence, eligible roster counts,
realized selected demographics and municipal shortages. These aggregate diagnostics
are part of the airport trip output; the microcensus stage still returns its
DataFrame normally and writes no separate survey exports.

`synthesis.airports.locations` returns these two pools as `(homes, hotels)`. Both have
`municipality_id`, `x`, `y`, `weight` and `ovgk`. The resident pool additionally
contains unique person IDs and pre-matching attributes; weights are demographic
selection preferences. The visitor pool has `enterprise_id` and
`accommodation_type`, with weight one per establishment. Pool attributes record
source/excluded counts and demographic model provenance.
Changing timing or sampling settings reuses this stage's Synpp cache.

Visitors sample hotels or dormitories; municipalities without either contain residents
and use homes. The former missing-hotel visitor exclusions are avoided by this
assignment. A missing resident home pool can still be reported as unresolved.
`airport_trips_missing_hotel: unresolved` is retained for compatibility; it does
not enable a visitor-to-home fallback.

Foreign regional ODs remain available in the upstream airport stages as source
accounting, but are **out of scope for this trip stage**. They are not expanded,
located or written to the unresolved-demand table. Only their excluded aggregate
counts are recorded in the summary. Passenger demand is not redistributed from
foreign regions to Switzerland. Located trips plus unresolved accommodation
counts conserve the sampled Swiss-endpoint source total exactly.

## Airport access points

Each airport has two reviewed physical endpoints in
`airports/<geneva|zurich>/data_from_websites/airport_access_points.json`:

| Airport | PT interchange centroid (latitude, longitude) | Road parking point (latitude, longitude) |
| --- | --- | --- |
| Geneva | 46.23219067, 6.11135865 | P1: 46.23015470, 6.10890920 |
| Zurich | 47.45027425, 8.56307367 | P2: 47.45079320, 8.56270800 |

PT points average the distinct main rail and terminal bus/tram parent stops in
the cached Swiss GTFS, avoiding platform duplicates. The stored components are
Geneva `Parent8501026`, `Parent8592775`, `Parent8592935`; Zurich `Parent8503016`
and `Parent8573205`. Freight/staff stops are excluded. The road coordinates use
OSM Geneva P1 node 767859777 and Zurich P2 way 28990522's bounding-box centre,
cross-checked against official airport parking guidance. They are representative
facility locations, not routed entry links or separate terminal-level entrances.

Each JSON records method, source links and local GTFS/OSM excerpts. Edit the
reviewed points explicitly when they change; ordinary runs are offline. The
configured `airport_trips_access_points` maps GVA/ZRH to these two JSON paths.
Their contents are included in trip signatures. Output `airport_access_point`
is `pt` or `road`, with projected `airport_x`/`airport_y`. Both ground directions
use the same two access points. Actual distance and time use the selected
mode-specific endpoint. The gravity-model reference points stay unchanged, so
this refinement does not move or refit the municipal allocation model.

## Travel and departure times

```text
distance_m = euclidean_distance_m * airport_trips_detour_factor
travel_time_seconds = distance_m / (effective_mode_speed_kmh / 3.6)
```

The default detour factor is 1.35. All road categories, including car passenger, taxi, ride-hail and other, default to 60 km/h. Rail remains 70 and bus/tram 30 km/h. For travel **to** the airport, sample a flight,
subtract the existing check-in buffer, respect the terminal opening assumption,
and subtract the estimated access travel time. For travel **from** the airport,
sample an arriving flight and add the existing clearance buffer; the estimated
travel time determines arrival at the sampled endpoint.

`departure_time` is local wall-clock seconds relative to the sampled flight day;
`arrival_time = departure_time + travel_time_seconds`. Negative values and values
above 86,400 are retained. `departure_day_offset` and `departure_time_of_day`
allow consumers to interpret midnight crossings. `flight_seconds`,
`airport_event_seconds`, `source_flight_id` and `source_date` preserve timing
provenance. Snapshot dates are evidence dates, not a simulated calendar date.

`airport_trips_day` accepts `monday` through `sunday`, `weekday`, `weekend`,
`all`, or Monday=0 through Sunday=6. Day groups average normalized calendar-date
curves equally. Zurich's archived snapshot is partial: grouped days average
available dates and unavailable named days raise an error. Selecting a day
changes timing, not average daily demand; measured demand by weekday is absent.

There is no network routing, congestion model or PT timetable here. Flight
weights, check-in and clearance distributions retain the limitations documented
in the [Geneva](../../data/airports/geneva.md) and [Zurich](../../data/airports/zurich.md) timing guides. Drop-off
passenger counts do not create escort return trips; occupancy is not inferred.

## Configuration and output

`airport_resident_demographic_prior_strength` defaults to `20.0`; it must be
finite and positive. This controls smoothing, not demand or population sampling.


Example inside the pipeline configuration:

```yaml
run:
  - synthesis.airports.trips
config:
  data_path: /your/input/data
  random_seed: 518745
  input_downsampling: 0.1
  airport_trips_day: weekday
  airport_trips_detour_factor: 1.35
  airport_trips_missing_hotel: unresolved
```

The airport stages retain their existing model and evidence prerequisites. Add
these options to your usual pipeline configuration rather than replacing its
required spatial settings with this abbreviated example.

| Option | Default |
| --- | --- |
| `airport_trips_day` | `all` |
| `airport_trips_detour_factor` | `1.35`; at least 1 |
| `airport_trips_speeds_kmh` | Mapping of all seven access modes to the speeds above |
| `airport_trips_missing_hotel` | `unresolved`; visitors must use hotels/dormitories |
| `airport_trips_mode_data` | `airports/geneva/geneva_trip_sampling.csv` |
| `airport_trips_regional_mode_data` | `airports/geneva/geneva_regional_mode_shares.csv` |
| `airport_trips_access_points` | GVA/ZRH maps to each airport’s `data_from_websites/airport_access_points.json` |
| `airport_trips_folder` | `airports/trips` |

Each `daily_trips_integer` count from `data.airports.ods` is expanded exactly
once. Demand scaling and integer rounding happen upstream using
`input_downsampling`; this stage does not apply another fraction. Remove the
former `airport_trips_sample_rate` setting. `sampling_rate` and `sample_weight`
are carried from the scaled ODs, with weight equal to the inverse sampling rate.
Randomness uses independent seeded
streams for traveller profile, endpoint, mode and timing. Changing speed or
detour assumptions therefore does not change sampled modes or flight choices.

The default output folder is `<data_path>/airports/trips/`:

| File | Content |
| --- | --- |
| `trips.parquet` | Located passenger movements returned by the stage |
| `unresolved_demand.csv` | Swiss-endpoint counts with unavailable accommodation locations |
| `summary.json` | Counts by airport, profile, mode and location source, input evidence and assumptions |
| `signature.json` | Input signature and output checksums |

Trip IDs are unique across airports; `od_id` links each movement to its input
OD row. Origin/destination coordinate columns are `origin_x`, `origin_y`,
`destination_x`, `destination_y`, alongside regional and municipal IDs. The
airport-side municipality ID is null, with the airport identified by `airport`
and its corresponding region ID. The returned DataFrame also carries the
summary and unresolved file path in `attrs`.

For each airport, `summary.json` reports `sampled_trips_total`, located and
unresolved counts, and `unresolved_reasons`. `sampled_trips_total` concerns Swiss
endpoints only. `in_scope_daily_trips` and `in_scope_daily_trips_integer` report
the Swiss-endpoint source demand; `excluded_foreign_daily_trips` and its integer
counterpart describe the discarded foreign scope. `resident_profile_share`,
`visitor_profile_share` and `profile_share_source` report the population mixture
and its provenance before missing-accommodation exclusions. `timing_coverage` records the selected
calendar dates, flight count, covered/missing day names and `complete_day_group`
for each direction. An incomplete weekday average is explicitly identified.

The input/in-scope counts now refer to the **scaled** population sample.
`full_population_daily_trips` records the original Swiss-endpoint expectation,
and `sampling_rate` records `input_downsampling`. Foreign excluded aggregates
remain full-population source-accounting figures. Located and unresolved counts
sum exactly to the scaled integer total, including at rates zero and one.

Signatures cover OD data, eligible homes/hotels, airport model, mode evidence,
flight snapshots, settings and sampling code. Unchanged inputs reuse the saved
trips; corrupted or missing output artifacts trigger regeneration. This does
not recalibrate the saved airport allocation parameters.

The Synpp validation hook checks output contents as well as source contents, so
this recovery also applies when Synpp would otherwise reuse its stage result.

## Migration and cache contract

Use `synthesis.airports.trips` instead of the removed `data.airports.trips` stage.
OD stages and their files remain under `data.airports`; traveller selection,
location sampling and mode allocation live under `synthesis.airports`. The
default signed exports remain at `<data_path>/airports/trips/` for compatibility.
The new trip schema/version invalidates earlier coordinate-only results. Input
signatures include resident IDs/attributes/home coordinates, demographic weights
and aggregate model provenance, accommodation, OD tables, mode evidence and
airport point files. The population sampling fraction is not applied a second time to demand or the candidate roster.

## Injecting airport plans into the scenario

Airport integration is enabled by default (`use_airports: true`). Set
`use_airports: false` to disable it.
`synthesis.prepared.population` calls
`inject_airport_trips_in_population(population, trips)` in this module. The
function leaves upstream rosters and matching outputs unchanged, removes each
selected resident's ordinary prepared plan and vehicles, and supplies a minimal
airport plan with the same person ID. Visitors become new agents with
collision-checked `airport_visitor_<trip_id>` IDs.

Each plan has one leg and two purposes: `home`/`airport` for residents and
`secondary_home`/`airport` for visitors, ordered by direction. This purpose
convention is also used by the returned trip DataFrame. Prepared facilities
and vehicle stages supply their matching resources. MATSim writes the optional
airport tables alongside freight/LCV and uses a separate airport subpopulation
with fixed modes/times and route replanning only. Resident age is retained for
Java household adult counts; detailed ordinary agent attributes are unnecessary.

See [population integration](../../data/airports/population_integration.md) for
IDs, households, resource handling and timing limits.
