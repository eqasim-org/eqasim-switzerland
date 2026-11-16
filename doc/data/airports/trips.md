# Sampled airport access trips

See the [airport workflow](README.md) for the complete dependency graph, source
inventory, cache behaviour and refresh checklist.

`data.airports.trips` returns one combined pandas DataFrame of daily passenger
movements for Geneva (`GVA`) and Zurich (`ZRH`), restricted to Swiss ground
endpoints. Foreign ground access is excluded before expansion and rounding.
One endpoint is the airport;
the other is sampled within the municipality assigned by the existing airport
OD stages. Coordinates use EPSG:2056. These are individual passenger movements,
not vehicle trips or activities attached to the synthetic population.

## Inputs and evidence

| Input | Use |
| --- | --- |
| `data.airports.geneva` | Municipal demand, direction and observed resident/visitor profile |
| `data.airports.zurich` | Municipal demand and direction |
| `data.airports.model` | Airport coordinates and the transferred Zurich resident/visitor mixture |
| `data.airports.locations` | Reusable checked home/hotel pools, cached separately from trip settings |
| `data.statpop.persons` | Full permanent main-residence population and home coordinates, independently of synthetic population sampling |
| `data.spatial.municipalities` | Historical ID mapping and polygons; sampled points must lie in their assigned municipality |
| `data.statent.statent` | Actual NOGA 551 hotel establishment coordinates; dormitories excluded |
| `airports/geneva/geneva_trip_sampling.csv` and sibling JSON | Reviewed Swiss-endpoint mode shares, tied to the supplied Geneva PDF by checksums |
| Airport flight snapshots | Scheduled flight times and existing check-in/clearance assumptions |

The Geneva airport-supplied 2025 survey, physical page 21, reports the following
shares for Switzerland as the ground endpoint country:

| Access mode | Share | Default effective speed (km/h) |
| --- | ---: | ---: |
| Drop-off / pick-up | 20% | 60 |
| Private car | 10% | 60 |
| Rail | 28% | 70 |
| Bus / tram / Aérobus | 23% | 30 |
| Taxi | 6% | 50 |
| Ride-hail | 6% | 50 |
| Other grouped modes | 7% | 40 |

These country-specific shares differ from the pooled mode shares on page 15.
Both airports use the page-21 Swiss shares, as requested. Applying Geneva's
shares to Zurich, and treating modes as independent of municipality, direction
and profile, are assumptions. `access_mode` retains the seven categories;
`mode` groups them as `car`, `pt` or `other`. The grouped `other` category is not
interpreted as walking. All speeds are configurable modelling assumptions,
not measurements from the report.

## Residents, visitors and location sampling

Endpoint country, residence and nationality are different concepts. A Swiss
endpoint can belong to a Swiss resident or an overseas visitor. Passport
nationality is unavailable and is not inferred.

`is_visitor` is a boolean **residence-profile proxy** requested for location sampling:

- `True`: visitor/incoming-other profile; sample a hotel.
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
precede integer rounding, sampling and unresolved accommodation; realized
located-trip shares may differ.

For Geneva, the existing `outgoing` profile is treated as resident travel,
including its return movement from the airport. `incoming_other` is treated as
visitor/circuit travel, including its movement back to the airport. This
preserves the survey's raked regional/profile margins. The underlying survey
profile shares on physical page 4 are 57% outgoing, 35% incoming, 4% other and
3% transit; the existing OD stage excludes transit and reconciles rounding.

Zurich has no measured equivalent profile split. It transfers the saved model's
Geneva Swiss visitor share, conditioning that share on the allocated municipality:

```text
P(visitor | municipality i, Zurich) =
    Geneva_Swiss_visitor_share * visitor_probability[i] / pooled_probability[i]
```

This keeps the resident/visitor spatial pattern consistent with the calibrated
municipality model. `visitor_probability` records this probability;
`traveller_type_source` distinguishes Geneva observations from Zurich transfer.
Realized counts vary reproducibly with the configured seed.

Residents sample homes, weighted by the number of permanent main-residence
STATPOP people at each coordinate. Administrative/unknown coordinates are
excluded using available geocoding flags and polygon checks. Visitors sample
uniformly among distinct NOGA 551 hotels in that municipality. Hotel room counts
are unavailable. Source tables are read from upstream stages without modifying
their caches.

`data.airports.locations` returns these two pools as `(homes, hotels)`. Both have
`municipality_id`, `x`, `y` and `weight`; hotels also have `enterprise_id`.
Home weights count people at the coordinate; each hotel has weight one. Pool
attributes record eligible/source resident counts and coordinate exclusions.
Changing timing or sampling settings reuses this stage's Synpp cache.

Visitors always sample hotels. If their assigned municipality has no usable
hotel, those movements go to `unresolved_demand.csv` with
`no_hotel_in_municipality`. There is no visitor-to-home fallback and no change
to the OD municipality. `airport_trips_missing_hotel` must be `unresolved`; the
previous `home` policy is no longer supported. Missing resident home pools are
also reported as unresolved.

Foreign regional ODs remain available in the upstream airport stages as source
accounting, but are **out of scope for this trip stage**. They are not expanded,
located or written to the unresolved-demand table. Only their excluded aggregate
counts are recorded in the summary. Passenger demand is not redistributed from
foreign regions to Switzerland. Located trips plus unresolved accommodation
counts conserve the sampled Swiss-endpoint source total exactly.

## Travel and departure times

```text
distance_m = euclidean_distance_m * airport_trips_detour_factor
travel_time_seconds = distance_m / (effective_mode_speed_kmh / 3.6)
```

The default detour factor is 1.35. For travel **to** the airport, sample a flight,
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
in the [Geneva](geneva.md) and [Zurich](zurich.md) timing guides. Drop-off
passenger counts do not create escort return trips; occupancy is not inferred.

## Configuration and output

Example inside the pipeline configuration:

```yaml
run:
  - data.airports.trips
config:
  data_path: /your/input/data
  random_seed: 518745
  airport_trips_day: weekday
  airport_trips_sample_rate: 1.0
  airport_trips_detour_factor: 1.35
  airport_trips_missing_hotel: unresolved
```

The airport stages retain their existing model and evidence prerequisites. Add
these options to your usual pipeline configuration rather than replacing its
required spatial settings with this abbreviated example.

| Option | Default |
| --- | --- |
| `airport_trips_day` | `all` |
| `airport_trips_sample_rate` | `1.0`; explicit fraction in [0, 1], independent of `input_downsampling` |
| `airport_trips_detour_factor` | `1.35`; at least 1 |
| `airport_trips_speeds_kmh` | Mapping of all seven access modes to the speeds above |
| `airport_trips_missing_hotel` | `unresolved`; visitors must use hotels |
| `airport_trips_mode_data` | `airports/geneva/geneva_trip_sampling.csv` |
| `airport_trips_folder` | `airports/trips` |

Each Swiss-endpoint source `daily_trips_integer` count is expanded exactly at sample rate 1.
Lower rates use deterministic largest-remainder rounding within each airport;
`sample_weight` is the inverse sampling rate. Randomness uses independent seeded
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

Signatures cover OD data, eligible homes/hotels, airport model, mode evidence,
flight snapshots, settings and sampling code. Unchanged inputs reuse the saved
trips; corrupted or missing output artifacts trigger regeneration. This does
not recalibrate the saved airport allocation parameters.

The Synpp validation hook checks output contents as well as source contents, so
this recovery also applies when Synpp would otherwise reuse its stage result.
