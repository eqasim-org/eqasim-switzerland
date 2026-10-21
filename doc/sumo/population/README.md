# SUMO population and plans

SUMO stores population attributes and a person's plan together. Each `<person>`
inside a `<routes>` file contains metadata `<param>` elements and a connected
sequence of movements and activity `<stop>` elements. There is no separate
native demographic file that a plan references by person ID.

## Stages and contracts

| Stage | Cached result |
| --- | --- |
| `sumo.scenario.population.population` | Dictionary with copied `persons` and `activities` DataFrames |
| `sumo.scenario.population.plans` | Absolute path to `population.rou.xml.gz`, containing people and their plans |
| `sumo.scenario.population.trips` | Absolute path to `vehicles.trips.xml.gz`, containing freight and commercial vehicle trips |

The preparation stage uses `synthesis.population.models.subscriptions`,
`synthesis.population.activities`, and `synthesis.population.spatial.locations`.
Like `matsim.scenario.population`, it chooses `mode_choice.run` when
`enable_mode_choice` is enabled, otherwise `synthesis.population.trips_with_departures`.
Final modes and corrected departures are joined by `person_id` and `trip_index`;
sampled arrival times remain unchanged. Upstream DataFrames are never modified.

`include_external_population` and `include_cross_border` select the corresponding
existing population stages. Source person IDs, scalar person attributes, activity
indices and trip IDs are retained. The `population_source` parameter records
`resident`, `external` or `crossborder`. Attribute names and category codes remain
those of the source tables; no MATSim Java attribute classes are written.

Exports use `sumo_modes: [car, truck]` by default; `truck` also selects LCV.
Unselected legs are omitted and mixed chains split into contiguous selected
segments, with `source_person_id` retaining the original person key. No movement
is invented across an omitted leg. `sumo_modes: null` exports all original modes;
in that case non-travellers remain with a single home stop. Duplicate identifiers,
duplicate activity keys, missing chains, missing coordinates, unsupported modes
and inconsistent activity/trip times raise errors instead of dropping people.

## Export

From the repository root:

```bash
uv run -m synpp config_dib_10.yml \
    --run sumo.scenario.population.plans \
    --working-directory cache_sumo
```

This selects population export instead of the configuration's normal MATSim
output target. It can execute uncached synthesis dependencies. It does not build
the SUMO network, install SUMO or launch a simulation. A downstream stage can
register this stage and load its returned path with `context.stage(...)`.

Export freight and delivery traffic separately:

```bash
uv run -m synpp config_dib_10.yml \
    --run sumo.scenario.population.trips \
    --working-directory cache_sumo
```

This stage conditionally reads `synthesis.freight.trips` with `use_freight` and
`synthesis.lcv.trips` with `use_lcv`, also respecting `sumo_modes`. The identifiers are `freight_<agent_id>` and
`lcv_<trip_id>`; the source identifier is also stored as a parameter. It declares
SUMO vehicle types with `vClass="truck"` and `vClass="delivery"` respectively.
Original freight vehicle classifications remain parameters; detailed vehicle
dimensions, emissions and fleet calibration are not inferred. It writes trips
in departure order and does not invent freight passengers or arrival times.

## Plan representation

| Source mode | SUMO element |
| --- | --- |
| `walk` | `<walk>` |
| `car` | `<personTrip modes="car">` |
| `bike` / `bicycle` | `<personTrip modes="bicycle">` |
| `pt` | `<personTrip modes="public">` |
| `car_passenger` | `<ride lines="ANY">` by default |
| `taxi` | `<personTrip modes="taxi">` |

Each activity becomes a stop with `actType` and an absolute `until` time.
Movement parameters preserve the original mode, trip index, optional trip ID,
departure, and sampled arrival. Actual simulated arrivals are determined by
SUMO; the arrival parameter does not impose a travel time. Source activity
metadata is attached to its stop. Person departures are sorted for SUMO loading.

Private vehicle personTrips use SUMO's automatic vehicle generation. Walking
remains an available alternative in SUMO's intermodal personTrip routing, so
these are mode requests rather than an enforcement of estimated mode shares.
PT requires transit vehicles, stops and schedules. Rides require supplied
vehicles stopping at the pickup and destination. No passenger is converted
into an additional driver. An optional `passenger_vehicle_id` on a source trip
sets the ride's vehicle reference. `sumo_car_passenger_mode: taxi` explicitly
selects taxi requests instead; a taxi fleet and dispatch setup are still needed.

Modes ending in `_loop` retain their full source mode in parameters and use the
corresponding base movement type. No intermediate loop route is invented from
coincident endpoints; loop routing needs a later implementation.

## Coordinates and runtime

Demand is written with native `fromLonLat` / `toLonLat` coordinates; stops use
`lon` / `lat`. The writer converts the source CRS to WGS84 with explicit x/y
axis order. SUMO applies its own network projection and `netOffset` when loading
these coordinates. This avoids treating EPSG:2056 coordinates as already offset
network coordinates and lets the same demand be used with another network.
Coordinates are not snapped or clipped during export. The target network must
cover the activities and permit their modes. In particular, the current Swiss
road-only extract may need extending for foreign activities and active modes.

Full daily person plans can be loaded directly by SUMO; a small two-car-leg
fixture is tested this way. In SUMO 1.28.0, offline `duarouter` can emit duplicate
automatically generated private-vehicle IDs for repeated car personTrips.
Do not assume its full-day output can be simulated unchanged. Route generation
and service/fleet preparation are separate work from this demand export.

## Options and code organization

| Option | Default | Purpose |
| --- | --- | --- |
| `sumo_modes` | `[car, truck]` | Shared export/loading selection; truck includes LCV; null keeps all modes |
| `sumo_population_crs` | `EPSG:2056` | CRS shared by input activity and home coordinates |
| `sumo_population_end_time` | `108000` seconds | End time for a final/non-traveller activity without an end time |
| `sumo_car_passenger_mode` | `ride` | Preserve rides, or explicitly request `taxi` |
| `population_compresslevel` | `1` | Shared gzip compression level |

`tools.py` holds reusable table preparation, identifier and time validation,
coordinate conversion and deterministic binary/gzip streams.
`writer.py` defines the streaming `PopulationWriter`, with methods for person
contexts, generic parameters, activity stops, personTrips, walks, rides, vehicle
types and vehicle trips. It writes one person at a time, escapes XML through
`lxml`, supports plain XML or gzip by filename, and does not import or install
SUMO Python packages into the project environment.

References: [SUMO persons and plans](https://sumo.dlr.de/docs/Specification/Persons.html),
[intermodal routing](https://sumo.dlr.de/docs/IntermodalRouting.html),
[generic parameters](https://sumo.dlr.de/docs/Simulation/GenericParameters.html).
