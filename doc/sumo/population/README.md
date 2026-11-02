# SUMO population and plans

SUMO stores population attributes and a person's plan together. Each `<person>`
inside a `<routes>` file contains metadata `<param>` elements and a connected
sequence of movements and activity `<stop>` elements. There is no separate
native demographic file that a plan references by person ID.

## Stages and contracts

| Stage | Cached result |
| --- | --- |
| `sumo.scenario.population.population` | Paths to `persons.arrow`, `activities.arrow` and `vehicle_trips.arrow` |
| `sumo.scenario.population.plans` | Absolute path to `population.rou.xml.gz`, containing people and their plans |
| `sumo.scenario.population.trips` | Absolute path to `vehicles.trips.xml.gz`, containing freight and commercial vehicle trips |

The preparation stage uses `synthesis.population.models.subscriptions`,
`synthesis.population.activities`, and `synthesis.population.spatial.locations`.
Like `matsim.scenario.population`, it chooses `mode_choice.run` when
`enable_mode_choice` is enabled, otherwise `synthesis.population.trips_with_departures`.
Final modes and corrected departures are joined by `person_id` and `trip_index`;
sampled arrival times remain unchanged. Upstream DataFrames are never modified.

`exlude_external_population_in_sumo: true` (the spelling is intentional) excludes
the external population from SUMO by default, independently of MATSim. Set it to
`false` together with `include_external_population: true` to include that source.
`include_cross_border` still selects the separate cross-border population stage. Source person IDs, scalar person attributes, activity
indices and trip IDs are retained. The `population_source` parameter records
`resident`, `external` or `crossborder`. Attribute names and category codes remain
those of the source tables; no MATSim Java attribute classes are written.
Cross-border chains use the shared [timing correction](../../data/cross_border/README.md)
to fit inbound travel after day start while preserving activity durations.

## Vehicle-demand performance

Population preparation converts external tables once through
`sumo.tables.as_polars`, performs joins and validation with Polars, and writes
`persons.arrow`, `activities.arrow` and `vehicle_trips.arrow` in its cache.
It returns only their paths. The full population is no longer serialized into
Synpp's pickle, and consumers use memory-mapped Arrow IPC files.

Vehicle preparation pairs adjacent activities in batches, including destinations
across batch boundaries, and projects coordinates in bulk. Demand sorts columns
in RAM with Polars and writes vehicle XML through the default C++/Arrow backend.
It avoids per-person DataFrames, per-person dictionaries, SQLite and per-trip
Python XML objects for population demand. The lxml fallback is selected with
`sumo_native_xml: false`.

See [Polars caches and native XML](../native/README.md) for contracts, compiler
requirements, thread settings and benchmark results. The first run rebuilds the
affected SUMO caches; running processes must be restarted to use the changes.

## Mode selection and validation

Exports use `sumo_modes: [car, truck]` by default; `truck` also selects LCV.
Unselected legs are omitted and mixed chains split into contiguous selected
segments, with `source_person_id` retaining the original person key. No movement
is invented across an omitted leg. `sumo_modes: null` exports all original modes;
in that case non-travellers remain with a single home stop. Duplicate identifiers,
duplicate activity keys, missing chains, missing coordinates, unsupported modes
and inconsistent activity/trip times raise errors instead of dropping people.
Vehicle demand resolves coordinates only for selected trip endpoints. Inactive
people and omitted legs do not require locations for vehicle export; a missing
location on a selected leg still raises an error naming its person and activity.
Source travel times are optional references. Missing, nonfinite or negative
arrival times, and arrivals before departure, leave that reference unset while
retaining the trip and its departure. SUMO computes the loaded travel time;
simulation departure times must still be finite and nonnegative.

External exports can repeat an activity after joining trip modes, including
both `car` and `car_loop` for the same activity key. Before combining populations,
the SUMO stage collapses those rows only when all other activity data agree and
the modes have the same base type. It retains the first source row and its
original mode, matching MATSim's ordering. Different times, locations or base
modes still raise an error; source caches are left unchanged.

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

## Road-route reuse and Swiss-border demand clipping

SUMO demand and simulation require `enable_mode_choice: true`. The default
national clipping also requires the Java car backend, because aggregate travel
times/distances do not identify border crossings. Use this configuration before
running mode choice:

```yaml
enable_mode_choice: true
prepare_sumo: true
mode_choice_car_routing_backend: java
exlude_external_population_in_sumo: true
clip_trips_to_border_in_sumo: true
```

`prepare_sumo` defaults to `false` in the mode-choice package. Setting it to
`true` makes the existing Java routing pass export itineraries, independently of
`route_population_once` (which still controls MATSim facility/route reuse).
Without pre-exported paths, SUMO preparation completes the missing routes; it
cannot recover paths from aggregate costs alone. Ordinary non-SUMO runs retain
their current defaults. External-population exclusion remains independent of
cross-border inclusion: disabling it adds the external source without changing
anyone's mode choice.

The preparation stages are:

| Stage | Result |
| --- | --- |
| `sumo.scenario.unrouted_demand` | Sorted selected demand in `demand.arrow`, plus vehicle-type definitions; no XML trip export yet |
| `sumo.routing.network` | Directed MATSim road shapes, modes, lengths and speeds in `road_links.arrow` |
| `sumo.routing.paths` | Reused/completed itineraries in `road_routes.arrow`, plus `routing_report.json` |
| `sumo.scenario.border` | Swiss polygon in LV95 WKB |
| `sumo.scenario.selected` | Original domestic and crossing partitions; outside-to-outside trips removed |
| `sumo.scenario.clipped` | Clipped originals, fixed border anchors and `border_clipping.json` |
| `sumo.scenario.scaled` | Retained demand expanded to 100%, before SUMO assignment |
| `sumo.scenario.demand` | Sorted expanded vehicle trips in Arrow and SUMO XML |

`sumo.scenario.selected` uses vectorized endpoint containment on the original
sample: domestic trips pass through, outside-to-outside trips are dropped
(including transit), and only inbound/outbound trips need road-path clipping.
`sumo.scenario.clipped` clips these originals before `sumo.scenario.scaled`
expands the retained demand. Boundary anchors stay fixed during expansion.

Mode-choice paths are matched by person ID, trip index, car/truck mode and
endpoints/departure (2 m and 1 s tolerances). Missing or changed requests are
routed together by the same Java router on its extended network, including
selected cross-border/external trips and commercial demand. This completion
pass performs road routing only, with macroscopic reassignment disabled; it does
not run mode choice again or modify source modes. Its router process is stopped
before SUMO preparation continues. The shared network reader cache is used and
corresponding directed detailed curves are retained.

For inbound/outbound originals, the national cutter uses actual link paths and
the precomputed border-link intersection fractions. Domestic visits are retained;
leaving and re-entering creates IDs ending in `__swiss_1`, etc. Trips classified
as domestic skip this path inspection, while outside-to-outside trips are removed
before routing. Source/person/trip identifiers remain attached. Invalid crossing
itineraries cause explicit errors rather than a straight-line fallback. Final
expanded demand shares the native bulk XML writer and departure sorting.
Person-plan XML from `population.plans` remains a separate, untrimmed export.

Entry/exit locations follow the detailed road geometry. The itinerary export
provides a total road travel time, not individual link clocks. Entry times are
therefore **estimates**, obtained by distributing that total over the traversed
links in proportion to their free-flow costs, including partial endpoint links.
A retained trip departs at its original departure plus the estimated time to
its entry. This preserves the routed total, including its aggregate congestion
effect, but does not reconstruct congestion separately on each foreign link.
The subsequent SUMO assignment computes loaded domestic routes/costs, and the
Lausanne/other regional cutter uses those assigned roads and loaded entry times.
MATSim's provisional domestic path is used to determine national visits; SUMO
may choose another domestic path during assignment.

For a single-link itinerary, Java exports the complete directed link without
within-link activity positions. If projecting the activity coordinates would
reverse or collapse that traversal, the cutter clips the exported full link
using its routed travel time. It records these cases in
`border_clipping.json` as `full_link_endpoint_fallbacks`. This preserves the
exported road direction; exact within-link activity positions are unavailable
in these cases.

Kept endpoints are matched onto nearby SUMO edges on the Swiss side, preserving
travel direction and passenger/truck/delivery permissions.
`sumo_mapmatch_distance: null` (the default) removes the distance cutoff. MATSim IDs are never treated as SUMO
IDs. Lane shapes with zero geometric length cannot define a heading and are
excluded from endpoint matching; their edges remain in the routing network.
Matching starts at 500 m and expands in 500 m increments until a valid road
is found. The distance-plus-heading score bounds the remaining candidates,
and nearest-road shortcuts skip provably empty searches. A positive
`sumo_mapmatch_distance` explicitly restores a maximum distance. Direction, vehicle access and Swiss-side checks
still apply. The demand stage logs network indexing and clipping progress.
Road-path clipping also runs in spawned worker processes. By default,
`sumo_demand_workers: null` uses the configured `threads`, capped to roughly one
worker per 10,000 input trips to avoid small partitions. Set a positive integer
to override this cap (the allocation's `threads` limit still applies), or `1`
for serial execution. Each worker reads an Arrow trip partition and only its
referenced MATSim road geometries. The parent performs final endpoint matching
with one SUMO network/index, merges the reports and sorts all departures before
export. Workers exchange cache paths, and temporary Arrow parts are cleaned up.
Trip IDs, timing, output XML and report counts are independent of worker count.
Without a cutoff, distance alone cannot reject an endpoint. A network with
no road satisfying vehicle access, direction and Swiss-side constraints still
fails explicitly after the finite network has been exhausted; it cannot loop
forever or silently drop demand. Supplied edge/XY endpoints are resolved lazily through
the georeferenced SUMO network. Clipped inputs with `via` constraints are rejected
because arbitrary waypoints cannot be preserved by this OD routing interface.

Route export needs the compatible Java runtime pinned by `matsim.runtime.eqasim`.
The old bundled `switzerland-2.0.0-cluster.jar` lacks the required flags. The SUMO
configuration uses `eqasim_path: ""` and the pinned commit so Synpp builds and
caches the compatible runtime; the existing JAR is left untouched.

All table stages return cache paths, not DataFrames. Geometry classification is
precomputed per link, and the route-to-demand join is performed once, rather
than rebuilding an entire route lookup per batch. Only changed trip payloads
are rewritten; wholly domestic trips retain the native Arrow/C++ writer path.
