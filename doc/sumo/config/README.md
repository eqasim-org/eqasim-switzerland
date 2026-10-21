# Simulation configurations and loaded assignment

The presets target SUMO **1.28.0**. They are research starting points, not a
Zurich-calibrated model. Use observed counts, travel times, turning movements,
vehicle composition and actual signal plans to calibrate a city scenario.

## Stages

| Stage | Result | Executable |
| --- | --- | --- |
| `sumo.config.microscopic` | `microscopic.sumocfg` | `sumo` |
| `sumo.config.mesoscopic` | `mesoscopic.sumocfg` | `sumo` |
| `sumo.config.macroscopic` | `macroscopic.marocfg` | `marouter` |
| `sumo.config.loaded` | Dictionary containing assignment and loaded-routing configs | `marouter`, then `duarouter` |
| `sumo.config.config` | Dictionary collecting micro, meso, macro and loaded configs | Config generation only |
| `sumo.simulation.run` | Simulation output paths; selects micro/meso with `sumo_simulation_model` | Runs `sumo` |
| `sumo.router.assignment` | Loaded weights, assigned flows, query routes and `trip_metrics.parquet` | Runs both assignment and routing |
| `sumo.simulation.load_network` | Full assigned routes with edge exit times, metrics and loading report | Native macro assignment by default; simulation-based DUA opt-in |
| `sumo.simulation.run_microscopic` | Microscopic execution using `sumo.cutter.scenario` | Runs `sumo` |
| `sumo.simulation.run_mesoscopic` | Mesoscopic execution using `sumo.cutter.scenario` | Runs `sumo --mesosim` |
| `sumo.run` | Selects one of the two regional run stages through `sumo_simulation_model` | Delegates to selected stage |

The first three config stages return an absolute config filename. The loaded
config stage returns `assignment_config`, `routing_config`, `weights`, `routes`
and expected query `count`. Config stages do not launch simulations or loaded
assignment. The macroscopic stages first map demand endpoints and validate
connectivity using `duarouter`.

The current `sumo.simulation.run` obtains its selected config from
`sumo.config.config`, which prepares all four products. For the distinction
between that runner and regional `sumo.run`, see the
[stage flowcharts](../WORKFLOW.md).

Execution stages copy their configs and generated additional files to their own
cache directories. Inputs are read-only and all execution outputs stay in the
execution stage; cached config producers and cached simulations are preserved.

## Microscopic urban preset

The default uses Krauss car following, a **0.5 s** integration step, **1 s** action
interval, and ballistic integration. These are explicit modeling choices that
retain a conventional decision interval while using smoother position updates.
Vehicle-type settings can override the defaults. If you change the action
interval or a vehicle's `tau`, check their consistency; changing these parameters
changes behavior and capacity. [SUMO time and integration settings](https://sumo.dlr.de/docs/Simulation/Basic_Definition.html),
[car-following parameters](https://sumo.dlr.de/docs/Car-Following-Models.html).

It records junction collisions as warnings, disables jam teleportation, and
retains unfinished and undeparted trips in tripinfo. The default horizon is
30 hours. An unfinished trip is not a completed travel-time observation.
Tripinfo contains simulated travel time and route length; summary, vehicle-route
and 15-minute edge statistics are also written.
[Tripinfo output](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html).

Both simulation presets retain assigned paths by default
(`sumo_rerouting_probability: 0`). Enabling this probability models informed
drivers: observations update every 60 seconds over five observations and
rerouting is allowed every 300 seconds. Set `sumo_rerouting_probability` and
`sumo_rerouting_period` for that behavior explicitly. Online adaptation does
not establish user equilibrium.
[SUMO routing device](https://sumo.dlr.de/docs/Simulation/Routing.html).

For Zurich, supply a Zurich network and compatible demand. Guessed OSM traffic
lights are an initial network representation; load municipal signal programs
and transit infrastructure using `sumo_additional_files` when available. The
preset does not cut nationwide demand into a city or invent boundary traffic.

## Mesoscopic urban preset

The mesoscopic preset uses **meso-LTM**, explicit junction control, per-lane
queues, multiple turning queues, and the standard 98 m maximum segment length.
Signal travel-time and flow penalties are zero because the signal controller
itself is active; positive TLS penalties can replace that behavior.
LTM models upstream gap propagation and is suitable for urban queuing. Separate
lane queues retain lane/mode permissions. The native headway parameters remain
unchanged as an initial baseline. This model has less detail than microscopic
simulation, and passenger boarding durations and several lane-level outputs
are unavailable. [SUMO mesoscopic model and limitations](https://sumo.dlr.de/docs/Simulation/Meso.html).

## Macroscopic congestion and per-trip metrics

The default regional workflow (`sumo.run`) uses `sumo.simulation.load_network`:
map endpoints, assign traffic with `marouter`, realize its assigned route
distributions, cut those paths using loaded entry-time estimates, and run the
regional micro/mesoscopic simulation. There is no route search after `marouter`
and no full-network mesoscopic simulation by default. See the
[loading and cutter guide](../cutter/README.md) for defaults, integer route-share
realization, time-slice estimates and native vehicle-class limitations.

The separate **loaded-skims** workflow (`sumo.router.assignment`, explicitly
selected with `sumo.config.loaded`) performs these steps:

1. Prepare individual road demand and queries for every source trip.
2. Map geographic endpoints to connected SUMO edges, failing on invalid routes.
3. Run `marouter` incremental assignment: load demand in 50 equal increments,
   updating costs from assigned flow in hourly intervals.
4. Write final edge travel times and volumes to `loaded.edges.xml.gz`.
5. Route every query using those costs with `duarouter`, without adding the
   queries to road flow a second time.
6. Export loaded time in seconds and full-edge route length in metres for every
   query, retaining identifiers and source travel-time references.

`assigned.flows.xml.gz` describes the assignment's route distribution. Final
query routes are shortest-path skims on the resulting loaded network; their
distribution need not reproduce the assignment's mixed routes. They do not
constitute an additional loading iteration. Missing query outputs cause an
error instead of silently returning a partial metric table.

This uses SUMO's **native volume-delay functions**. Their costs increase with
volume/capacity and assignment revisits routes as flow changes. They are fixed
in the SUMO implementation, with capacity assumptions based on lane count,
speed and priority conventions. The preset enables `capacities.default` to
avoid treating imported OSM priorities as SUMO's legacy assignment road classes.
For the exact 1.28.0 branches, consult the pinned
[capacity and cost implementation](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/marouter/ROMAAssignments.cpp).

This supplies congestion-dependent macroscopic skims, rather than microscopic
queues or signal delay calibration. The implementation does not expose
arbitrary per-link capacity or classical BPR alpha/beta parameters. Classical
BPR commonly uses `t = t0 * (1 + 0.15 * (v/c)^4)`; it would require a separate
configurable cost implementation to use that exact equation here.
[Native macroscopic assignment](https://sumo.dlr.de/docs/marouter.html),
[FHWA discussion of BPR and local calibration](https://www.fhwa.dot.gov/Environment/air_quality/conformity/research/sample_methodologies/emismeth07.cfm).

The default assignment method is `incremental`. `SUE` is available through
`sumo_assignment_method`. `UE` is rejected: SUMO 1.28.0 falls back to SUE for
that value, so it must not be described as deterministic equilibrium.
[Native method selection](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/marouter/marouter_main.cpp).
Incremental loading is not an equilibrium-convergence guarantee. Investigate
sensitivity to interval size, iteration count and route choice assumptions.

### Which trips generate flow?

The shared default `sumo_modes: [car, truck]` selects private drivers and freight,
including LCV. With `sumo_modes: null`, car/taxi trips plus freight and commercial
vehicle trips generate road flow. Every person trip, including passenger, walk and PT trips,
receives a **car-network query** at its departure time. Those non-driving trips
do not create additional cars. Their returned times are hypothetical car-network
times, not estimates of walking, cycling or public-transport journey times.
Source modes remain in the metric table.

For iterative simulation loading and a polygon-based regional scenario, use
[the loading and cutter workflow](../cutter/README.md). It retains the inside
parts of crossing routes and their loaded boundary-entry clocks.

`sumo_assignment_all_as_car: true` instead loads every person trip as an extra
car. This is an explicit alternative-demand assumption. Supply actual bus and
service-vehicle trips if their contribution to congestion is needed; passenger
trip counts alone do not define a public-transport vehicle fleet.

For supplied `sumo_assignment_trip_files`, each native `<trip>` is a road vehicle
and contributes flow. Multiple files and vehicle-type definitions are supported;
persons, aggregate flows and scheduled-stop vehicle demand are rejected by this
individual-trip workflow. Duplicate trip IDs and conflicting type definitions
also raise errors.

### Existing travel times and sampled demand

Source per-trip travel times are retained as `reference_travel_time_s`. A total
trip time does not identify which edges incurred congestion and is not imposed
as a path cost. You can seed routing with previous **edge** costs through
`sumo_assignment_previous_weights`: a SUMO weight file containing interval
`begin`/`end` and `<edge id="..." traveltime="..."/>`. Use
`sumo_assignment_weight_adaption` in `[0, 1)` to retain some prior cost influence;
with zero, supplied weights seed route choice and new costs follow current flow.
Imported costs must use current network edge IDs.

`sumo_assignment_scale` expands demand counts before native assignment. Use 1
for complete demand and 10 for a representative 10% sample. Capacity stays at
the network scale. A shared factor assumes consistent sampling across resident,
foreign, freight and commercial inputs; verify this before interpreting levels
of congestion. In the standalone loaded-skims workflow the metric file contains
one row per original query. The regional loading workflow instead exports all
native scale clones and keeps their assigned paths through cutting and simulation.

Hourly slices are macroscopic approximations; queues and spillback are not
propagated between intervals. Routing does not extrapolate the last loaded
interval indefinitely. Unweighted times use native network travel times.
Loop endpoints do not define intermediate loop routes. Distances cover whole
mapped edges, rather than exact door-to-door distances.

## Running

The ignored local `config_sumo_loaded.yml` copies the data/scenario settings and
working cache from `config_dib_10.yml`, selects `sumo.router.assignment`, and sets
the demand expansion to 10 for its current 10% sample:

```bash
uv run -m synpp config_sumo_loaded.yml
```

Creating that file did not start an assignment. Running it can execute uncached
upstream synthesis and mapping stages. To generate only a native assignment
config, use `--run sumo.config.macroscopic`; use `--run sumo.config.loaded` for
both native recipes without executing the loaded assignment.

Portable examples with explicit existing input files are
[microscopic.example.yml](microscopic.example.yml) and
[loaded.example.yml](loaded.example.yml). To generate or execute the mesoscopic
variant from the city example:

```bash
uv run -m synpp doc/sumo/config/microscopic.example.yml --run sumo.config.mesoscopic
uv run -m synpp doc/sumo/config/microscopic.example.yml --run sumo.simulation.run \
    --config sumo_simulation_model mesoscopic
```

Main input overrides are `sumo_network_file`, `sumo_route_files`, and
`sumo_additional_files` for simulation, and `sumo_assignment_trip_files` for
assignment. When absent, the existing network and population stages supply
inputs. `sumo_mapmatch_distance` defaults to 1000 metres; it is a maximum mapping
distance, not permission to silently discard out-of-area trips.

`sumo_network_file` accepts either prepared SUMO XML or an OSM XML/PBF source.
The shared road-network stage detects its header and converts OSM before any
consumer uses it; relative inputs also fall back to `data_path/osm`.

Advanced native model options are section/option dictionaries in
`sumo_micro_options`, `sumo_meso_options` or `sumo_assignment_options`, for example:

```yaml
sumo_micro_options:
  processing:
    default.carfollowmodel: IDM
```

File destinations remain managed by the pipeline so execution outputs remain
inside the consuming stage. All profiles use the configured `random_seed`.
