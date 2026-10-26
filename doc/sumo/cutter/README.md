# Loaded network and regional scenarios

`sumo.simulation.load_network` loads the full scenario before cutting it. The
default method is native `marouter` **incremental macroscopic assignment**.
Demand is loaded in increments and route choice responds to flow-dependent
edge costs. A seeded, balanced integer realization retains its assigned route
distribution. There is no full-network traffic simulation and no `duarouter`
route search after assignment. `duarouter` only maps and validates endpoints
before `marouter`. The cutter uses those assigned paths and estimated entry
times, then the regional stage runs microscopic or mesoscopic traffic.

Set `sumo_loading_method: SUE` for native stochastic assignment, or explicitly
select `dua` for iterative routing and traffic simulation via `duaIterate.py`.
DUA starts from network speeds and updates routes with measured simulated
costs; MSA progressively reduces route-changing probability. Only that opt-in
method represents full-network queues and junction interactions during loading.

`sumo_modes` is shared by population plans, commercial trips, vehicle demand,
loading and connectivity checks. Its default is `[car, truck]`; **truck includes
LCV** (`vClass="delivery"`). Select `[car]` for private drivers, `[lcv]` for only
light commercials, or `[car, truck, lcv]` explicitly. The population tables retain
their source records; exports omit unselected movements. Person plans split at
omitted legs instead of connecting activities across missing travel. Passenger
trips do not create additional cars. `sumo_modes: null` retains the previous
all-mode export/query behavior; it does not create a transit fleet.

Demand preparation defaults to `exlude_external_population_in_sumo: true` and
`clip_trips_to_border_in_sumo: true`. The latter reuses/completes mode-choice road paths before
loading and cuts their actual crossings, with estimated border-entry times. See [Swiss-border demand clipping](../population/README.md#road-route-reuse-and-swiss-border-demand-clipping).
The routed regional cutter below still follows assigned roads and loaded times.

## Stages

| Stage | Result |
| --- | --- |
| `sumo.scenario.demand` | Selected individual vehicle loading trips and optional car-network queries |
| `sumo.scenario.network.routing` | Cached Arrow road geometry and per-class origin/destination reachability on the full assignment network |
| `sumo.simulation.load_network` | Full network, explicit routes with edge exit times, edge weights, `trip_metrics.arrow` and `loading_report.json` |
| `sumo.cutter.geometry` | Polygon union in the network CRS and local offset, preserving holes |
| `sumo.cutter.split_network` | Full network split at polygon boundaries, before loading |
| `sumo.cutter.network` | Region-only directed network and `edge_mapping.json` |
| `sumo.cutter.trips` | Every inside visit with adjusted departure and endpoint positions, plus `cut_report.json` |
| `sumo.cutter.scenario` | Copied regional network/routes, mapping, report and runnable `microscopic.sumocfg` |
| `sumo.simulation.run_microscopic` | Microscopic execution of the cut scenario, with outputs in its own cache |
| `sumo.simulation.run_mesoscopic` | Mesoscopic execution of the cut scenario, with outputs in its own cache |
| `sumo.run` | Selects one regional run stage using `sumo_simulation_model` |

Without a region, loading uses the original full network. With a region, its
dependency splits crossing edges **before** full-network loading. The eventual
entry departure comes from the recorded exit of the preceding outside edge;
the cutter need not assume uniform speed along a congested crossing edge.
Splitting preserves directed road shapes, lane access, permitted turns and
retained signal programs. It introduces an unregulated zero-radius node on each
crossing. These additional mesoscopic segment boundaries can slightly change
numerical loading results.

Native `netconvert` exports and rebuilds plain network definitions; Shapely/GEOS
clips the geometry and igraph computes components. No additional C++ extension
or installation in the project UV environment is needed. SUMO's standard
`cut_net.py` and `cutRoutes.py` work at whole-edge boundaries; fractional crossing
edges need the additional mapping used here.

## Reachable trip endpoints

Before assignment, `sumo.scenario.network.routing` indexes the exact full
network, including any regional boundary splits. Its `routing_edges.arrow`
contains road shapes and separate origin/destination masks for each simulated
vehicle class. The graph uses permitted lane-to-lane turns, rather than only
geometric proximity or shared junctions.

Origins must be able to reach the main strongly connected road core;
destinations must be reachable from it. One-way entry and exit branches remain
eligible for their respective roles. Small acyclic networks use their main
weak component and incoming/outgoing degree checks instead. No network edges
are deleted by this index.

Loading maps geographic endpoints onto eligible edges. Explicit endpoints are
preserved when eligible; otherwise they are rematched, retaining their travel
direction. Search expands from 500 m in 500 m steps.
`sumo_mapmatch_distance: null` (default) imposes no distance cutoff; a positive
value explicitly restores a maximum distance.
Swiss-border clipping also constrains replacement positions to the Swiss side.
This prevents a nearby road with no incoming turns from becoming an unreachable
destination. Trips, departures, vehicle types and source metadata are preserved.
`endpoint_mapping.json` records counts, expanded endpoints, maximum search
radius, maximum mapping distance and examples
of repaired edges; the loading report includes the same summary. An endpoint
with no suitable road anywhere on the supplied network fails explicitly; demand is never silently dropped.
Native `duarouter` still validates the resulting trips before macroscopic
assignment, and no route search is added after `marouter`.

## Cutting behavior

Set `sumo_region_geometry` to a polygon file readable by GeoPandas/GDAL, such as
SHP, GPKG or GeoJSON. `sumo_region_layer` selects a layer where necessary.
The file's CRS is transformed using the network's projection and `netOffset`.
For an unlabelled geometry, set `sumo_region_crs`; for a nongeographic test or
plain network without projection metadata, set `sumo_network_crs` to its
unshifted coordinate CRS. Polygon parts are unioned and holes retained.

The cutter follows each loaded route and retains its contiguous inside pieces:

- A crossing trip is included even when both endpoints lie outside the region.
- A trip wholly outside is omitted.
- A trip leaving and re-entering becomes separate vehicles, with suffixes
  `__cut_0`, `__cut_1`, etc. No route connects across the outside gap.
- Original person/trip metadata survives. `sumo_parent_vehicle_id` and
  `sumo_cut_visit` identify the source and visit; departure and arrival positions
  follow the retained portion of the route.
- Crossing vehicles enter on the best route-compatible lane at maximum allowed
  speed, following SUMO's route-cutting defaults; their actual boundary speed
  and lane are not inferred from an edge travel time.
- Departures are sorted by the loaded entry clock. Edge exit times describe
  the full-network loading; running the regional scenario recomputes its traffic.

Clipping follows each directed edge's reference shape. Lane widths can straddle
the polygon, as at any finite-width road boundary. Connectivity means the weak
graph of **permitted road turns**: one-way entry/exit boundaries prevent demanding
strong connectivity. By default `sumo_cutter_components: largest` retains the
largest such component and reports discarded edges. `error` fails on disconnected
geometry; `all` explicitly retains every component. Routes never bridge a removed
component or a forbidden turn. `sumo_cutter_min_edge_length` defaults to 0.01 m.

The trip report counts omitted trips, repeated visits and visits requiring
within-edge time interpolation. The standard boundary-split loading workflow
has no such interpolation; the low-level cutter also supports unsplit routes
and labels interpolated times explicitly.

## Loading settings and interpretation

| Option | Default | Meaning |
| --- | --- | --- |
| `sumo_loading_method` | `incremental` | Native macro `incremental` or `SUE`; simulation-based `dua` is opt-in |
| `sumo_loading_mesoscopic` | `true` | For `dua` only: mesoscopic loading; false uses microscopic loading |
| `sumo_loading_route_choice` | `msa` | `msa`, `gawron` or `logit`, for DUA |
| `sumo_assignment_iterations` | `50` | Maximum DUA iterations or native macro increments/iterations |
| `sumo_assignment_interval` | `3600` s | Aggregation of loaded edge costs |
| `sumo_assignment_scale` | `1` | Demand expansion; use 10 only for representative 10% demand |
| `sumo_loading_convergence_window` | `5` | DUA mean travel-time observation window |
| `sumo_loading_convergence_tolerance` | `0.01` | Relative standard deviation threshold; zero disables early stopping |
| `sumo_loading_require_convergence` | `false` | Fail when DUA misses this stability threshold |
| `sumo_begin` / `sumo_end` | `0` / `108000` s | All departures must be covered; DUA additionally requires completed trips |

For DUA, the convergence report includes iteration means, relative standard deviation
and `equilibrium_guaranteed: false`. Stable mean travel time alone does not prove
user equilibrium or route-level convergence. Assess iteration and aggregation
sensitivity before interpreting an assignment as settled. Undeparted or
unfinished DUA vehicles cause an error; extend the horizon or inspect deadlocks.
Teleportation is disabled.

Both macro and DUA expansion export native scale clones; their
`sumo_original_trip_id` identifies the original trip. The expansion factor must
be at least one to retain every original trip. Macroscopic route shares use
largest-remainder rounding and a configured-seed shuffle of vehicle identities;
each assigned route count differs from its fractional target by less than one
vehicle per distribution and departure slice. No new paths are calculated.
With a region, both macro methods use the boundary-split network. Boundary times
are **estimates** accumulated from loaded edge costs in each trip's departure
slice, with partial endpoint positions accounted for. They do not model queue
propagation between slices or demonstrate user equilibrium. The realization
report records this approximation. See [macroscopic cost assumptions](../config/README.md).

SUMO 1.28.0 `marouter` searches with its default vehicle class, even for typed
truck/LCV input. All exported routes are checked against their actual vehicle
class and permitted turns. An invalid native assignment fails explicitly;
use `sumo_loading_method: dua` when class restrictions require class-aware
assignment. The exporter does not reroute or silently modify assigned flows.

`trip_metrics.arrow` contains source IDs, original travel-time references,
loaded travel time and distance. DUA time measures actual departure to arrival;
its distance uses native simulated route length where available. Macroscopic
and regional cut distances sum travelled normal-edge lengths, with endpoint
position corrections; junction-internal connector lengths are excluded there.
Requested departure is retained in the route parameter `sumo_requested_depart`.
Source travel times are references, not imposed costs. Regional simulation
reproduces the extracted paths and entry demand, but cannot preserve outside
queues or downstream congestion beyond its boundary. Choose a sufficiently
large region when those effects matter.

## Running

Use the existing synthesis cache and working settings:

```bash
uv run -m synpp config_dib_10.yml --run sumo.simulation.load_network \
    --config sumo_assignment_scale 10

uv run -m synpp config_dib_10.yml --run sumo.cutter.scenario \
    --config sumo_assignment_scale 10 \
    --config sumo_region_geometry /path/to/zurich_region.gpkg
```

These commands can execute uncached synthesis and a full-network assignment.
Configure the region before loading to reuse its boundary-split result.
The local ignored `config_sumo_load.yml` selects the loading stage using the
current 10% synthesis configuration. The portable
[cut.example.yml](cut.example.yml) demonstrates existing network/trip inputs.
Use the returned bundle config with the isolated runtime's `sumo.run` helper.
Both regional presets default to `sumo_rerouting_probability: 0`, preserving
the extracted paths. Enable online rerouting explicitly if desired.

## Run the cut scenario

Set `sumo_simulation_model` to `microscopic` (default) or `mesoscopic` and run
`sumo.run`. Both choices consume `sumo.cutter.scenario` and generate a fresh local
config using the corresponding shared preset. The microscopic preset uses
`sumo_micro_*` settings; the mesoscopic preset uses `sumo_meso_*` settings.
`sumo_loading_mesoscopic` controls only explicitly selected full-network DUA
loading; it has no effect on the default macroscopic assignment.

```bash
uv run -m synpp config_sumo_load.yml --run sumo.run \
    --config sumo_region_geometry /path/to/zurich_region.gpkg \
    --config sumo_simulation_model microscopic

uv run -m synpp config_sumo_load.yml --run sumo.run \
    --config sumo_region_geometry /path/to/zurich_region.gpkg \
    --config sumo_simulation_model mesoscopic
```

Alternatively select `--run sumo.simulation.run_microscopic` or
`--run sumo.simulation.run_mesoscopic` directly. Those fixed stages use the named
model regardless of `sumo_simulation_model`. The portable
[run.example.yml](run.example.yml) shows the dispatcher with existing inputs.
These targets can build missing upstream demand, loading and cutter caches.

Each fixed stage returns `model`, `config`, `network`, `input_routes`, input
`count`, and output paths `tripinfo`, `edge_data`, `summary` and `routes`.
`simulation_report.json` records those paths. `count` describes the input demand;
inspect tripinfo for completed, unfinished and undeparted vehicles at `sumo_end`.
The shared additional statistics file and all standard simulation outputs are
written in the run stage cache. The cached cutter network and routes are read
without modification; switching models keeps separate execution caches.

The final run stages do not apply `sumo_assignment_scale` again. They simulate
the vehicles already present in the cut routes, including scale clones from
either loading method. No new sampling or expansion is applied by these stages.

The older `sumo.simulation.run` remains the runner for separately generated
`sumo.config.microscopic` / `sumo.config.mesoscopic` configs and explicit inputs;
use `sumo.run` for the complete cutter-to-simulation workflow.

Reusable scenario readers/writers live in `sumo.scenario.tools`,
`sumo.scenario.network.tools` and `sumo.scenario.population.writer`.
Spatial and temporal cutting helpers live in `sumo.cutter.tools`. Upstream
network, demand and loading caches are never edited.

References: [SUMO dynamic user assignment](https://sumo.dlr.de/docs/Demand/Dynamic_User_Assignment.html),
[assignment tools](https://sumo.dlr.de/docs/Tools/Assign.html),
[route-cutting tools](https://sumo.dlr.de/docs/Tools/Routes.html),
[plain network definitions](https://sumo.dlr.de/docs/Networks/PlainXML.html).
