# Synthetic-population mode choice

This package adapts the tour-choice workflow from `origin/eqasim-mode-choice`
(commit `ab472acc1a7c3bbe23947650d42348bdd4f04b36`) to the current population
schema and `dmc` model, including its log-ratio ASC calibration. It contains no
estimator or skim-matrix reader. Car routing supports Java and Pandana.

## Running

`config_dib_10.yml` is the pipeline configuration. Run from the repository root:

```sh
uv run -m synpp config_dib_10.yml
```

The mode-choice options in the configuration are:

- `enable_mode_choice: true`: both `synthesis.output` and
  `matsim.scenario.population` consume the selected modes.
- `estimate_dmc: false`, `dmc_mode_parameters_path`, and
  `dmc_cost_parameters_path`: explicit paths to the existing fitted artifacts.
  Both paths must be supplied together. They take precedence over estimation,
  are checked for existence, and their modification times invalidate the cache.

Supply the existing fit through the two parameter-path options above.
JAX and Biogeme write the same parameter schema; either export works. The files
are read without modification, including their calibrated global and cantonal
constants. No estimator is imported during population mode choice.

`mode_choice.run` returns the original pandas trip table with modes replaced by
`(person_id, trip_index)`, retaining all other columns and rows. It also writes
`choices.parquet` in its Synpp stage directory. Location synthesis and departure
optimization continue to use the original `synthesis.population.trips` and
`synthesis.population.activities` stages. A separate
`synthesis.population.trips_with_departures` stage resolves optional departure
corrections before mode choice, avoiding a circular dependency. Add `synthesis.output` or `matsim.output` to the configuration's `run` list when those
outputs are required, or add `mode_choice.run` to generate choices alone.
The mode-choice stage does not run MATSim.

## Population alpha calibration

Mode utility always adds the two supplied constants:

```text
utility(person, mode) = other utility terms
                      + global_mode_ASC[mode]
                      + canton_mode_ASC[person.residence_canton, mode]
```

Global constants use `car.alpha_u`, `pt.alpha_u`, `bike.alpha_u`, `walk.alpha_u`
and `cp.alpha_u`; additive canton constants use the existing `swissCanton.car`,
`swissCanton.pt`, `swissCanton.bike`, `swissCanton.walk` and `swissCanton.cp` maps.
The residence canton comes from the person's canton, not a trip's destination.
Both the default and estimated `dmc.params` sources retain these maps, including
when population calibration is disabled. A missing canton offset defaults to zero.

Options under `config:`:

```yaml
mode_choice_calibrate_alphas: true       # false keeps all supplied constants
mode_choice_calibration_level: global   # global or canton
mode_choice_calibration_max_evaluations: 50
mode_choice_calibration_tolerance: 0.0001
```

| Setting | Parameters adjusted | Targets |
| --- | --- | --- |
| Calibration disabled | None | No calibration targets loaded |
| `global` | Global mode ASCs; canton offsets stay fixed | Global survey mode shares |
| `canton` | Each residence canton's mode offsets; global ASCs stay fixed | That canton's survey mode shares |

PT is the fixed reference in both cases: its global constant and canton offsets
remain at their supplied values. Other fitted coefficients also remain fixed.
Canton calibration starts from the existing offsets, so it can refine defaults
or previously calibrated estimates without discarding them. The package default
level is `global`. These options are separate from MATSim's
`calibrate_alphas_in_matsim` and `alphaCalibration.level` settings.

Targets come from the two CSVs returned by `data.microcensus.shares`: global
`mode,mode_share` rows or wide `canton_name` rows with a column for each mode.
Survey shares use person weights and are normalized after rounding. Simulated
shares count chosen trips longer than one metre, grouped by residence canton
when requested. Preserved special/open tours and separately injected external
populations are excluded. These remain **trip-share targets for tour-level
choices**.

Canton calibration partitions each prepared person batch by residence, retaining
whole persons and tours. Each canton has its own convergence test and best
result; converged cantons stop independently. The evaluation limit applies per
canton. Targets for cantons absent from the simulated population are reported as
skipped, and their existing offsets are preserved. Missing targets, missing or
inconsistent residence cantons, and active cantons with no distance-eligible
trips raise an error instead of using a national fallback.

For each mode other than PT, evaluation index `i` starts at zero:

```text
beta = min(0.8, 1 - 1 / (0.5 * i + 1))
alpha_m += (1 - beta) * (log(target_m / simulated_m)
                        - log(target_pt / simulated_pt))
```

Shares are floored at `1e-6` inside the logarithms to handle zeros; the PT target
must be positive in each calibrated population/canton. Convergence requires the maximum absolute target-share error
to be at most the configured tolerance (0.0001 means 0.01 percentage points).
A lack of change alone is not convergence. If the evaluation limit is reached,
the best evaluated constants and their corresponding choices are retained, with
a warning and `converged: false` in the report. Small samples or incompatible
availability constraints can prevent meeting the tolerance.

Routing, costs and base utilities are calculated once. Calibration prepares each
batch once, retaining tour boundaries, canonical visit patterns, split plans,
trip-count masks and numeric base utilities. Each iteration applies only alpha
differences and recomputes utility-dependent selection weights; unavailable modes
stay unavailable. Counts use integer mode codes and cached indices. Only the best
evaluation is converted to output choice tables after calibration finishes.

A bounded topology cache is shared across all cantons and iterations. Retained
person/tour random streams give stable comparisons; their reservations are released
after each calibrated population/canton, including on failure. Prepared metadata
and utilities require memory proportional to the eligible population. There is no
per-iteration disk cache. Person batch limits and car/bike continuity remain
enforced on every evaluation.

The stage directory contains `choices.parquet`, `calibrated_parameters.yml`
(the loaded model and cost parameters with the selected alphas), and
`calibration.json` (targets, final shares, convergence status, best evaluation,
and the alpha/share history). For `canton`, the report contains those diagnostics
under each canton name, plus skipped target cantons; its top-level evaluation
count is the sum of canton evaluations. Saved choices come from the selected evaluated
constants; no unevaluated final update is exported. The source fit files are
never modified. MATSim continues to receive `dmc.params` and its separately
configured simulation calibration; these population constants are not silently
substituted into the simulation model. To explicitly reuse the calibrated
artifact on a subsequent run, supply it as `dmc_mode_parameters_path`, keep the
original `dmc_cost_parameters_path`, and disable population calibration.

## Routing and features

- `mode_choice_car_routing_backend: java` (default) uses the persistent
  `org.eqasim.switzerland.ch_cmdp.utils.routing.MultiModeRouterServer` for car,
  PT, walk and bike. Seconds/metres from the server are converted to minutes/km.
  Car passengers reuse car routes. Walk/bike routing follows the supplied MATSim
  config (network routing or teleportation); the Python cost stage preserves its times.
- `mode_choice_car_routing_backend: pandana` uses Pandana for car and car
  passengers, Java for PT, and Euclidean distance factors for walk/bike.
  `walk_speed_m_per_s`, `walk_distance_factor`, `bike_speed_m_per_s`, and
  `bike_distance_factor` override defaults of 1.3, 1.3, 4.0, and 1.4 for this backend.
- **Current Java API limitation:** the cached multimodal server returns only PT
  total travel time and distance. Those totals are retained as
  `pt_total_travel_time_min` and `pt_total_distance_km`, but cannot replace DMC's
  separately weighted in-vehicle, access/egress and transfer components.
  `RunBatchPublicTransportRouter` therefore still supplies those components,
  excluding initial waiting. `RunComputeTransitPrices` still calculates fares
  using each synthetic person's subscriptions. This compatibility path duplicates
  PT routing and loads separate Java processes; it can be removed once the server
  exposes the same component semantics and subscription-aware fares. Its totals
  are not substituted for in-vehicle times or used to invent transfer times.
- Java uses the pipeline's runtime, eqasim JAR, mapped network, transit schedule,
  and `calibration.pt_pricing.generate_config`. Events are passed to the server
  for Java car routing; Pandana retains its optional hourly linkstats input.
- Costs, income equivalence, destination densities, PT quality, age, purposes,
  municipality effects, and utility exponents follow the current DMC model.
  Polars utility equations are tested numerically against JAX. Tabular work uses
  Polars; existing spatial helpers use GeoPandas on unique destinations.
- `synthesis.population.trips_with_departures` supplies final departure times.
  With `correct_departure_time: true`, it merges optimized times by
  `(person_id, trip_index)` before mode choice, Java routing/fares, Pandana hour
  selection, and parking-cost calculations. Missing corrections retain the raw
  departure. With the flag false, it returns the sampled trips and does not
  schedule departure optimization.
- `synthesis.population.trips` and `synthesis.population.activities` retain their
  original behavior. Location synthesis and the optimizer consume these stages;
  neither depends on `trips_with_departures`. MATSim and `synthesis.output` use
  the resolved trips, optionally with modes replaced by `mode_choice.run`, and
  align exported activity end times with the trip departures. This also applies
  when mode choice is disabled, without mutating the original activity table or
  reapplying departure optimization.
- The departure optimizer still supplies departures only. Sampled arrivals are
  retained, matching the previous MATSim behavior; derived trip/activity durations
  are recomputed. This refactor does not estimate arrival times or repair existing
  overlaps in independently optimized schedules.

Optional congestion inputs in `config_dib_10.yml`:

```yaml
mode_choice_car_routing_backend: pandana  # or java (default)
mode_choice_car_linkstats_path: /path/to/output_linkstats.txt.gz
mode_choice_car_events_path: ""         # for Java: /path/to/output_events.xml.gz
```

Only the selected backend's input is used. An empty path means free-flow car
routing. A supplied file must exist; changes to its modification time, size or
inode invalidate the routing stage cache. These are input files from a previous
simulation, not dependencies on the simulation currently being constructed.

**Pandana:** reads `matsim.scenario.network.mapped["network"]` and optionally
MATSim tab-separated linkstats (`.gz` supported), with string `LINK` IDs and
`TRAVELTIME0-1avg` through `TRAVELTIME23-24avg` columns in seconds. It selects
`floor(departure_time / 3600) % 24` for each trip, matching the old branch. That
hour's weights apply to the entire path; the clock is not advanced along links.
Only hours present in the requests are loaded. Missing hours, links, and
nonpositive or non-finite observations fall back to `length / freespeed`.
Observed times are bounded below by free flow and above by travel at 1 km/h,
following the old branch, while retaining floating-point precision.

The router builds a directed contraction hierarchy for each used departure hour,
reuses it for that hour's batches, and releases it before building the next.
Without linkstats it builds one free-flow graph. Car links are selected before
snapping to the nearest network nodes. Pandana does not apply Java routing
disutilities or turn restrictions. Node snapping approximates Java link-based
routing; access/egress distances are not added to in-vehicle car time or distance.

Both time and distance are summed along the fastest path for the selected hour.
Parallel links are resolved by that hour's fastest time, then length and link ID,
so the selected parallel link can change with congestion. Unreachable routes
produce null values, making car unavailable downstream. Repeated snapped OD
pairs within each routing batch share a query. Graphs remain in memory and are
never pickled. `mode_choice_routing_batch_size` limits Pandana trip/path batches;
network/linkstats tables and the final route table still reside in memory.
Pandana's native OpenMP thread count follows `OMP_NUM_THREADS`.

**Java:** when `mode_choice_car_events_path` is set, Python passes
`--events-path <file>` to `org.eqasim.core.tools.routing.RunBatchRoadRouter`.
The Java implementation is responsible for constructing event-based travel times
and using them during routing. This change provides and tests the Python CLI
contract; congestion behavior inside the Java router is not implemented or
verified here. PT routing and fares continue to use their existing Java calls
and do not receive the car events option.

Missing or invalid car/PT routes make those modes unavailable. Car also requires
an adult with a licence and synthetic car availability; bicycle availability
uses the synthetic binary flag and the training distance limit. Walking remains
available as a fallback, including beyond the survey estimation's distance cap.
Long-distance walking can therefore be selected; no route is invented or filled
with an arbitrary large penalty.

## Tours and reproducibility

Tours are ordered by person and trip index and split at home departures. One
complete mode sequence is drawn with probability proportional to the exponential
of its summed trip utilities. Car and bike must be picked up where left and
returned to the tour's starting location. Actual coordinates establish continuity,
so two activities sharing a purpose do not imply the same location.

`mode_choice_sampling_method` selects one of two equivalent logit samplers:

- `probability` (default): normalize total log weights with a stable softmax and
  sample from the resulting probabilities.
- `gumbel`: add independent standard Gumbel noise to conditional total log
  weights and take the maximum.

Both methods draw **complete mode sequences**, using the sum of trip utilities.
They use forward/backward factorization rather than enumerating every sequence.
At each backward decision the score includes the log-summed utility of all
compatible prefixes and is conditioned on the selected suffix. Thus hierarchical
Gumbel-max has the same whole-tour distribution as independently perturbing every
complete alternative. It does not add independent noise to individual trip
utilities and then greedily choose modes. The two methods share a distribution,
not necessarily the same realization for a given seed.

`mode_choice_max_trips_per_tour` defaults to **8 trips per computational segment**.
Longer tours are recursively split at work/work-secondary activities in the
central half; when no suitable work anchor exists, the split is at the middle
activity. Trips are never cut in half. A 20-trip tour is therefore handled in
segments of at most 8 trips. A single person still stays in one person batch.

Each boundary carries the possible locations of **both** car and bicycle, plus
the utility weights for reaching those states. Backward selection is conditioned
on returning both vehicles to the original tour's start. A vehicle parked before
a split must be retrieved at that same location afterwards; a split does not
make another vehicle available at work. These computational splits preserve the
whole-tour choice distribution, and tests also verify identical seeded choices
across split thresholds. This permits drive-to-work/walk-subtour/drive-home chains.

Random streams are keyed by `random_seed`, person ID and original tour index.
Changing batch sizes, input row order, cache capacity or the segment limit does
not change choices within a sampling method. Original tour IDs survive removal
of special tours, keeping later tours' random streams stable.

Tours containing loop modes, remote-work legs, or border activities, and chains
that do not both start and end at home, retain their original modes as whole
chains. Freight, LCV and separately injected external populations retain their
existing processing. Persons with no trips require no choice.

`mode_choice_person_batch_size` defaults to 10,000 people. Costs, utility
evaluation, validation and sampling all run inside each batch, and a person's
complete day stays together (including parking-duration calculations). Sorted
input is sliced without materializing all batch frames. Without ASC calibration, only compact choices
accumulate across batches; the upstream routed input and final output still
reside in memory. A single person's trips are never split to satisfy this limit.
Tour sampling converts each batch to arrays once instead of constructing Polars
frames for every tour.

`mode_choice_routing_batch_size` defaults to 2,048 and controls Java worker
batches. `mode_choice_server_chunk_size` defaults to 65,536 requests per CSV
chunk. The server loads the network and routing data once; these options do not
bound the loaded scenario size. CSV parsing reads only the columns needed downstream.

### Server lifecycle

`mode_choice.routing.server` is an early dependency of feature preparation and
routing. Its execution launches Java immediately and returns serializable settings;
it does not wait for data loading. Loading can overlap subsequent Python work once
the mapped network and routing config dependencies have finished. Synpp controls
stage ordering, so overlap with every upstream synthesis stage is not guaranteed.

The process binds to loopback on an automatically assigned port. The client discovers
the endpoint in `server_logs_starting_at_<UTC timestamp>.logs` in the server
stage's cache directory (`context.path()`). Both Java stdout and stderr go only
to this file. Each actual launch creates a new file, including restarts from cached
settings, so previous launch logs are not overwritten. Synpp can still remove old
files when it rebuilds a stage cache directory. Feature
preparation and routing both ensure that the process is running: cached Synpp settings
from another invocation never count as a live server. Readiness is checked only when
routing is needed; initialization failures and timeouts give explicit errors.

`mode_choice.run` stops the owned server in `finally`, including on selection errors
or empty populations. Preparation/routing errors also stop it. An `atexit` fallback
covers normal Python exit, including standalone preparation/routing runs. A forced
process kill cannot execute Python cleanup. Only the process created by this client
is stopped; it never attaches to an arbitrary server at a configured port.

Optional settings (seconds except chunk size):

```yaml
mode_choice_server_startup_timeout: 1800.0
mode_choice_server_request_timeout: 7200.0
mode_choice_server_shutdown_timeout: 30.0
mode_choice_server_poll_interval: 0.5
mode_choice_server_chunk_size: 65536
```

`mode_choice.routing.RoutingServer.RoutingServer` also provides `route_csv`,
`route_batch` and `route` for other Python consumers. Coordinate-pair helpers accept
departure times in seconds; their result dictionaries retain Java's seconds/metres.
Only the population routing adapter converts units. Java request failures propagate;
the client does not silently substitute another mode or a zero-length route.

## In-memory caching and layout

Each pipeline execution creates one bounded LRU transition-graph cache, shared
across all person batches. The key describes the visit pattern and vehicle
boundary states, not raw coordinates: home/work/shop/work/home can reuse a graph
across different people and places. Cached arrays are immutable; current utility
values and availability masks are applied afresh. Prepared tour plans retain graph
keys, so they do not pin transition arrays outside the bounded graph cache.
Utility-dependent selection weights are recomputed for each evaluation.

- `mode_choice_cache_max_entries`: at most 1,024 retained segment graphs.
- `mode_choice_cache_max_edges`: at most 200,000 retained transitions across
  those graphs. An oversized graph is used without retaining it.
- Setting either limit to zero disables graph retention. Eviction affects speed,
  not results. Hit/miss/entry/edge counts are logged at the end of selection.
- `mode_choice_random_cache_max_values`: at most 2,000,000 retained float64
  random values (16 MB) across active prepared batches. Zero disables retention.
  Each batch holds a flat buffer with independent per-tour slices: Gumbel values
  for Gumbel sampling, uniform values for probability sampling. Tours that do not
  fit the remaining budget generate their draws in bulk on each evaluation;
  changing this limit does not change choices.
- Calibration resets each tour's random cursor on every evaluation, using common
  random numbers for stable alpha comparisons. The Python API
  `PreparedBatch.sample_codes(draw_index=N)` provides reproducible new draws for
  positive indices. Index zero preserves the original seeded stream. Buffers never
  wrap or share draws between different people.
- Fitted parameter files use a separate small in-memory cache keyed by resolved
  path, modification time, size and inode. Callers receive independent copies.
  Utility expressions are built once and reused across batches.

These performance caches create no files. Synpp's existing stage-output caching
and the `choices.parquet` output artifact remain part of the normal pipeline.
The graph limit bounds retained cache memory, not the full loaded population or
Java routing memory.

The package is organized as follows:

```text
mode_choice/
  defaults.py           # all runtime defaults, thresholds and policy constants
  run.py                # Synpp entry point and person-batch orchestration
  data/prepare.py       # synthetic population and spatial features
  model/                # fitted parameters, costs and utility expressions
  calibration/          # target shares, log-ratio optimizer and in-memory replay
  routing/              # Java PT/fares and selectable Java or Pandana car routes
  tours/                # batching, tour construction, splits and prepared arrays
  selection/            # selectors, state-aware sampler, graphs and random buffers
```

`defaults.py` is a Synpp dependency, so edits invalidate the consuming stages.
Fitted coefficients/scales remain in the DMC artifacts and shared estimation
constants remain in `dmc.constants`, rather than being copied into this package.

Example options under `config:` in `config_dib_10.yml`:

```yaml
mode_choice_person_batch_size: 10000
mode_choice_sampling_method: probability  # or gumbel
mode_choice_max_trips_per_tour: 8
mode_choice_cache_max_entries: 1024
mode_choice_cache_max_edges: 200000
mode_choice_random_cache_max_values: 2000000
```

## Checks

```sh
uv run python -m pytest tests/test_mode_choice*.py tests/test_departure_times.py -q
```

Tests cover JAX/Biogeme utility parity, cantonal offsets, exact whole-tour and
subtour probabilities for both samplers, work/midpoint split placement, 20-trip
car/bike continuity, split/unsplit equality, cache reuse/eviction/invalidation,
parameter mutation isolation, shuffled inputs, full-stage person batching,
route failures and Java CSV/CLI contracts. Real Pandana tests cover directed
paths, disconnected routes, parallel links, distance/time consistency, network
file loading, hourly congestion, midnight rollover, fallback/bounding rules,
parallel-link changes and batch invariance. Java events forwarding and congestion
input cache invalidation are also tested. Departure-time tests cover both flag
settings, raw-stage isolation, missing corrections, stable trip keys and order,
activity/leg consistency, and corrected Java requests and Pandana hour selection.
Calibration tests cover the legacy
update formula, convergence for both samplers, zero shares, nonconvergence,
batch/cache invariance, routing/cost reuse, and replay of saved parameters.
Canton tests cover additive global/residence-canton utilities, both DMC parameter
sources, independent convergence, unchanged global/PT constants, and saved-offset
replay for both samplers.
Java calls in tests are mocked; a full
pipeline run requires the configured network, transit, population and model files.

The configured fitted JAX artifact was additionally checked on 10,000 feature
rows, including cantonal offsets: maximum absolute utility difference from JAX
was 5.33e-15. Regression tests exercise randomized coefficients, unsigned age
values, zero features, large utilities, and person IDs above 2**53.

Cache benchmark (2026-10-06): 10,000 four-trip tours, three runs per setting,
eight Polars threads, same-process timings excluding Java routing:

| Method | Cache disabled (median) | Cache enabled (median) |
| --- | ---: | ---: |
| Probability | 3.96 s | 1.93 s |
| Gumbel | 2.91 s | 1.80 s |

Cached and uncached choices matched exactly within each method. The repeated
visit pattern produced 9,999 hits and one miss, retaining one 50-edge graph;
real-population gains depend on the variety of visit patterns.

Prepared-batch benchmark (2026-10-08): 2,000 four-trip tours, five selections with
changing alpha adjustments, median of three rounds, eight Polars threads. Both
paths use the current sampler; the baseline rebuilds preparation for each selection.
Routing is excluded, and the retained path materializes choices only at the end.

| Method | Repeated preparation | Retained preparation | Speedup |
| --- | ---: | ---: | ---: |
| Probability | 5.329 s | 3.405 s | 1.56× |
| Gumbel | 4.675 s | 3.092 s | 1.51× |

One-time preparation took 0.370 s and 0.304 s respectively, retaining 8,000 uniform
or 30,000 Gumbel values. Choices and counts matched exactly for every evaluated
alpha vector. These are selection measurements on repeated visit patterns, not
full-pipeline speedups. Tests additionally cover bounded/no random retention,
independent draw indices, preparation reuse, exception cleanup, and agreement with
sequential seeded draws and exact tour probabilities for both samplers.

Server integration checks:

```sh
uv run python -m pytest tests/test_routing_server.py -q
EQASIM_TEST_JAR=/path/to/switzerland.jar uv run python -m pytest tests/test_routing_server_java.py -q
```

The optional smoke test starts the actual cached Java server on a small generated
network and transit timetable, routes car/PT/walk/bike through one process, and
verifies shutdown. Python tests cover asynchronous loading, reuse, restart from
cached settings, loading failure, process exit, timeout, both backends, units,
corrected departure requests, retained walk/bike times and cleanup after failures.
