# Experimental FLASH-TB backend

Implementation: `mode_choice/routing/flash/`. Upstream: [TransitRouting/FLASH-TB](https://github.com/TransitRouting/FLASH-TB), pinned to `23963f5ff14e5f761c16a25ab4694da7900e64ba` (MIT). Java remains the default.

This is a working experimental router, **not yet a production-equivalent replacement for Swiss Java routing**. It uses real FLASH-TB preprocessing and arc-flag queries, not a Python imitation. Regression checks and matched Java comparisons are executable below.

## What is implemented

- `runtime.py`: clone the pinned source into its Synpp cache, compile `native/router.cpp` against upstream headers. User checkouts are verified and never modified. Requires Linux, C++17, OpenMP and NUMA development headers/library. Configure `flash_tb_numa_include` if `numa.h` is outside the compiler's include paths. No compiler subprocess runs for the default Java backend.
- `timetable.py`: read **`data.gtfs.cleaned/output/`**, select one service day, convert to RAPTOR/Trip-Based data, compute transitive transfer shortcuts, deterministic geographic partition cells and genuine arc flags. The stage uses the shared data-folder cache described below. The partition is a simple longitude partition; it is not KaHIP/METIS tuning.
- `router.py`: EPSG:2056 coordinates; nearby-stop access/egress using a spatial index; query the Pareto set; choose earliest arrival, then fewer boardings; aggregate metrics with Polars. Bounds memory through `pt.py` batching. One native process keeps the timetable and indices loaded across production batches and is closed on success or failure.
- `pt.py`: return the normal trip-keyed PT variables and use the PT stage's independently selected pricing provider and usual `cost.parquet`. Missing/unreachable routes get the existing 1e9-minute sentinel. Original IDs and order are preserved.
- `benchmark.py`, `sample.py`, `check.py`: reproducible GTFS fixtures, private regional extracts, actual Java RAPTOR comparisons, and executable regression checks. Benchmark files are written only to the supplied output directory.

GTFS service calendars and calendar exceptions are evaluated for the selected date; times beyond 24:00 remain seconds within that service day. Upstream transfer edges are transitively closed. This may permit longer chains of walking than Java's configured local walking search. The adapter corrects implicit boarding buffers when reporting in-vehicle and waiting time.

## Configuration

```yaml
mode_choice_pt_routing_backend: flash  # default: java
mode_choice_pt_price_backend: python   # default for flash; java is optional
flash_tb_date: "20260318"                # optional if gtfs_date is already an explicit date
flash_tb_access_stops: 4
flash_tb_max_access_distance_m: 1500.0   # Euclidean radius at each endpoint
flash_tb_walk_speed_m_per_s: 1.3
flash_tb_walk_detour_factor: 1.3
flash_tb_query_batch_size: 10000
flash_tb_preprocessing_threads: 1
flash_tb_query_threads: null  # Use the pipeline's threads; set 1 for serial queries.
flash_tb_partition_cells: 32
route_population_once: false
prepare_sumo: false
```

For a local development checkout also set `flash_tb_source_path` and, when needed, `flash_tb_numa_include`. No paths are hard-coded to the cluster. If `gtfs_date` is `dayWithMostServices`, set `flash_tb_date` explicitly; the adapter does not silently guess a service day.

## Shared preprocessing cache

Production mode choice, the PT comparison and the performance audit all reuse
`<data_path>/gtfs/flash_tb/current/`. The Synpp timetable result contains paths to
this bundle; changing a population sample, working directory, query executable
or thread count does not repeat preprocessing.

`manifest.json` stores a deterministic SHA-256 identity of the selected source
GTFS archive paths/names, nanosecond modification timestamps and sizes. The
optional replacement TPG archive is included; unrelated archives are excluded.
The GTFS cleaning stage uses these same source signatures, so changing a ZIP
without changing its size also refreshes the cleaned input. Archive contents
are not hashed on every run. Renaming, modifying or resizing a selected source
triggers a rebuild; preserving all three deliberately will not be detected.

Only one completed version is retained. A new bundle is built in `.building/`,
validated and installed under an exclusive file lock, then the previous bundle
is removed. Failed builds leave the previous completed data intact. Parallel
callers reuse the first completed build, and routers hold shared locks until
closed so replacement cannot delete files still in use. Interrupted installation
is recovered on the next call. Preprocessing logs stay with the shared bundle.

The manifest also checks the explicit service date, partition count and binary
format version. With unchanged GTFS, incompatible settings raise an error rather
than silently reusing the wrong timetable. To intentionally change these settings,
remove `gtfs/flash_tb/current/` while no routing job is using it, then run again.
Use consistent GTFS cleaning settings across configurations sharing `data_path`.
The cache is derived data, so a missing or incomplete bundle is rebuilt too.

Standalone benchmarks and regression fixtures use the same cache implementation
inside their own output/temporary directories; they cannot replace the shared
country-wide data with a small fixture. `cache.import_prepared` supports an
explicit one-time copy of a verified legacy result. It preserves the original
Synpp files and normalizes legacy grouped numeric stop IDs in the copies.
It is never used as a fallback when a GTFS source changes.

## Important limits

- **Fares:** upstream FLASH-TB has no Swiss fare model. `mode_choice_pt_price_backend: java` requests fares through the shared Java server's `/cost` endpoint. Raw PT requests still require routing internally in Java and price that Java itinerary, which may differ from FLASH. It is explicitly logged as a hybrid. This policy does **not** deliver a pure FLASH-TB speedup. `python` (the default for FLASH) uses the existing approximate `dmc.cost.pt.estimate_simple_cost` model, with the fitted regional-pass radius and population subscription/age inputs; it is not equivalent to detailed Swiss fares.
- **Objective:** earliest arrival / boarding-count Pareto labels do not reproduce the Swiss generalized-utility route objective. Route ties can have different transfers, in-vehicle time and distance. DMC coefficients and tour selection downstream remain unchanged.
- **Distance:** in-vehicle distance sums great-circle segments between consecutive stops of the recovered boarded segment; it does not follow rail/road network geometry or GTFS shapes. Walking distance is also geometric. Do not interpret these as MATSim network distances.
- **Access:** only the configured nearest stops within the radius are queried. A useful farther platform may be missed. Walking access is beeline with detour/speed factors, not a street-network search.
- **GTFS semantics:** this inherits the upstream importer (calendar/frequency support, its stop-buffer/transfer conventions and any unsupported GTFS restrictions). Accessibility, pickup/drop-off and production transfer-rule parity have not been certified against pt2matsim. Frequency-based services are not part of the measured comparison.
- **Exports:** native stop/leg IDs are not MATSim schedule references. `route_population_once` and `prepare_sumo` are rejected rather than exporting misleading Java-compatible routes. Exact fare pricing and MATSim itinerary mapping remain future integration work.
- This task did not replace departure correction's separate router or remove the existing early Java server dependency used by car routing.

## Run regression checks and benchmarks

Clone the upstream repository, then from the pipeline root:

```sh
git clone https://github.com/TransitRouting/FLASH-TB.git /tmp/FLASH-TB
git -C /tmp/FLASH-TB checkout 23963f5ff14e5f761c16a25ab4694da7900e64ba
uv run -m mode_choice.routing.flash.benchmark \
  --source /tmp/FLASH-TB --jar /path/to/switzerland.jar \
  --output /tmp/flash-fixture --queries 1000 \
  --numa-include /path/to/numactl/include
uv run -m mode_choice.routing.flash.check --binary /tmp/flash-fixture/flash_tb_router
```

`--numa-include` is unnecessary if NUMA headers are already on the compiler's include path. The adapter needs only the transit headers; it does not build upstream TP/ULTRA executables or require their sparsehash submodule.

For real data, first make a small private sample, then date-filter it identically for both engines:

```sh
uv run -m mode_choice.routing.flash.sample --feed /path/to/cleaned-GTFS/output \
  --output /tmp/flash-region --bounds 6.1 46.18 6.18 46.24 --routes 2
uv run -m mode_choice.routing.flash.benchmark \
  --source /tmp/FLASH-TB --binary /tmp/flash-fixture/flash_tb_router \
  --jar /path/to/switzerland.jar --gtfs /tmp/flash-region \
  --date 20260318 --output /tmp/flash-real --queries 1000 --cells 4
```

The Java benchmark uses actual `SwissRailRaptor` from the supplied eqasim JAR. Both engines receive the same dated timetable and transitive transfer graph, at-stop requests, one thread, and earliest-arrival costs. Java excludes pure-walking fallbacks from transit results. Java is warmed with 100 requests; its recorded loop includes itinerary extraction and CSV output. FLASH timings separately include native loading, its query/output loop, and complete Python-wrapper time. This is not a comparison against production Swiss fare/utility parameters or the HTTP server, and it is not a whole-pipeline speed prediction.

The output includes `report.json`, full `comparison.csv`, request/result Parquets and build/preprocessing/Java logs. `flash` and plain `tb` use the same preprocessed timetable as a correctness cross-check. The benchmark runs in fresh processes; production batching additionally reuses the native process.

## Measured results (2026-10-28)

One run per fixture, seed 518745, service day 2026-03-18, 1,000 requests each. These are small-network smoke benchmarks, not estimates for the full Swiss population. The real sample was extracted from an existing cleaned 2026 GTFS feed around Geneva: routes `91-33-C-j26-1` and `91-L4-A-j26-1`, 71 stops after date filtering. Full local comparisons are retained under `mode_choice/routing/flash/benchmark_results/` (ignored generated output); the commands above recreate them.

| Measurement | Controlled GTFS | Real two-route GTFS sample |
| --- | ---: | ---: |
| Stops | 4 | 71 |
| Preprocessing | 0.010 s | 3.104 s |
| FLASH native query/output loop | 0.018 s | 0.011 s |
| FLASH complete Python routing call, including native process loading | 0.061 s | 0.044 s |
| Additional Python router initialization | 0.012 s | 0.012 s |
| Plain TB complete Python routing call | 0.057 s | 0.044 s |
| Java warmed query/output loop | 0.054 s | 0.203 s |
| Java process including initialization | 0.972 s | 1.540 s |
| FLASH reachable requests | 1000 | 941 |
| Java reachable requests | 1000 | 960 |
| Jointly reachable requests compared below | 1000 | 941 |
| Mean absolute arrival-time difference | effectively zero | 2.36 seconds |
| Transfer-count agreement | 100% | 92.35% |
| Mean absolute in-vehicle distance difference | 6.84 metres | 5.40 kilometres |

FLASH and unpruned TB agreed on earliest-arrival times for every request in both fixtures. Arc flags did not demonstrate a consistent benefit over plain TB on these small samples. Python vectorized aggregation and access handling are included in the wrapper timings; fare computation is excluded from both engines.

The real-data distance gap and reachability/transfer differences are material. Tied arrival-time journeys, transfer/boarding conventions, geometric distance and Java's route reconstruction can produce different itineraries. This comparison does not isolate every discrepancy or certify model equivalence. In particular, do not switch production DMC to FLASH based solely on the timing result. No full-country preprocessing, large timetable memory benchmark, production HTTP-server benchmark or detailed Swiss fare comparison was performed.

Validation: executable native regressions passed (direct, transfer, unreachable, service date, ID/order preservation, TB equivalence, native process reuse and cleanup, empty and invalid inputs); seven backend integration tests passed; all three Java/FLASH fare dependency configurations resolved. The broader existing configuration test reports a separate mismatch for `secondary_nn_no_car_no_pt_weight` between analysis and secondary-location stages; this task does not change those defaults.

## Routing-folder migration

FLASH moved from `dmc.flash_tb` to `mode_choice.routing.flash`; its Synpp stages are now `mode_choice.routing.flash.runtime` and `mode_choice.routing.flash.timetable`. The `flash_tb` routing value remains accepted as an alias for `flash`. Replace the old `flash_tb_fare_model` option with `mode_choice_pt_price_backend`, whose values are `java` and `python`. Leaving the new option unset selects Python pricing for FLASH and Java pricing for Java routing.
