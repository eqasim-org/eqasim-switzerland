# Comparing Java and FLASH public-transport routing

`analysis.pt_routers.analyse` runs the production
`mode_choice.routing_variables.pt` implementation twice: once with the Java
multimodal server, once with FLASH-TB. Both receive the identical, seeded sample
of synthetic-population trips. It saves complete results and figures beneath
`output_path/pt_routers/`, with `index.html` as the entry point.

Run from the repository root:

```bash
uv run -m synpp config_test_pt.yml
# Or submit the dedicated 24-CPU, two-hour job:
mkdir -p logs
sbatch run_test_pt.sh
```

The test configuration copies `config_dib_sumo.yml` but selects only this analysis.
It uses a private working directory, the completed source configuration's prepared
trips and Java launch settings, and the existing fitted DMC parameter files.
These source caches are read-only. Neither population synthesis nor model fitting
is repeated. Java runs first and stops before FLASH preprocessing to avoid
holding both timetables in memory simultaneously. A separately pinned Java build
provides the current PT component exports without changing the source scenario.

For another configuration, add `analysis.pt_routers.analyse` to `run`. By default
the input stage depends on `mode_choice.data.prepare`, Java uses
`mode_choice.routing.java.server`, and FLASH uses `data.gtfs.cleaned/output`.
Optional existing-cache sources are:

| Option | Meaning |
| --- | --- |
| `pt_router_analysis_source_inputs` | Synpp pickle of prepared population trips |
| `pt_router_analysis_source_server` | Synpp pickle of Java server launch settings |
| `pt_router_analysis_source_gtfs` | Optional extracted GTFS for independent frequency diagnostics; routing preprocessing uses the current `data.gtfs.cleaned` stage |
| `pt_router_analysis_eqasim_commit` | Compatible Java commit for the private runtime |
| `pt_router_analysis_eqasim_path` | Existing compatible JAR, instead of building it |
| `pt_router_analysis_sample_size` | Maximum requests, default 2,000 |
| `pt_router_analysis_repeats` | Complete routing invocations per backend, default 2 |
| `pt_router_analysis_verify_serial` | Compare all FLASH results to a one-worker baseline, default true |
| `pt_router_analysis_output_folder` | Subdirectory under `output_path`, default `pt_routers` |
| `pt_router_analysis_frequencies` | Read scheduled GTFS departures, default true |

Source paths must describe the same scenario and service day. Use only trusted
Synpp pickle files. The comparison disables road assignment for its private Java
server and requests PT only. Set `prepare_sumo: false` and
`route_population_once: false`, as FLASH cannot export population itineraries.
The full feed is retained; sampling requests does not reduce preprocessing size.

FLASH preprocessing is shared automatically with production mode choice under
`<data_path>/gtfs/flash_tb/current/`; no completed-timetable pickle option is needed.
The selected GTFS archive names/paths, modification timestamps and sizes determine
reuse. A changed source replaces the previous shared bundle after successful
preprocessing, while unchanged sources reuse it across working directories and
query-worker counts. Service-day and partition settings must agree. See the
[shared-cache lifecycle](../../mode_choice/routing/flash/README.md#shared-preprocessing-cache).
The report keeps the original preprocessing duration and records cache reuse
separately. Routing preprocessing always reads the current cleaned GTFS stage;
an old diagnostic feed override cannot rebuild the cache from stale GTFS.

Equal-arrival, equal-transfer FLASH journeys use query and journey indices as
stable tie-breaks, so Polars grouping order cannot change the selected variables.

## Results and interpretation

- `requests.parquet`: exact sampled requests, including stable trip identifiers.
- `java_routes.parquet`, `flash_routes.parquet`: production PT utility variables.
- Separate `*_cost.parquet` files: Python `estimate_simple_cost` fares for both
  backends, so Java fare computation does not affect the FLASH timing.
- `comparison.parquet` / `.csv`, `metrics.csv`, `summary.json`: ID-aligned
  comparisons, reachability, mean and absolute differences.
- `routing_times.csv`, `timings.json`, `runtime.png`: routing-stage wall time
  and throughput, with Java initialization and FLASH preprocessing separately.
- Time, distance, transfer, reachability and selected-vehicle-mode figures.
- `vehicle_modes.csv`: boardings and passenger-leg kilometres by vehicle mode
  in the sampled selected journeys. These are not fleet vehicle kilometres.
- `scheduled_departures.csv`: full-feed vehicle-trip starts per service-day hour
  and mode. Frequency-based GTFS services contribute expected departures.
- Raw Java itineraries and selected FLASH legs for further investigation.

Travel time includes initial waiting. Paired statistics include only trips
reachable by both routers; the reachability chart also shows one-sided failures.
Java optimizes its configured routing disutility; FLASH chooses earliest arrival,
then fewer transfers. FLASH distances are sums of stop-to-stop geometric segments,
whereas Java distances follow its network. These differences must be considered
before interpreting a numerical disagreement as a bug.

Routing timings include request processing, parsing, diagnostic output and Python
fares. One persistent FLASH worker pool is retained across repetitions: the first
call includes native initialization, and later calls measure warm routing. The
serial baseline uses the same fare and export code and must match every trip
exactly. Runtime builds and downstream
plotting are excluded. Java saves its results and overview before FLASH begins;
if the two-hour Slurm limit interrupts full-feed preprocessing, those partial
outputs remain available. Detailed logs stay in the private stage caches.

## Auditing performance

The original comparison recreated FLASH's native process on every repetition.
The current comparison retains the pool across both trip batches and repetitions.
`flash_tb_query_threads` defaults to `threads`; set it to 1 for serial routing.
`flash_tb_preprocessing_threads` controls preprocessing separately. The native
OpenMP pool shares one read-only timetable and gives each worker private query
state. Request chunks run in parallel and results are written in input order.
Worker counts and native-loop timings are recorded per batch, alongside total
stage times. The one-worker check detects differences in every routing variable.

The 10,000-trip configuration uses 24 query workers and three repetitions. It
reuses the completed single-day timetable, writes to `pt_routers_10000/`, and
retains the earlier 2,000-trip output directory.

`analysis.pt_routers.profile_flash` separates these costs, repeats queries through
the same native process, and counts GTFS services independently for the requested
date. It also measures Java on the same allocation, writes Python profiles, and
compares four candidate stops per endpoint with a one-stop diagnostic case.
The latter changes reachability and should not replace the default solely for
speed. The audit uses the existing timetable; it does not repeat preprocessing.

```bash
sbatch run_profile_pt.sh
```

`config_profile_pt.yml` selects this stage and writes to
`output_path/pt_routers_performance_audit/index.html`. FLASH raw-routing timings
exclude fares and selected-leg parquet export, while Java's measurements cover
its complete production stage. The report states these scopes explicitly.
The pinned native import uses identical start/end dates and applies both weekly
calendars and calendar-date exceptions; it does not unroll all feed dates.

## Inspecting journeys found only by FLASH

The report ends with a FLASH-only cohort section: time/distance/transfer statistics,
walking and departure-hour indicators, transit modes used and 50 random endpoint
examples. Both-router journeys provide a reference, using FLASH variables for
both groups. The sample uses `random_seed`, without replacement; coordinates are
shown as WGS84 latitude/longitude. CSV exports also retain EPSG:2056 coordinates.

The Java CSV is checked for omitted rows, empty results and finite raw times that
were subsequently invalidated. These observations do not determine individual
failure causes. Walking-fallback, stop-search, network snapping and timetable
differences require further Java diagnostics to confirm.

To update this section using saved results, without invoking any routing stages:

```bash
uv run -m analysis.pt_routers.regenerate config_test_pt.yml
```

## Controlled SwissRailRaptor diagnosis

`native/RaptorDiagnosis.java` can be compiled against the existing standalone
eqasim JAR. It loads the existing network, schedule and Eqasim PT costs read-only,
then runs the same requests with independent `RaptorParameters`. Counterfactuals
vary the access-stop search radius, direct-walk factor and use of original
coordinates instead of network-link centres. The transfer-walk limit stays fixed.
The diagnostic preserves raw `EMPTY`, walking-only, PT and invalid-result outcomes
before the server converts them to unavailable alternatives.

The experiment in `java_diagnosis/` tests all FLASH-only requests and 100 seeded
matched controls. Its baseline must reproduce saved availability exactly before
recovery counts are accepted. Counts for different parameter changes overlap.
An optional follow-up tests only the unresolved requests at a 3 km access radius.
These tests run directly against Java; they do not execute a pipeline or rebuild
FLASH data. Their results are diagnostic and do not modify production defaults.

After generating the CSVs, append the measured results to the saved report with:

```bash
uv run -m analysis.pt_routers.raptor_diagnosis <comparison-output-directory>
```
