# Polars tables, IPC caches and native XML

SUMO tabular processing uses Polars. `sumo.tables.as_polars` is the shared
boundary for pandas/GeoPandas tables from other pipeline packages, Polars
tables, Arrow tables and cached file paths. Geometries are stored as WKB binary;
Python Object columns are excluded from table caches. Source tables are not edited.
Networks remain graph structures and geometry operations use Shapely/GEOS.

`sumo.tables.write_table` writes uncompressed Arrow IPC (`.arrow`).
`read_table` uses memory-mapped Polars IPC reads. This favors read speed and
avoids Python-object deserialization, at the cost of larger files than
compressed Parquet. Existing external Parquet inputs can still be read.

## Stage contracts

- `sumo.scenario.population.population` returns a dictionary of absolute paths:
  `persons`, `activities`, and mode-filtered `vehicle_trips`.
- `sumo.scenario.demand` returns departure-sorted XML in `queries`, the sorted
  IPC file in `table`, and counts and departure bounds.
- Loading and standalone assignment return `trip_metrics.arrow` through their
  existing `metrics` field.

Use `as_polars(result["persons"])` or `read_table(result["persons"])` rather than
treating a stage result as an in-memory DataFrame. Synpp only pickles the small
manifest of paths. Demand combines Arrow-backed columns and sorts with Polars
in RAM; it does not use SQLite. The implementation intentionally prioritizes
speed over minimizing RAM.

## Native vehicle XML

`sumo.native.runtime` compiles `sumo/native/write_trips.cpp` into its own stage
cache. It uses the compiler and the Arrow headers/library already supplied by
PyArrow, plus system zlib. The SUMO launcher loads GCC and sets
`POLARS_MAX_THREADS` from the Slurm CPU allocation unless explicitly overridden.
There are no new uv dependencies or changes to the private SUMO installation.

With default `sumo_native_xml: true`, the C++ writer reads the sorted IPC file
directly and writes buffered gzip/plain SUMO XML. Population trips require no
Python record dictionaries or XML elements. Existing vehicle-trip XML is parsed
with C-backed lxml and retained as serialized elements, preserving arbitrary
attributes and parameters when mixed with population trips. Other SUMO XML
readers continue to use lxml's binary streaming parser.

The writer escapes attribute values, preserves Unicode and whitespace, rejects
invalid XML characters, and reports progress each million trips. Its build cache
validation includes the C++ source, Arrow version and machine architecture.
Set `sumo_native_xml: false` to use the shared lxml writer when a compiler is
unavailable. Both paths retain the same trip IDs, modes, times and metadata;
equivalent numeric spellings such as `10` and `10.0` may differ in the XML.

## Class-aware macroscopic assignment

`sumo.native.marouter` builds a private SUMO 1.28.0 assignment executable.
Upstream marouter retains vehicle type IDs in its demand matrix but ignores
their definitions and routes every cell with the default passenger vehicle.
This rejects legal delivery-only endpoints and can assign freight to roads
that exclude it.

The patch in `sumo/native/marouter.patch` loads type definitions and uses the
cell's actual type for path search, costs and route probabilities. Bulk search
reuse includes both origin and type. Car, truck and delivery flows still share
the same link costs and congestion calculation. No permissions are relaxed and
the regional workflow does not reroute after assignment.

Trips whose endpoints occupy the same directed edge are grouped separately
when their arrival position precedes their departure position. Those groups
use SUMO's native loop search during assignment, so their return paths also
contribute to congestion. Forward trips on that edge retain their direct path.
Loop search disables bulk-tree reuse while testing different successor edges.

Both `sumo.simulation.load_network` and `sumo.router.assignment` use this binary.
The stage verifies the pinned source archive checksum, compiles only marouter,
and returns its cached executable path. Patch changes invalidate the build.
Build logs remain under its `build/` directory. The first run downloads and
compiles the source; subsequent runs reuse it. The standard launcher loads
`cmake`, `xerces-c`, `fox` and `mesa-glu`, in addition to GCC, and the assignment
configuration uses the pipeline's `threads` value for native routing threads.
Keep these modules available when running the cached executable too.
The supported version is currently `sumo_version: 1.28.0`.

## Local benchmarks

On a 200,000-row synthetic table, with local cached reads:

| Format | Write | Read plus touching values | File size |
| --- | --- | --- | --- |
| IPC, uncompressed | 0.034 s | 0.005 s | 15.40 MB |
| IPC, LZ4 | 0.034 s | 0.045 s | 2.50 MB |
| Parquet, Snappy | 0.043 s | 0.069 s | 1.65 MB |

For 200,000 vehicle trips, including Polars sorting, IPC output and gzip XML,
the lxml path took 4.05 s and C++ took 0.44 s (about 9 times faster).
These are small, cache-hot benchmarks, not estimates for national runs.

The next pipeline run rebuilds affected SUMO caches for these new contracts.
Running processes keep their previously imported Python code and must be
restarted to benefit. Do not run two jobs against the same writable cache.
