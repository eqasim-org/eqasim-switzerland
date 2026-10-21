# SUMO scenario

The SUMO stages are independent of the MATSim network pipeline. They read one
OSM extract and do not merge the configured MATSim `osm_file` list.

Start with the [workflow and stage flowcharts](WORKFLOW.md) for the complete
SUMO structure, including network loading, regional cutting, simulation and
the standalone config/assignment branches. Other pipeline stages are grouped
as inputs in those diagrams.

## Install the runtime

From the repository root:

```bash
uv run -m synpp config_dib_10.yml --run sumo.runtime.sumo \
    --working-directory cache_sumo
```

This installs the official `eclipse-sumo` applications and matching `sumo-data`
into a private virtual environment under the runtime stage's Synpp cache.
The default version is `1.28.0`; override it with `sumo_version`.
Neither `pyproject.toml`, `uv.lock`, nor the project environment is changed by
the installer. Installation needs access to PyPI; network and simulation stages
reuse the cached runtime dependency. Explicitly selecting `sumo.runtime.sumo`
again reinstalls it because Synpp reruns explicitly requested targets.
Load the cluster's `eth_proxy` module when needed,
as in `run.sh`. No administrator privileges or system installation is required.
These examples use the repository's ignored `cache_sumo/` directory. Omit
`--working-directory cache_sumo` to use the configuration's working directory.

SUMO is available through the helper below; it is not added to your shell's PATH.
The stage returns a dictionary with `home`, `python`, and `version`.

## Build the road network

```bash
uv run -m synpp config_dib_10.yml --run sumo.scenario.network.road_network \
    --working-directory cache_sumo
```

The `--run` option selects this stage instead of the configuration's normal
MATSim output target. OSM conversion depends on `sumo.runtime.sumo`; returning
a prepared SUMO network has no runtime dependency.

By default, the source is `data_path/osm/<latest_osm_for_correction>`, whose
default is `switzerland-261004.osm.pbf`, the latest Swiss extract currently used
by this repository. This selects an existing local file; it does not download
new OSM releases or automatically choose files by modification time.
Set `sumo_network_file` to an absolute path, an existing path relative to the
repository root, or a filename under `data_path/osm`. The stage checks the
supplied path first, then looks under `data_path/osm` if it is not a file.
Existing direct paths do not require the `data_path/osm` directory to exist.
To select a different single extract:

```bash
uv run -m synpp config_dib_10.yml --run sumo.scenario.network.road_network \
    --working-directory cache_sumo \
    --config sumo_network_file switzerland-latest-2026.osm.pbf
```

This single option accepts a prepared SUMO network, OSM XML (including gzip),
or OSM PBF. `detect_network_format()` in `sumo.scenario.network.tools` reads at
most 64 KiB of the header and returns `sumo`, `osm` or `osm_pbf`. It recognizes
gzip by its magic bytes, XML by its root element (`<net>` or `<osm>`) after any
declaration/comments, and PBF by its first OSM header. Names and suffixes do not
override the contents. Unknown formats and headers exceeding that bound raise
a clear error; the detector never scans the whole network.

`sumo_osm_file` remains a compatibility fallback when `sumo_network_file` is
unset. If both are unset, `latest_osm_for_correction` selects the default OSM.

PBF input is streamed to compressed OSM XML in this stage's temporary directory,
then removed after conversion. XML and compressed XML inputs are read directly.
Source files and upstream caches remain untouched. The configured `random_seed`
is passed to `netconvert`. The default conversion options simplify geometry,
infer ramps and traffic lights, join junctions, and retain edges usable by
passenger cars, taxis, delivery vehicles, buses, trucks, motorcycles or mopeds.
They retain the largest connected component, remove isolated roads, import OSM
turn lanes and lane access, use EPSG:2056, and enable `--ignore-errors`.
Pedestrian-only and bicycle-only edges are omitted by this filter. Multimodal
demand may need a network with those edges included.

`sumo_netconvert_options` accepts a YAML list of strings replacing those defaults.
Input, output, seed, original OSM names and log paths are supplied by the stage.
Inspect `netconvert.log` in the stage cache when assessing conversion warnings.
Input path, size and modification time participate in cache validation.
As with other Synpp stages, an explicitly selected network target runs again;
downstream consumers reuse its valid cache.

The stage returns the absolute path to its cached `sumo_road_network.net.xml.gz`.
To reuse an already prepared SUMO `.net.xml` or `.net.xml.gz` file instead, set:

```yaml
sumo_network_file: /path/to/prepared.net.xml.gz
```

The road-network stage returns that file's absolute path directly, checks that
it exists, and tracks changes to it for cache validation. It does not copy or
modify the file, register the runtime dependency, or require OSM inputs and
conversion settings. Every network consumer depends on the road-network stage,
so an OSM input is converted before it reaches simulation, routing or cutting.
Source-change cache validation belongs to that stage for both input formats.

A downstream stage declares and loads it with:

```python
def configure(context):
    context.stage("sumo.scenario.network.road_network")


def execute(context):

    # Load the cached SUMO network.

    return context.stage("sumo.scenario.network.road_network")
```

## Run SUMO tools from another stage

```python
from sumo.runtime import sumo


def configure(context):
    context.stage("sumo.runtime.sumo")


def execute(context):

    # Run a tool using the isolated runtime.

    return sumo.run(context, ["--version"], catch_output=True)
```

Use `sumo.run(context, ["-c", "scenario.sumocfg"])` to run a prepared simulation.
Set `binary="netconvert"` or another installed SUMO executable to select a tool.
The helper sets `SUMO_HOME` and PATH only for the child process, defaults to the
calling stage's cache directory, streams output, and raises on a nonzero exit.
It uses the wheel's bundled projection database through `PROJ_DATA` and
`PROJ_LIB` when those variables have not already been set. This is needed for
explicit CRS names such as `EPSG:2056`.
`catch_output=True` returns decoded stdout; otherwise it returns the exit code.
`cwd` can select another working directory.

## Population and demand

[Population and plan export](population/README.md) describes the person-plan
file, separate freight/commercial trips, shared tools, and XML writer.
[Simulation presets and loaded assignment](config/README.md) covers microscopic,
mesoscopic and macroscopic configs, execution stages, and congestion-aware
per-trip travel times and distances.
[Network loading and regional scenarios](cutter/README.md) describes default
macroscopic assignment, shared `sumo_modes`, polygon clipping, estimated
crossing-trip entry times and runnable regional bundles. Its default passes
`marouter`'s assigned paths directly to the cutter and preserves them in the
regional run; simulation-based full-network DUA is an explicit option.
`sumo.run` executes that cut scenario; `sumo_simulation_model` selects
`microscopic` (default) or `mesoscopic`. See the
[regional execution commands](cutter/README.md#run-the-cut-scenario).

References: [official installation instructions](https://sumo.dlr.de/docs/Downloads.php),
[OSM import and conversion options](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html).
