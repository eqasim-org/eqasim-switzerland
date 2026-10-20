# Repository guidance

This repository is the Python eqasim Switzerland pipeline, orchestrated by `synpp`.
It builds a synthetic population, creates a MATSim transport scenario, runs it,
and analyzes the results. Project documentation lives in [doc/](doc/INDEX.md).
Run commands from the repository root unless a document explicitly says otherwise.

## Configuration and execution

- Inspect `config_dib_10.yml` before changing or running the pipeline. It is the
  working source of truth for stages, inputs, outputs, resource settings and
  scenario choices. Private configuration files are usually ignored by Git.
- The standard command is `uv run -m synpp config_dib_10.yml`. Other local
  configurations select different samples, analysis runs or secondary-location
  algorithms. Inspect the configuration actually named in the user's request.
- `config.yml` is the short beginner template. Keep it focused on important
  inputs and scenario choices, with short comments. Do not copy private paths,
  credentials or every advanced option into it.
- Declare an option's default on its first use in `configure(context)`. Subsequent
  reads use `context.config("option")`. Keep defaults consistent across stages;
  Synpp can reject conflicting declarations even before executing a stage.
  Do not introduce a separate `pipeline_defaults` registry.
- Use paths and values from the configuration or stage context instead of
  hard-coding cluster locations. Do not print entire private configurations:
  they can contain API keys.

## Where things belong

- `data/`: source datasets, GTFS/OSM cleaning, spatial layers and cross-border inputs.
- `synthesis/`: household/person synthesis, trips, departures and activity locations.
- `synthesis/population/spatial/secondary/` and `secondary_nn/`: RDA and neural
  secondary-location assignment. Keep algorithm-specific dependencies conditional.
- `dmc/`: survey preparation, estimation, parameter export and value-of-time logic.
- `mode_choice/`: applying those utilities to the synthetic population and routing.
- `matsim/config/`: shared MATSim config generation and XML editing tools.
- `matsim/scenario/`, `simulation/`, `cutter/`, `output.py`: scenario construction,
  preparation/execution, regional cutting and export respectively.
- `analysis/`: mode shares, counts, travel times and public-transport comparisons.
- `calibration/`: calibration inputs and regional/manual corrections.

## Stage contracts and caches

- Register configuration and dependencies in `configure(context)` before using
  them in `execute(context)`. Preserve conditional dependencies and avoid cycles.
- `context.stage(name)` returns that stage's Python result; it is not necessarily
  a filename. `context.path(name)` gives its cache directory. Inspect the producer
  before constructing a path or changing a return type.
- In particular, `data.gtfs.cleaned` writes extracted feed files under its
  `output/` directory, including `output/routes.txt`. Its return value `"gtfs"`
  is not the directory to append to the stage path.
- Treat upstream caches as read-only. Copy inputs into the consuming stage before
  editing them. Export copies of simulation results so cached simulations remain
  usable; do not move their files out of the cache.
- Use the configured random seed for sampling. Preserve person/trip identifiers,
  row alignment and reproducibility. Avoid unseeded global RNG state and Python's
  process-dependent `hash()` for persistent cache keys.

## MATSim and network conventions

- Generate the base config in `matsim.config.config`; edit local copies through
  `matsim.config.tools`. Do not restore duplicate generators or XML string replacements.
- `matsim.simulation.prepare` copies scenario inputs, detailed geometry and
  `network_calibration_files/` into its directory. Calibration inputs use relative
  paths in the prepared config. `matsim.output` copies these prepared files.
- Preparation returns the completed config path. Check the current run-stage
  contract before changing how Java main classes and launch arguments are chosen.
- Keep link IDs and detailed geometries aligned through filtering, merging,
  splitting and PT correction. Existing detailed curves must survive; new or
  reversed links need matching directed shapes. Counts match current simulated
  IDs directly, without reconstructing old merged-link fragments.
- The mapper feeds optional `matsim.scenario.network.corrected`, followed by
  `matsim.scenario.network.pt`. Consumers of the final PT network must use its
  corresponding geometry and schedule rather than mixing stages.
- Preserve turn restrictions, `old_osm_id` provenance, toll boundaries and summed
  tolls when merging links. Maintain car and truck connectivity; disconnected
  truck access is removed from allowed modes rather than deleting those links.
- Pass `context=context` to `matsim.readers.read_network` so it can reuse the
  shared binary cache. Cache identity includes source path, modification time,
  size and reader options/schema. Open XML parser streams in binary mode,
  including compressed files; `lxml.iterparse` requires bytes.
- MATSim output households may omit custom attributes. Mode-share analysis reads
  output households first, then falls back to the parent scenario's
  `<output_prefix>households.xml.gz` if income classes are absent.

## DMC and population conventions

- Home and destination OVGK qualities are `homePtQuality` and
  `destinationPtQuality`: A/B/C/D/none map to 0/1/2/3/4.
- PT utility uses an estimated beta times quality raised to an estimated exponent
  for each feature. Both quality betas start at -0.05 and exponents at 1.
  Keep JAX, Biogeme, population utilities, parameter mappings and writers consistent.
- Default JAX VOT constraints target car 32 and combined PT 18 CHF/hour. These are
  external calibration assumptions; do not describe them as unconstrained estimates.
- Keep departure corrections keyed by `person_id` and `trip_index`. Preserve
  sampled arrival times, non-travellers and activity/trip consistency.

## Cluster runtime and validation

- Inspect the relevant launch script before changing modules or Slurm resources.
  `run.sh`, `run_analysis_one.sh` and `run_secondary_comparison.sh` load the proxy
  and enable `MAVEN_USE_ENV_PROXY=true`. The shared Maven wrapper creates temporary
  proxy settings; `eth_proxy` environment variables alone do not configure Maven.
  Preserve the user's `~/.m2/settings.xml`.
- For a Java failure, find the first Maven/Java error before the Python traceback.
  Check repository access, the actual JAR path, archive validity and required
  classes before changing Python call sites or JVM memory settings.
- Prefer focused validation: YAML parsing, affected dependency graphs, local
  regression tests, shell syntax checks, small fixtures or read-only checks on
  existing outputs. Do not launch a full population build or MATSim run for a
  small analysis, documentation or reader change.
- `tests/` is intentionally local and ignored. Keep local tests available, but
  do not track them, force-add them or add ignore exceptions for individual tests.
- Preserve existing staged and unstaged user work. Do not include unrelated
  changes in a commit, rewrite caches, or push without the user's instruction.

## Documentation

- Keep project guides, module READMEs and changelogs in `doc/`, mirroring package
  paths where useful. [doc/INDEX.md](doc/INDEX.md) is the documentation index.
- Keep this `AGENTS.md` at the repository root for automatic discovery. Update
  documentation links and package metadata when moving documents.
- Leave third-party environment files and generated analysis artifacts in their
  own directories. Moving source documentation must not alter installed packages
  or break generated report workflows.
