# Shared MATSim configuration

`matsim.config.config` is the only stage that calls `RunGenerateConfig` and the
CMDP `RunAdaptConfig`. It returns the absolute path to an adapted config. It
depends on the Java runtime, configuration values and the car-routing utility
stage, so routing, calibration and population stages can reuse it without a
dependency cycle.

The shared config contains the prefix, sampling, seed, threads, freight
strategies, car cost model, bicycle routing, SBB transit settings and PT pricing
module. `eqasim.routingDistanceUtility` is set here from
`calibration.car_routing_vot.optimal_value`, so every consumer inherits it.
Its scenario input filenames and PT pricing filenames are relative;
the stage does not load population, network, schedule, pricing or fitted
calibration files.

Each consumer copies the config into its own directory before editing it:

| Consumer | Local changes |
| --- | --- |
| `matsim.simulation.prepare` | Local pricing paths, calibrated PT parameters, counts and travel-time targets, calibration regions, simulation and calibration settings |
| `matsim.config.pt` | Local network/schedule/vehicles/pricing files and PT parameters; remove population inputs and freight replanning |
| `mode_choice.routing.config` | Copy the prepared PT config, resolve its transit/pricing paths and select fitted cost parameters |
| Cutter and output stages | Edit their existing cut or simulation-output config using shared tools; preserve fitted scenario settings |
| `matsim.simulation.run_from_inputs` | Copy the supplied already-adapted config, preserve custom/fitted settings, resolve supporting paths and update local files, sampling and freight strategies |

`matsim.config.pt` replaces both PT routing and PT pricing config generators.
PT calibration requests this same stage with `calibrate_pt_routing_params=False`
under the alias `pt_calibration_config`. This supplies initial parameters without
creating a cycle; other consumers use the configured or fitted parameters.

`matsim.config.config.adapt_for_routing` applies local PT pricing paths and routing
parameters. `adapt_for_simulation` also applies network calibration, mode-share
calibration, delays, DMC paths, iteration counts and output intervals using the
helpers in `matsim.config.tools`. Preparation applies simulation settings after
population routing and validation, keeping calibration inactive during those steps.

`matsim.simulation.prepare` returns the completed config path.
`matsim.simulation.run` selects the Java runner and its launch switches. The
waiting-to-enter-traffic option and VDF network-event generation remain Java
launch switches because the runners require them.
PT2MATSim converter and mapper templates are different config formats and remain
in their respective stages; they use the same shared XML helpers.

## Portable prepared scenarios

Preparation uses `adapt_for_routing(..., local=True)` before population routing,
then `adapt_for_simulation(..., local=True)` once routing and validation finish.
The `local` option uses filenames beside the config for pricing and DMC inputs,
and the configured prefix for cantonal mode shares. Simulation settings copy
count targets, travel-time targets and special-region files into
`network_calibration_files/` once and reference them with relative paths.

Moving the whole prepared folder preserves these references. Optional beta
calibration also needs the pipeline cache: its `eqasimCachePath` is relative to
the prepared directory in local mode, so retain the same relative cache layout.
The default `local=False` behavior remains available for other consumers.
The pipeline resolves legacy Java calibration paths in launch arguments when the
simulation runs from a different directory; the prepared XML remains unchanged.
For a direct Java invocation, run from the prepared scenario directory.

Shared XML writes enable pretty printing and normalize indentation to four
spaces, including after editing compact XML. Comments and the DOCTYPE survive.

## Editing

All XML editing helpers and simulation configuration helpers live in
`matsim.config.tools`. The previous `matsim.simulation.config_utils` module and
config editors in `matsim.cutter.cutter_tools` were removed. Cutter clipping and
geometry helpers remain in `cutter_tools`.

```python
from pathlib import Path
from matsim.config import tools

# configure(context)
context.stage("matsim.config.config")

# execute(context)
path = Path(context.path()) / "local_config.xml"
tools.copy_config(context.stage("matsim.config.config"), path)
tools.change_params(path, {
    "plans.inputPlansFile": "local_population.xml.gz",
    "eqasim.costParametersPath": "/path/to/cost_parameters.yml",
}, create=True)
```

`change_params` updates existing fields strictly by default. Use `create=True`
when adding missing modules, nested parameter sets or parameters. Dot-separated
paths support nested parameter sets and zero-based indices. Booleans and `None`
are written as `true`/`false` and `null`. Comments and DOCTYPE are retained, and
identical duplicate config entries are collapsed.

For editing an XML tree, use `get_module`, `get_parameter_set`, `get_param` and
`set_param`, then `write_config`. `get_parameter_set` matches identifying fields
such as `fromMode`/`toMode` or `scheduleMode`; updating transfer penalties and
mode assignments is idempotent.

`copy_config(..., input_paths={...})` resolves selected relative input paths
against the source config before relocating it. Nested parameter sets and
semicolon-separated calibration region files are supported.

## Validation

Focused tests cover config generation, consumer-specific overrides, unchanged
source templates, nested fields, PT parameter idempotence, external custom
settings, network converter/mapper options, and prepared queue/VDF simulations
with alpha/beta calibration. Java generation and reloading of
PT-modified configs can be exercised with the configured eqasim JAR. Before
running Java checks, verify that the selected JAR is a valid archive and contains
the required main classes. Cluster builds need explicit Maven proxy settings;
the launchers enable these through `MAVEN_USE_ENV_PROXY=true`.

Dependency graphs are checked without executing pipeline stages. The default
configuration has one shared generator. Testing PT parameter calibration also
requires an explicit `eqasim_commit`: the existing main and calibration runtime
stages declare different defaults for that option. The graph checks pin this
option consistently; this refactor does not change the runtime JAR versions.
