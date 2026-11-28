# Shared stage performance monitoring

Import the reusable decorator in a Synpp stage:

```python
from scripts.monitoring import monitored


def configure(context):
    pass


@monitored
def execute(context):

    # Run the stage's work.

    return result
```

SUMO and mode-choice stages use it. Other packages can opt in with the same
import. Helper functions called inside a stage are not separate stage executions;
the decorator checks Synpp's cache identity to avoid counting backend adapters twice.

The top-level stage package selects a separate monitor, persistent state and figure:

| Stage prefix | Report directory under the working directory |
| --- | --- |
| `sumo.` | `sumo_monitoring/` |
| `mode_choice.` | `mode_choice_monitoring/` |
| Any other opted-in package | `<package>_monitoring/` |

Each directory contains `state.json`, `performance.json` and `performance.png`,
plus timestamped run archives. Existing SUMO monitoring records remain usable
at their previous paths. Reports contain the module name, original run/job IDs,
stage wall time, sampled peak process-tree RSS, sampled CPU time and disk I/O.
A separate background worker per active package samples every two seconds.
Matplotlib rendering is serialized across workers.

A stage's full cache path identifies its record. Restarts retain earlier
measurements; rerunning a stage replaces that instance's previous attempt.
Running stages checkpoint every 30 seconds; stage entry/exit checkpoints are
immediate, atomic JSON writes. Previously running attempts are marked interrupted
when restored. Figures distinguish retained and current measurements. The combined
timeline excludes downtime and superseded attempts. Measurements include child
processes, and shared RSS pages can be counted more than once.

`monitor(context)` also supports a manually scoped operation; pass `name=` to
choose its full stage name explicitly. `export_monitor(context, destination)`
registers an export of the current package's reports. Its optional `module=`
selects a package explicitly. SUMO's existing output stage exports SUMO performance
figures as before; mode-choice figures stay in the working directory above.

SUMO-specific trip, teleport and bottleneck reporting remains in
`scripts.monitoring.quality.sumo`, separate from the generic performance sampler.
