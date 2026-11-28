# SUMO monitoring and simulation quality

Performance monitoring is shared with mode choice; see the
[reusable stage monitor](../scripts/monitoring.md). SUMO simulation-quality
readers live under `scripts.monitoring.quality.sumo`.

All SUMO stages decorate `execute(context)` with `@monitored`, imported from
`scripts.monitoring`. A single background worker samples Linux
process counters every two seconds. It includes the pipeline's Python process
and descendant SUMO, Java and native helper processes. Stage boundaries also
trigger samples. On a restart, earlier stage measurements are restored from a
persistent checkpoint. Stages reused by Synpp retain their recorded time and
memory; stages that execute again replace their previous measurement.

The worker records elapsed stage time, peak sampled process-tree RSS, sampled
CPU time and operating-system disk I/O counters. RSS includes shared pages more
than once when processes share memory, and short-lived subprocess peaks may fall
between samples. These are diagnostic measurements, not Slurm accounting totals.
Sampling and rendering run on the background worker. A small JSON checkpoint is
written synchronously at stage entry and exit so that a later exception cannot
lose completed measurements. Running stages are also checkpointed every 30 seconds.

## Resume after a failure

The checkpoint is a dictionary stored at:

```text
<working_directory>/sumo_monitoring/state.json
```

Its key is the full Synpp stage cache path. Repeated execution of the same stage
instance replaces its duration, memory samples and status. Different parameterized
instances, such as the native index/cut/export builds, remain separate. Records
belong to the working directory; measurements for other configuration variants
are retained as separate stage instances too. Each record keeps its original run
ID, job ID and CPU allocation.

Updates merge under a file lock and atomically replace the JSON. No large scenario
tables are stored in this checkpoint. The first use also imports existing per-run
`performance.json` reports, if present, so measurements made before this feature
can be reused. Removing the entire `sumo_monitoring/` directory starts fresh.

For example, if assignment finishes and regional cutting fails, the next run
keeps assignment's original time and peak memory, replaces the failed cutting
attempt with its new measurement, and adds later stages as they complete. The
failed attempt's duration is not added to the replacement duration. An unfinished
measurement from a killed process is shown as **interrupted**, using its last
checkpoint as a lower bound, until that stage runs again. Same-process notebook
reruns are also recognized as separate executions.

The figure distinguishes **retained** measurements from those collected in the
current run. Its memory timeline combines the latest measurement for each stage,
with separate segments; downtime between jobs and superseded attempts are
excluded. It is a composite workflow profile, not a continuous trace of one job.
Original per-run archives are retained.

A figure update is requested at stage boundaries and coalesced to at most one
update every five seconds. The latest combined report is available at:

```text
<output_path>/<output_id>/monitoring/sumo.png
<working_directory>/sumo_monitoring/performance.json
```

JSON stays in the working directory; the figure is written directly to the output
folder without an export step. Each execution also writes a JSON archive under
`sumo_monitoring/<timestamp>_<pid>/`. Normal interpreter shutdown finalizes the
last update. An abrupt termination leaves the latest complete checkpoint and
figure. Monitoring failures are logged without replacing stage exceptions.

The figure includes `sumo.output` itself and is finalized asynchronously at normal
process exit. SUMO’s simulation-quality reports are still handled by that stage.

## Quality report

The independent `sumo.monitoring` stage depends on `sumo.run`. It streams the
regional simulation's `tripinfo`, `summary`, `edge_data` and `statistics` outputs,
without reading national routes or modifying cached results. `sumo.output`
copies its products to `<output_path>/sumo/quality/`:

- `index.html`: local report with the overview figure and bottleneck table.
- `quality.png`: vehicle outcomes, queues, largest road delays and a regional map.
- `quality.json`: simulation statistics and report definitions.
- `bottlenecks.csv`: normal edges ranked by accumulated vehicle time loss, with
  mean speeds, waiting time, teleport counts and the worst reporting interval.

Completed, unfinished, undeparted and vaporized trips are counted separately.
Travel-time summaries use completed trips, with one-second bins for median and
95th-percentile durations. Mean edge speeds are weighted by sampled
vehicle-seconds. Summed edge `timeLoss` is a total across vehicles, not a per-trip
average. Teleport totals and reasons come from SUMO's structured `statistic-output`;
summary counters are not added together. For older cached outputs without this
file the total is explicitly unavailable. A high-delay edge is evidence of a
bottleneck, not proof that its signal controller caused it.

See [SUMO statistics](https://sumo.dlr.de/docs/Simulation/Output/StatisticOutput.html)
and [edge measures](https://sumo.dlr.de/docs/Simulation/Output/Lane-_or_Edge-based_Traffic_Measures.html).

## Preserve assigned routes at departure

Regional demand export sets `sumo_initial_route_retention: 0.8` by default.
A stable, seeded draw selects approximately 80% of vehicle IDs and adds:

```xml
<param key="device.rerouting.pre-period" value="0"/>
```

These vehicles keep their complete assigned route before departure. All vehicles
remain eligible for the usual GPS device (`sumo_rerouting_probability: 1.0`),
and reroute after departure at `sumo_rerouting_period: 60` seconds using updated
network observations. Retention is an initial-route policy, not a promise that
80% of paths remain fixed throughout the simulation. The remaining approximately
20% may reroute before insertion using SUMO's default pre-period. Explicit
per-vehicle `device.rerouting.pre-period` parameters are preserved.

Set retention to `1` to protect every initial route, or `0` to let SUMO's
pre-departure rerouting operate for all equipped vehicles. The selection is
identical for native and Python exporters and independent of batching/thread
counts. It changes only the final regional demand export, leaving national
assignment and cutting separately cached. Retention applies to pipeline-produced
regional demand, not arbitrary user-supplied XML in standalone config stages.

Both regional presets enable rerouting and `sumo_time_to_teleport: 600` by
default. Teleportation can clear a deadlock but distorts simulated travel times;
the report makes these events visible. Neither rerouting nor teleportation
replaces validation of demand, junction capacity and traffic-light programs.
See [SUMO routing-device settings](https://sumo.dlr.de/docs/Demand/Automatic_Routing.html).
