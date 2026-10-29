# Cross-border activity timing

`data.cross_border.activities` adapts sampled microcensus chains to cross-border
locations. `retime_with_travel_times` estimates travel times from distance and
mode, works backwards from the first Swiss activity and forwards from the last,
and keeps inserted border stops at one second.

If inbound travel would require departure before the initial activity starts,
the Swiss activity block is delayed by the minimum necessary offset. Its
activity durations and relative times are preserved. The initial activity keeps
its start, and the return journey follows the delayed last Swiss activity.
The final activity retains its original end unless the return arrives later.
Other chains retain their Swiss activity times. Through traffic keeps its
original first departure and is scheduled forwards.

The stage logs the number of delayed chains and the maximum delay. This rule
applies to both MATSim and SUMO inputs. Rerunning the pipeline detects the stage
change and rebuilds dependent caches; existing cache files need no manual edits.
