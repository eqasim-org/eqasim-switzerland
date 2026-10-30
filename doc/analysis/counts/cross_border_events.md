# Shared cross-border event counts

`analysis.counts.cross_border_flow_events` streams simulation events once for the
car and PT maps. Its cached tables contain unscaled link-entry counts and PT
rider counts by stop pair/hour. Each consuming map applies `input_downsampling`
once. Counts are traversals, not unique daily travellers.

The groups are foreign cross-border agents, Swiss-resident cross-border agents,
and non-cross-border French residents. Total flow also includes travellers outside
these groups. Car entries retain the existing `:car` vehicle convention; this is
not a freight-volume report.

PT driver identities come from `TransitDriverStarts` and are excluded from
passenger counts. Only transit vehicles retain onboard sets; empty sets and parsed
XML elements are released as the stream advances. A vehicle changing line uses
its current driver-start event, translated with the schedule's line labels.
Arrivals are counted before alighting, so a rider belongs to the segment just
completed. The hourly bin is the arrival hour, folded modulo 24 as before.

Input discovery uses the shared gzip/Zstandard-aware output resolver. Both maps
can be selected together through `analysis.pt.all_comparisons`. Map extents are
configured by `cross_border_flow_extents`, with Geneva and Lausanne defaults
relative to `data_path`. The basemap uses swisstopo tiles without a private API
key. Road-flow map geometries are node-to-node inspection lines keyed to current
simulation link IDs; detailed geometry remains available for count-station
matching and is not overwritten.

Local fixtures check driver exclusion, the three traveller groups, transfers and
line changes, empty inputs, parser cleanup, gzip/Zstandard streams and output
resolution. No full event file or simulation rerun is needed for these checks.
