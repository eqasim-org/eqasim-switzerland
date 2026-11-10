# GTFS transfer time map

Besides `transit_schedule_map.html`, `analysis.pt.transit_schedule.visualize`
writes `transfer_times_map.html` to the same folder. This map shows the minimum
transfer times of the cleaned GTFS (`data.gtfs.cleaned`) in the Geneva and
Lausanne perimeters. It replaces the former `data.gtfs.visualize` stage, whose
line and schedule maps were removed.

| Option | Default |
|---|---|
| `analysis.pt.transit_schedule_transfers` | `true` (set `false` to skip the map) |
| `analysis.pt.tpg_perimeter` | `spatial/MMT/CMDP_Limites_WG84.shp` |
| `analysis.pt.tl_perimeter` | `spatial/LausanneExtent/LausanneExtent.shp` |

The perimeters are the same settings as the TPG and TL comparisons, relative to
`data_path` or absolute. They are independent of the extent of the schedule map
(`analysis.pt.transit_schedule_extent_paths`). The stage also writes
`transfer_times.csv`: the rules shown, with their origin and the distance between
the stops.

The map is drawn by station, not by platform. The stops of a station are one point:
those with the same DiDok number (the `didok` column of `stops.txt`, or the UIC code of
an SNCF stop area such as `StopArea:OCE85011205`), else the same `parent_station`; a stop
with neither is its own station. The platforms of Lausanne and its SNCF stop area are
therefore one point, while `Lausanne, gare` (the buses and the M2, another DiDok number)
is another one.

The map shows:

- the stations (from zoom 14), coloured by the most common transfer time within the
  station: the rules between two of its stops (platform to platform) and within a stop
  (`from_stop_id = to_stop_id`). They are grey without one;
- the rules between two stations that have at least one stop inside a perimeter, as one
  line per direction, coloured by the most common minimum transfer time of its rules
  (under 1, 1–2, 2–3, 3–5, 5–10 and 10 or more minutes; each class can be hidden). Both
  directions are drawn side by side. The tooltip gives the interval of the times
  (for example `7–8.1 min`), the most frequent one, and the number of possible
  transfers (pairs of stops of `transfers.txt`) the line stands for;
- the rules computed by `data.gtfs.cleaned` (between feeds, set by hand or from a
  nearby stop) dashed, when its `diagnostics/transfers.csv` exists. A pair of stations
  joined by both given and computed rules has one line of each. Rules it removed are
  not shown.

When several rules join the same pair of stops (for example per trip), the shortest
time is kept. Clicking or searching a station (by name or stop id) lists its stops
with their platforms, the times within it, and its lines to and from the other
stations. `transfer_times.csv` still lists every rule between stops. Nothing is
flagged as implausible. The modes in the tooltips are read from `stop_times.txt`.

Outside the pipeline, pass `--gtfs-transfers <data.gtfs.cleaned cache directory>`
and one or more `--transfer-perimeter <polygon file>` to `visualize.py`. The map is
then written next to `--output`.
