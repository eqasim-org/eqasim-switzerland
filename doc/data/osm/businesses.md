# Simulation-region businesses with typical weekday opening hours

Add `data.osm.businesses` to a Synpp `run` list. The stage reads the files in
`osm_file` (one filename or a list). When `latest_osm_for_correction` is
provided, it replaces the entry whose filename contains `switzerland`
(case insensitive), or is added if no Swiss entry exists. Otherwise the
configured Swiss extract is retained. At most one Swiss entry is allowed.
Relative filenames resolve under `data_path/osm`.

The stage uses `data.spatial.simulation_region`, including its configured
Swiss and foreign areas. Extract header bounding boxes are checked against
that region before scanning. Non-overlapping files are skipped; missing
header bounds cause a conservative scan. Bounding boxes can include areas
outside the extract's actual footprint, so some overlapping candidates can
produce no businesses. Each business's representative point must lie inside
or on the boundary of the actual simulation region.

Overlapping extracts are deduplicated by typed OSM ID (`n`, `w`, `r` prefixes).
The selected Swiss extract has priority, followed by the other files in their
configured order. Duplicate paths are processed once. `source_file` records
the winning source on each exported business. All source-file paths, sizes
and modification times participate in cache validation; the region is also
a stage dependency.
Only businesses with successfully parsed, fully known weekday opening hours
are exported. Missing, invalid, unsupported and uncertain schedules are excluded.

```yaml
run:
  - data.osm.businesses
config:
  data_path: /path/to/input/data
  osm_file:
    - switzerland-latest.osm.pbf
    - rhone-alpes-latest.osm.pbf
  # Optional replacement of the Swiss entry only:
  # latest_osm_for_correction: switzerland-newer.osm.pbf
  # Optional calendar for resolving holidays, explicit dates and sun events.
  data.osm.businesses.calendar_year: 2026
```

There is no reference-week setting. Usual hours are selected from **open
Monday–Friday dates across all seasons**, in `Europe/Zurich`. The evaluation
calendar defaults to 2026; the optional `calendar_year` must be an integer from
1901 to 9998. It resolves holidays, explicit date/year restrictions and sun
events reproducibly without relying on today's date.

For `Jun-Aug Mo-Su 11:00-24:00`, the resulting usual hours are **11:00 to
00:00, with closing day offset 1**, even when the stage runs in November.
Closed months do not outweigh open dates. The original seasonal rule is
retained: the usual clock times do not imply that this business opens all year.

## Categories and location

- `shopping`: active retail `shop=*` tags, including `shop=supermarket`, plus
  `amenity=pharmacy` and `amenity=fuel`.
- `leisure`: bars, pubs, cafés, restaurants, fast food, cinemas, theatres,
  nightclubs, selected sports/fitness venues, museums, heritage sites and zoos.
- `mixed`: both shopping and leisure evidence on the same OSM object.
- `other`: accommodation, offices, crafts and services such as hairdressers,
  banks, post offices, doctors, dentists, clinics, vehicle rentals and driving schools.

The shared mapping in `data.activity_purposes` aligns retail and leisure donors
with STATENT NOGA 2008 destination purposes and the microcensus `shop`/`leisure`
labels. Retail services and accommodation are not retail/leisure hour donors.
`offers_shop` and `offers_leisure` are the canonical flags; `is_shopping` and
`is_leisure` remain compatibility aliases. These are modelling rules, not an
official business register or a complete inventory. Bare `building=retail` or
`building=commercial` does not establish an individual business. Disused,
abandoned and vacant shops are excluded. Public/private access tags are retained
without assuming every place is accessible to all visitors.

Nodes, polygon ways, multipolygon relations and tagged non-closed ways are
supported. Polygon representative points are calculated in EPSG:2056, keeping
them on the mapped footprint. IDs preserve object type (`n123`, `w123`, `r123`).
Distinct objects remain distinct even if names coincide; there is no speculative
merging of shops within shopping centres or similarly named branches.

## Stage result and files

The return value is a dictionary with `businesses`, `weekday_hours` and
`metadata`, each an absolute file path in the stage cache:

- `businesses.parquet`: clean GeoParquet point layer in EPSG:2056, with projected
  `x`/`y`, WGS84 `lon`/`lat`, names, categories, category flags, raw business tags,
  `tags_json`, raw `opening_hours`, and the typical weekday columns below.
- `weekday_hours.parquet`: five representative weekday profiles per retained
  business, keyed by `osm_id` and `weekday`; there is no dated reference week.
- `metadata.json`: configured/processed/skipped source paths, duplicate count
  and precedence, evaluation year, retained category counts, excluded schedule
  counts, extraction/evaluation timings and cache statistics.

Each weekday profile includes `open_days` (dates with any known opening),
`evaluated_days` (all dates for that weekday in the year), and
`schedule_occurrences` (dates supporting the representative schedule). For a
summer-only business these counts expose its limited season even though its
usual clock hours are populated. These profiles summarize openings; they are
not a dated availability forecast for every week of the year.

CSV files are not produced. Missing-hour places are discarded before geometry
construction and tag serialization. Repeated non-solar schedules share a bounded
evaluation cache; location-specific sunrise/sunset rules retain their coordinates.
Ordinary weekday/time rules are evaluated once and weighted by the number of
each weekday in the year. Calendar-dependent rules use one interval iterator
across the full year and a forward scan of overlapping intervals. Only compact
summaries are cached/exported, rather than hundreds of daily rows per business.

## Typical weekday columns

| Column | Meaning |
| --- | --- |
| `opens_at` | First opening time of the typical schedule, as a Parquet time value |
| `closes_at` | Final closing time of that schedule, as a Parquet time value |
| `closes_day_offset` | Closing date offset: 0 for the same day, 1 for the next day |
| `typical_status` | `open` or `closed` |
| `typical_intervals` | JSON intervals in minutes from the opening day's midnight; overnight ends can exceed 1440 |
| `typical_open_minutes` | Total open duration, excluding lunch breaks |
| `typical_weekday_count` | Number of distinct weekday names sharing this schedule |
| `typical_weekdays` | JSON list of the matching weekdays |
| `typical_occurrences` | Number of open weekday dates sharing this schedule in the evaluation calendar |

The typical schedule is the most frequent **complete opening schedule among open
weekdays** across the whole year, with closed days and seasons excluded. Opening and closing times
are selected together. A Wednesday-only shop uses its Wednesday schedule; a shop
open Wednesday and Friday with the same hours uses those hours with a count of 2.
Friday or Monday exceptions do not override a schedule shared by other open days.
Ties prefer Monday–Friday order, then the earliest matching date.

For `Mo-Th 08:00-18:00; Fr 08:00-20:00`, the typical times are 08:00 and 18:00,
with a count of 4. A lunch break remains in `typical_intervals`; `opens_at` and
`closes_at` describe the overall span, not continuous opening during that span.
An evening business opening at 18:00 and closing at 02:00 has
`closes_day_offset=1`. Sessions are assigned to the weekday on which they start;
carry-over from the previous evening is not treated as a new midnight opening.
Continuous all-day opening uses 00:00 to 00:00 with offset 1 and 1440 open minutes.

Closed weekdays never outweigh a known opening. Only businesses with no weekday
opening at all have `typical_status=closed`, null opening/closing times and zero
typical open minutes. They remain in the dataset because their schedules were
successfully parsed. If the only weekday opening is carry-over from Sunday
night, that Monday interval is used rather than reporting the business closed.

Opening and uncertain intervals are JSON arrays of `[start_minute, end_minute]`
pairs measured from local midnight, with an exclusive end and 1440 meaning
midnight at the end of the day. Split hours retain their gaps. Overnight hours
are clipped to each calendar day, including carry-over from Sunday into Monday.
For example, `[[480,720],[780,1080]]` means 08:00–12:00 and 13:00–18:00.
The weekday table also contains `session_intervals`, assigning complete sessions
to their opening day in the same way as the typical schedule.

Exported daily statuses are `open` or `closed`; `open_minutes` and
`known_open_minutes` count established calendar-day opening. The parser still
distinguishes `unknown`, `missing`, `invalid` and `unsupported` internally, but
these businesses are excluded and counted in metadata. Raw tags are retained
for exported businesses.

The stage uses [opening-hours-py](https://pypi.org/project/opening-hours-py/)
for the [OSM opening-hours syntax](https://wiki.openstreetmap.org/wiki/Key:opening_hours),
including country holiday calendars inferred from business coordinates for
`PH` rules, and location-specific sun events. French businesses therefore use
French public holidays. The country calendar does not establish every
cantonal/local holiday. School-calendar
rules, open-ended closing times, comment-only schedules and alternative/fallback
expressions (`||`) are excluded as unsupported. Fallbacks can describe different
services (such as a staffed bank versus its ATM) and are not collapsed into one
firm schedule. Appointment comments are treated as uncertain. Always inspect
the raw rule and coverage statuses before using opening hours as constraints.

The optional `latest_osm_for_correction` registration is shared consistently
with network cleaning and SUMO. Those existing stages retain their historical
Swiss filename fallback when the setting is omitted; this new stage instead
uses the explicitly configured `osm_file` as described above.
