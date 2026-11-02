# French population explorer

Run independently, without building a population or simulation:

```bash
uv run -m analysis.french_activities --config config_dib_10.yml
```

Or add `analysis.french_activities` to the Synpp `run` list. Inputs come from
`external_population_folder`; output is
`analysis_path/french_activities/french_activities.html`.

The stage reads the source French population (not the sampled/merged simulation
population). Home GeoPackage coordinates use the layer CRS; activity WKT uses
EPSG:2154 as in `data.external_population.read_outputs`. Trips join explicit
preceding/following activity indices, not row order. Missing homes and trip
endpoints are excluded and reported. Non-travellers remain selectable residents.
Identical duplicate activity locations are collapsed; conflicting locations for
the same person/activity index raise an error rather than creating extra trips.

Name a polygon, click **Draw polygon**, click vertices on the map, then
**Finish**. Repeat to build a library of polygons. Assign any saved polygon to
the home, origin A or destination B filter; the same polygon can serve several
roles. Saved polygons can be renamed, redrawn or deleted. Clearing the geographic
filters keeps the library. Polygons are retained only while the page is open.
A home polygon selects residents; origin A and destination B polygons filter
their individual trips. Choose A → B or either direction, select modes and
origin and destination activity purposes, and enable trip display. The two
purpose filters are independent and both must match: for example, origin `home`
and destination `work` selects home-to-work trips. They use `preceding_purpose`
and `following_purpose` from the source trips. Even when either geographic
direction is enabled, purposes refer to the actual trip origin and destination.
Lines are straight origin–destination connections, not routed paths. Origins
are orange, destinations green, homes blue. Selecting no modes or purposes
shows no trips. Filters intersect; absent polygons impose no restriction.
Use the trip-line controls to change their color, width and opacity.

The person-attribute filter exposes the raw person CSV columns and values without
recoding them. Select one attribute, choose its values or numeric range, then
apply it. The panel reports the inferred type, categories or numeric range, and
empty-value count. Text filters use case-sensitive substring matching. To select
people with some car availability, for example, choose `car_availability` and
check `all` and `some`. The filter applies to residents and their trips, together
with the geographic, mode and purpose filters. Empty values can be included
explicitly; strings such as `False` and `none` remain literal categories.

All persons are loaded by default. `analysis.french_activities.max_persons`
(or CLI `--max-persons`) optionally selects a deterministic person sample using
`random_seed`; 0 means all persons. Trips of the sampled persons are retained.
The browser drawing limit is separate: it caps drawn symbols/lines but selection
totals still include all matching loaded records. Large exports can require
substantial browser memory; use the person sample if necessary.

The HTML embeds compressed data and can be opened directly in a modern browser
with `DecompressionStream` support. Internet is needed for Leaflet and map tiles.
Population records are not sent to a server, but map tile requests reveal the
viewed region to the tile provider. Treat the generated HTML as population data
when sharing it. A `--output` argument can override the destination for previews.
