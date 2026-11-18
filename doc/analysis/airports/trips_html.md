# Geneva and Zurich airport trip map

Generate the standalone HTML from the signed airport trip exports and the
municipality cache selected by a configuration:

```bash
uv run -m analysis.airports.trips_html config_dib_10.yml
```

The output is `<analysis_path>/airports/trips.html`. Use `--output PATH` to
choose another destination. The command reads existing inputs and does not run
the configuration's pipeline, change trip data, or scale demand again. It
rejects incomplete/corrupt trip exports and outputs from an older trip-model
version. If the inputs need rebuilding, select the normal Synpp stage:

```yaml
run:
  - analysis.airports.trips_html
```

Run it with `uv run -m synpp your_config.yml`. Synpp handles dependencies and
caching normally. The stage returns the HTML path. Census and spatial caches
remain read-only.

## Map and controls

The default display aggregates movements by municipality, airport and direction.
The non-airport point is the mean of the filtered trip endpoints; line width and
circle size show passenger count. Curved lines separate the two directions for
visibility; their shapes are display aids, not travel routes. Switch to
**Individual home / hotel endpoints** to inspect every generated location.
Coordinates are transformed from EPSG:2056 to WGS84 and rounded to six decimal
places for display; no trip sampling is applied by the report.

Filter Geneva/Zurich, direction, detailed or simulation mode, resident/visitor,
and ground departure hour. Negative departures and departures from the next day
are retained under **Previous / next day** and in **All hours**. Clicking a
histogram bar selects its departure hour. Departure times refer to ground trips,
not flights. Residents use households; visitors use hotels or dormitories, irrespective of
nationality. The report covers Swiss ground endpoints only.

The municipality table and mode shares follow the filters. Search the table by
name or municipality ID and click a row to zoom to its endpoint mean. The table
initially shows the largest 100 municipality–airport pairs; **Show all
municipalities** reveals the rest. CSV download includes all matching table
rows, regardless of the display limit. Table search affects the table/download;
it does not change the map filters. Counts represent sampled passenger movements,
not distinct people or vehicle counts. The sample fraction is displayed in the
header; there is no expansion to a full population.

## Sources and offline use

The HTML embeds all trip data, a simplified Swiss boundary, and Leaflet 1.9.4.
Generation downloads the Leaflet library and stylesheet from unpkg.com. Only
the swisstopo national-map background needs internet access when opening the
report. Its attribution is visible, and background opacity is adjustable.
Boundary, flows, endpoint markers, filters, tables and CSV remain usable if
background tiles fail.

**Data and modelling notes** records the source export checksum/signature,
creation time, trip-model version, mode-distance coefficients and per-airport
requested/feasible/assigned mode totals. No private configuration is embedded.

The airport modelling assumptions and source references are documented in the
[airport trip guide](../../synthesis/airports/trips.md).

The airport-side coordinates now distinguish PT interchanges from road parking
points. Both markers are shown for each airport; selecting a mode hides the
inapplicable marker. Flow aggregation retains the access-point distinction,
while the municipality table sums both. Visitor popups distinguish hotels from
dormitories. PT uses A/B/C first and class D only when necessary to meet the
airport PT quota, before falling back to car or car passenger.

Resident endpoint popups now show the selected synthetic person ID, age, sex and
employment. The report includes aggregate resident-selection diagnostics and
coarse demographic probabilities; visitor person IDs remain null.
