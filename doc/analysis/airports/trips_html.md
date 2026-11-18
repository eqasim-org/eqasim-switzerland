# Geneva and Zurich airport demand explorer

Run the public analysis stage from the repository root:

```bash
uv run -m synpp config_dib_10.yml --run analysis.airports.run
```

It reads `synthesis.airports.trips`, municipality boundaries and canton
boundaries. Synpp runs or reuses their dependencies normally. The stage checks
known airports, directions, modes, resident IDs, both endpoint coordinates and
Swiss canton membership before creating `<analysis_path>/airports/trips.html`.
It returns the HTML path. Opening the file does not require a running server.

`analysis.airports.trips_html` remains available as the older stage name. Both
entry points use the same renderer and invalidate their report caches when the
renderer or HTML template changes. They never scale demand a second time.

## Map views

Choose **Map display** in the left sidebar:

- **Canton flow lines** (default): each Swiss canton connects to each airport.
  The anchor is the mean of the selected home/accommodation points for that
  canton and airport. This keeps a canton split between airport catchments from
  appearing to send passengers from its opposite side.
- **Municipality flow lines**: each municipality connects to each airport.
  Its anchor is the mean of the selected home/accommodation coordinates.
- **Actual origin–destination lines**: every selected movement is drawn between
  its generated origin and destination. No points are resampled or discarded.
- **Individual home / hotel endpoints**: inspect each sampled home, hotel or
  dormitory, including the selected resident's ID and available demographics.

Line widths show sampled passenger counts. By default, darker blue also means
more movements; **Line colour** can switch to direction colours. To-airport lines
are solid and from-airport lines dashed. Both directions use straight connections.
Aggregations retain the PT/road airport-access distinction, so a line counts one
region, airport access point and direction. The table sums both access points.
Colour and width ranges adapt to the currently selected flows.

Aggregated and individual lines are OD links, **not routed road or rail journeys**.
Only one airport label is shown per airport, at its road access point, labelled
GVA or ZRH. Lines retain the actual PT/road endpoints used by the generated trips.
Click a line or endpoint for its details.

## Filters, statistics and table

Filter by airport, Swiss canton, direction, resident/visitor profile, simulated
or detailed access mode, and ground departure hour. **Previous / next day**
retains negative departures and times beyond midnight; **All hours** includes
them too. Clicking an hourly bar selects that hour.

The left sidebar updates with the current selection:

- Total sampled movements and to/from counts.
- Geneva and Zurich totals, residents, visitors and to/from splits separately.
- Combined resident and visitor counts.
- Ground departure histogram and adjacent-day count.
- Counts and percentages for all five simulated modes and seven detailed modes.

These are passenger movements at the population sampling rate, not vehicles,
full-population expansions or necessarily distinct visitors. Resident travellers
are selected without replacement. Visitor/resident describes the accommodation
profile, not nationality.

The table follows the displayed aggregation: cantons in canton view,
municipalities in the other views. It reports each region–airport pair's
counts, PT share, visitor share and mean Euclidean distance. Search by name or
ID, click a row to zoom, or download the filtered table as CSV. The initial
100-row limit only affects display; downloads include all matching rows.
Table search affects the table/download, not the map filters.

## Running only airport stages with existing population caches

For a deliberate refresh of airport processing while reusing existing census,
spatial and synthetic-population inputs, use Synpp debug mode:

```bash
uv run -m synpp config_dib_10.yml --debug true \
  --run data.statpop.municipality_counts \
  --run data.airports.geneva.model \
  --run data.airports.catchments \
  --run data.airports.geneva.ods \
  --run data.airports.zurich.ods \
  --run data.airports.ods \
  --run data.microcensus.airports \
  --run synthesis.airports.locations \
  --run synthesis.airports.trips \
  --run analysis.airports.run \
  --config:json airport_model_overwrite false
```

Debug mode executes the requested stages and connecting dependencies. It reuses
cached boundary inputs **without validating them**, and may choose an older
cache when the current hash is unavailable. Check its logs and confirm the
population sample, seed and scenario match the intended inputs. It does not
start matching or a simulation in this selection. Frozen airport allocation
coefficients are loaded, not recalibrated.

## Standalone rendering from saved trips

To inspect existing signed trip exports without executing demand stages:

```bash
uv run -m analysis.airports.trips_html config_dib_10.yml
```

Use `--output PATH` to choose another output. This command checks export
checksums and the trip-model version, then reads the municipality and canton
caches selected by the configuration. If either spatial cache is missing,
run the Synpp analysis stage instead.

## Background map and offline use

The report embeds all trip data, a simplified Swiss outline and Leaflet 1.9.4.
Generation fetches Leaflet from unpkg.com. Only the optional swisstopo national
map needs internet access when opening the HTML. Attribution is visible and
background opacity is adjustable. Filters, OD links, markers, statistics,
tables and downloads remain usable if tiles are unavailable.

**Data and modelling notes** includes source metadata, creation time, sampling
rate, trip-model version, mode-assignment diagnostics, unresolved demand and
resident-selection assumptions. No private configuration is embedded.
See the [demand walkthrough](../../data/airports/demand_walkthrough.md) and
[trip guide](../../synthesis/airports/trips.md) for the source evidence and modelling assumptions.
