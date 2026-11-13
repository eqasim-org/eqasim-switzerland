# Foreign freight regions and selected border anchors

Generate the standalone HTML using a pipeline configuration:

```bash
uv run -m analysis.freight.border_crossings config_dib_10.yml
```

Output: `<analysis_path>/freight_border_crossings/index.html`, with aggregate CSVs
and `provenance.json` beside it. The module also works as a Synpp stage:

```yaml
run:
  - analysis.freight.border_crossings
```

The standalone command reuses the comparison module's freight-only cache reader.
It resolves exact configured prerequisite caches, executes the NPVM freight stages
and the legacy wrapper, and never launches population/MATSim stages or rewrites
upstream caches. Missing prerequisite caches produce an explicit error. The
production `use_npvm_freight` selection does not exclude either comparison method.

## What is mapped

- Original foreign NPVM zones and legacy NUTS regions, including their names/IDs.
- The actual foreign endpoint coordinates stored in generated trips. These are
  Swiss-border anchors: eligible network-derived crossings in default NPVM mode,
  scored interview road points when `npvm_freight_use_interview_places: true`,
  or geometric projections in legacy/optional NPVM mode. They are
  not observations of road/customs crossing flows.
- The Swiss boundary and, for a selected region/anchor, dashed connections from
  the original source centroid to the chosen anchor. The lines explain spatial
  assignment; they are not trip paths or network routes.
- Optional straight OD lines for generated incoming/outgoing/foreign-to-foreign
  trips. Domestic-to-domestic trips are excluded from this report.

NPVM original source-zone IDs are expanded from sampled OD counts in trip-row
order. Legacy NUTS IDs are recovered by replaying the original multinomial OD
sampling with cached demand and the configured seed, checking component coordinate
alignment against the combined legacy trip table. The analysis does not recompute
or replace anchor coordinates. Original polygons are simplified by 500 m only
for display; original centroid and actual anchor coordinates retain full precision.
Additional NPVM zones without published polygons are displayed as circles at their
supplied source coordinates.

## Controls and counting

Choose NPVM, Legacy, or Both. Filter incoming ODs, outgoing ODs, or foreign-to-foreign
ODs. Click a source polygon, use the region list, or click an anchor/table row.
Selecting an anchor highlights all its contributing foreign regions. Coordinates
shared exactly by several regions are merged into one marker **within each
method**. Use the method selector or source-region list for overlapping polygons
and coincident markers from different methods.

Each incoming or outgoing OD contributes one foreign endpoint occurrence. A
foreign-to-foreign OD contributes two: transit entry and transit exit. It remains
one trip in the trip counter even when its two anchors coincide. These endpoint
occurrences are not observed or simulated link-crossing events, and a foreign-to-
foreign OD does not establish that its subsequently routed path crosses Switzerland.
NPVM domestic geography includes Liechtenstein/enclaves, while legacy domestic
identity uses Swiss NUTS IDs.

The four directional columns in the source-region table contain full region
counts. Its selected-direction total and anchor table follow the active selection
and direction. Tables display up to 150 rows; CSVs contain complete aggregates.
Searching the region table does not change the map selection. Clicking an anchor
row zooms to its actual generated coordinate; fitting regions helps inspect the
foreign regions associated with a shared anchor.

Distances to the Swiss boundary are computed from actual endpoint coordinates
in EPSG:2056. Anchors over 1 m from the boundary are counted in diagnostics and
outlined red. Being on the boundary does not demonstrate road accessibility.
No observed crossings, road restrictions or border counts are used to select them.

## Backgrounds and exports

The default is the attributed swisstopo national map with background-opacity
control. A verified optional NASA Blue Marble background provides global imagery
when inspecting distant foreign regions. It uses public GIBS WMTS tiles without
requesting an API key; it is imagery, not a current road map. See
[NASA's GIBS access documentation](https://nasa-gibs.github.io/gibs-api-docs/access-basics/).
Representative tiles for both backgrounds were checked when creating the report.
Polygons, anchors, connection lines and tables remain available if tiles fail;
ongoing online tile availability is not guaranteed. Leaflet and report data are
embedded in the HTML; generation downloads Leaflet from unpkg.com.

The CSV files record source-region totals, anchor coordinates/totals, and every
source-region-to-anchor allocation by direction. Counts refer to the configured
population sample without further downsampling or expansion. Source cache names,
scaling assumptions and read/recomputed input provenance are recorded in JSON.

The summary table above the map shows entering, leaving and transit trip totals
for both models, independent of map filters. Entering and leaving exclude transit;
transit ODs are counted once. These are totals at the configured sample fraction
and use the original source-zone domestic/foreign classification.
