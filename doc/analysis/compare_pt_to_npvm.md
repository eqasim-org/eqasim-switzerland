# PT and car demand versus NPVM zones and municipalities

Run the analysis against the existing simulation selected by the configuration:

```bash
uv run -m synpp config_dib_10.yml --run analysis.compare_pt_to_npvm.run
```

The stage reads `output_trips.csv` (`.gz`, `.zst` or plain), using
`analysis.counts.simulation_path` if supplied. Otherwise it uses
`output_path/output_id/simulation_directory`. It does not rebuild a population,
run a router or launch MATSim.

The **Transport mode** switch selects public transport or passenger cars;
**Geographic level** selects NPVM zones or municipalities. Percentage colours,
hover totals, incoming/outgoing OD controls and source selection apply to all
four combinations.

## Comparison basis

The reference is the public [ARE NPVM 2023 release](https://zenodo.org/records/15755932),
`3_Wegematrix_OEV_NPVM_2023.zip`, containing the combined `ÖV` demand matrix in
`OEV.omx`. Its source model is `OEV_NPVM_2023_DWV_final_export.ver`: an average
weekday (DWV). This is **all public transport**, not a rail-only matrix. The
[official report, section 4.2, table 5](https://www.are.admin.ch/dam/de/sd-web/4KquT1YIdGEZ/Aktualisierung%20NPVM%202023%20-%20Schlussbericht_Publikation_20260212.pdf)
identifies the model's train, bus, tram, metro, boat, funicular and cable-car
systems. A train-plus-bus journey counts once, rather than once per mode or
boarding. The reader checks the matrix `CODE` when provided, rejecting an
explicitly different mode such as car or freight.

Both matrix axes follow the **original NPVM zone numbers** from `lookup/NO`.
In NPVM-zone view, zones remain separate even when they share a municipality. Only matching zones
from the configured domestic and foreign NPVM geopackages are used; the
domestic file includes Liechtenstein/enclave coverage.

For PT, simulation trips with main mode `pt` or `pt_loop` are assigned by their original
LV95 origin/destination coordinates to the NPVM polygons. All resident groups
and nationalities are retained. An endpoint exactly on multiple boundaries
selects the smallest zone ID deterministically. There is no nearest-zone
snapping across borders. Unassigned endpoints are excluded from both compared
matrices, with excluded demand explicitly reported. The cross-border checkbox
controls whether mapped foreign endpoints contribute to the comparison.

Simulation counts are divided by `input_downsampling`. NPVM demand already
represents full demand. This assumes the configured sampling fraction describes
the supplied simulation. No adjustment is made between NPVM 2023 and the
synthetic population's target year. Intrazonal demand is included by default.

The **Geographic level** switch selects NPVM zones or municipalities in the
same HTML. Municipalities use the official `ID_Gem` crosswalk in the NPVM
polygons. Their shapes are unions of those polygons; both OD axes are summed
by municipality. Both views retain exactly the same journeys and full totals.
Foreign areas retain their original NPVM regions at municipality level: the
reference does not identify their demand by foreign municipality. No second
spatial assignment or sampling occurs. Within-municipality journeys
include journeys between different NPVM zones in the same municipality.

## Cross-border checkbox

**Hide foreign-region polygons** is an independent display option, enabled by
default. It hides foreign polygon fills and outlines, and disables their
polygon hover/click interactions. It preserves demand totals, colours in the
domestic area, the selected OD source and all OD lines, including lines to or
from hidden regions. Foreign regions remain accessible through the search and
OD table when cross-border demand is enabled. The mask persists across mode
and geographic-level switches. Use **Include cross-border trips** to change
which trips enter the comparison; the polygon mask changes only visibility.

**Include cross-border trips** is enabled by default and applies identically
to PT/car demand from NPVM and the simulation, at either geographic level:

- **On:** retain domestic/domestic, domestic/foreign and foreign/domestic OD
  pairs. Both incoming and outgoing international trips are included.
- **Off:** retain only OD pairs with both endpoints in the domestic NPVM area;
  hide foreign polygons, remove foreign OD lines and exclude foreign areas from
  search. Turning the checkbox off clears a selected foreign area, while a
  domestic selection and its source/direction are preserved.
- **Always excluded:** foreign/foreign pairs. Endpoint data alone cannot
  distinguish trips routed through Switzerland from trips staying abroad.

The foreign geopackage contains domestic zones too; duplicate domestic IDs are
removed, and domestic geometry is preferred for exact boundary ties. Published
foreign polygons partition the surrounding countries into 710 additional NPVM
regions. Their geometries are reprojected to LV95 before simulation matching.
Foreign regions keep their NPVM IDs and polygons in the municipality view;
no within-region demand is invented or redistributed to foreign municipalities.

Some OMX zone IDs have no published polygon. Their demand remains excluded and
is reported in component audits, along with simulation coordinates outside the
mapped coverage. Points are not assigned to arbitrary catchments for those
zones. This is a remaining coverage limitation, not zero demand.

Sparse exports contain the mapped domestic plus incoming/outgoing cross-border
scope. Summary CSVs include `is_domestic`, full-scope origin/destination totals,
and `*_domestic_origins`, `*_domestic_destinations`, `*_domestic_intrazonal`
for the unchecked state. `metadata.json` separately records domestic demand,
cross-border demand, foreign-to-foreign exclusions and unmapped demand.
Foreign polygons are hidden in the unchecked state, but matrix indices remain
stable. Repeated checkbox changes do not re-read or regenerate the data.

## Passenger-car comparison

The official road archive, `3_Fahrtenmatrizen_Strasse_NPVM_2023.zip`, provides
separate passenger-car, van and heavy-goods matrices. Only `PW_Binnen.omx`
(matrix `20`, `PW Binnen Aggregat (korr)`) and `PW_QZD.omx` (matrix `7`,
`PW QZD [KfzWege]`) are summed. Both use their own explicit zone lookups;
`LI_Binnen`, `LI_QZD`, `LW` and `LZ` are never added. See the
[official data description](https://zenodo.org/records/15755932/files/00_NPVM_2023_Datenbezug.pdf?download=1)
for vehicle categories and units. Only the two PW files are extracted into
private analysis storage; shared inputs are unchanged.

Car demand is **vehicle trips**, while PT demand is **person journeys**.
Simulation car demand uses `car` and `car_loop` driver trips, excluding
`car_passenger` / `car_passenger_loop` to avoid counting vehicle occupants as
additional vehicles. Truck modes are excluded. `lcv_` / `freight_` person IDs
and trips with freight origin/destination activity types are explicitly
excluded even if their main mode is `car`. No resident-nationality filter is
applied. Counts use the configured sample expansion.

The official PW domestic matrix contains numerical zeros around `-1e-16`.
The reader sets negative entries within `1e-12` of zero to zero and records
the corrected-cell count in its component audit. Material negative values
and nonfinite values still raise errors. Source files are not edited.

Switching PT/car preserves the current area selection, OD source, direction,
viewport and colour limits, allowing comparison of the same area across modes.
Mode-specific descriptions, units, totals and exclusions update together.

## Inputs and options

Existing settings `data_path`, `analysis_path`, `input_downsampling` and the
shared simulation-path settings supply paths and sample expansion. Additional
options are:

| Option | Default |
| --- | --- |
| `analysis.compare_pt_to_npvm.matrix_path` | Empty: obtain the official reference |
| `analysis.compare_pt_to_npvm.matrix_key` | `1` |
| `analysis.compare_pt_to_npvm.car_domestic_matrix_path` | Empty: official `PW_Binnen.omx` |
| `analysis.compare_pt_to_npvm.car_domestic_matrix_key` | `20` |
| `analysis.compare_pt_to_npvm.car_external_matrix_path` | Empty: official `PW_QZD.omx` |
| `analysis.compare_pt_to_npvm.car_external_matrix_key` | `7` |
| `analysis.compare_pt_to_npvm.foreign_zones_path` | `npvm/Verkehrszonen_Ausland_NPVM_2023.gpkg` |
| `analysis.compare_pt_to_npvm.zones_path` | `npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg` |

Configured input paths may be absolute or relative to `data_path`. By default,
the official reference is downloaded to `analysis_path/npvm_reference`, its ZIP
checksum checked, and its OMX extracted there. Shared input files and upstream
simulation caches are not edited.

## Single-map explorer

Results are written under the selected simulation's `compare_pt_to_npvm/`:

- `index.html`: one map with an NPVM-zone / municipality switch.
- `npvm_zone_od.npz`, `simulated_zone_od.npz`: complete sparse OD matrices
  in expanded PT person journeys per weekday, with float64 precision.
- `zone_index.csv`: matrix row/column order and original NPVM zone IDs.
- `zone_summary.csv`: zone origin, destination and intrazonal totals.
- `npvm_municipality_od.npz`, `simulated_municipality_od.npz`: municipality
  matrices aggregated from the zone matrices.
- `municipality_index.csv`, `municipality_summary.csv`: municipality matrix
  ordering, identifiers and aggregate totals.
- `car_npvm_zone_od.npz`, `car_simulated_zone_od.npz`,
  `car_npvm_municipality_od.npz`, `car_simulated_municipality_od.npz`: car
  matrices, in vehicle trips per weekday, sharing the corresponding index CSV.
- `car_zone_summary.csv`, `car_municipality_summary.csv`: car aggregate totals.
- `metadata.json`: shared scope and sampling settings, with separate PT/car
  definitions, units, reference components and simulation/exclusion audits
  under `modes.pt` and `modes.car`.

Previous `municipality_od.parquet` / `municipality_od.csv.gz` exports, if
present from the initial implementation, are historical outputs. The current
map uses the new sparse matrices above.

The map colours zones by `(simulated - NPVM) / NPVM * 100`. Blue means lower
simulated demand, red means higher. Independent lower and upper colour limits
clip extreme values. Grey indicates zero NPVM demand, for which the percentage
is undefined. Hovering always shows both models' originating and arriving
journey totals and percentage differences.

Click an area and choose **NPVM** or **Simulated** in its popup or sidebar. The
selected area is highlighted and the map draws the chosen source's ODs. Use
**Destinations** for outgoing journeys, or **Origins** for incoming journeys;
this also switches the aggregate colour basis. OD width represents the chosen
model's demand. OD tooltips and the table show both models for comparison.
Colours continue to represent aggregate zone differences, independently of
which source supplies the lines.

Line limits and thresholds affect the lines only; they never remove demand
from totals or colours. Intrazonal journeys can be excluded from totals and
OD tables, relative to the current zoning. Excluding same-area trips therefore
removes more journeys at municipality level. No line is drawn from an area to itself. Lines are schematic,
directed connections between points inside the zone polygons, not routed
transit paths. A search accepts zone IDs, displayed labels or municipality-name
substrings; municipality view accepts municipality IDs and names. Switching
levels clears the selected area and OD lines while preserving the viewport,
direction, colour limits and other controls. The Swiss overview button resets
the viewport.

The HTML embeds Leaflet, compact map shapes and already computed totals.
Display coordinates use five decimal places (about a metre); the analysis
geometries and counts are unchanged. **No full OD matrix is embedded or loaded
when the page opens.** Incoming and outgoing OD rows are prepared in Python and
stored in `assets/od/` as small files of 64 rows. Selecting an OD source loads
only the file containing the selected area and direction, for the active
mode and zoning. Already loaded blocks are reused. The browser looks up the
selected row and draws it; it does not aggregate demand or transpose matrices.

**Keep the `assets/` folder next to `index.html` when copying/downloading the
report.** Classic local script files support opening the report directly in a
browser without a web server. File previews that block scripts must be opened
in a normal browser. A failed OD-file load displays an error while the map and
aggregate statistics remain usable.

`assets/map_data.json.gz` stores the preprocessed geometry and totals for quick
HTML-only regeneration with `write_report`, without re-reading simulation trips
or NPVM source matrices. The sparse NPZ exports remain authoritative, with
float64 precision; display OD blocks use float32. Sparse data can be loaded with:

```python
import pandas as pd
from scipy.sparse import load_npz

zones = pd.read_csv("zone_index.csv")  # row/column positions match this order
npvm = load_npz("npvm_zone_od.npz")
simulated = load_npz("simulated_zone_od.npz")
```

The report opens directly as a local file in a modern browser supporting
`DecompressionStream`; no web server is needed. Only background tiles need
internet. Shapes, tables and statistics remain usable without tiles. The
swisstopo background has visible attribution and adjustable opacity. An
optional Esri global topographic background is available in the layer selector
for foreign regions beyond swisstopo coverage; representative tiles from both
providers were checked during implementation. The map and aggregate totals are ready before any OD file is requested.

The initial view uses municipalities, includes cross-border demand, and hides
foreign polygons. These controls remain independent and can be changed.
Jupyter HTML previews sandbox scripts; notebook-cell HTML also resolves relative
URLs against the notebook, rather than the report directory. If OD companion
files cannot load, open the downloaded report with its `assets/` directory in a
normal browser. Alternatively, expand **OD files / Jupyter preview** and select
the downloaded `assets/` folder. This reads the precomputed files locally, without
uploading data or executing file contents. Selecting a folder retries the current OD selection.
