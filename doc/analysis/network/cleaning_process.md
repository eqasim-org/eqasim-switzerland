# Network cleaning inspector

The Synpp stage `analysis.network.cleaning_process` compares the outputs of
`matsim.scenario.network.convert_osm_pt2matsim` (raw conversion) and
`matsim.scenario.network.convert_osm` (processed/cleaned conversion). Each network
uses its own stage's `detailed_network.csv`. It does not compare the later PT mapper
or simulation network. When available, the cleaner's audit records the operation
responsible for each removal.

## Build and open

For the usual workflow, one command prepares the latest cached stage data and
starts the server:

```bash
uv run -m analysis.network.run config_dib_10.yml
```

The config argument defaults to `config_dib_10.yml`. Keep the terminal running and
open the printed local or Euler Jupyter URL. Start it in a Jupyter terminal on
Euler so that its proxy can reach the server. Use `--port 8766` for another port.
Stop the server with Ctrl+C before restarting it after code or data changes.

The launcher reads `pipeline.json` from the configured `working_directory` and
selects the newest **completed** `convert_osm` run there, paired with the exact raw
`convert_osm_pt2matsim` dependency recorded for that run. It does not execute Synpp
or rerun any upstream stages. This is latest-completed selection within the working
directory, not a request to regenerate the scenario described by the config.
The terminal and manifest identify the selected producer stages.

Viewer data live under `<working_directory>/analysis.network.run.cache/<fingerprint>/`.
Unchanged XML and detailed geometry paths/sizes/modification times reuse the
same bundle. Changed network inputs select a separate dataset cache; older bundles remain
available. Inside a bundle, importer changes rebuild the network database,
audit or preprocessing changes rebuild the drawing index, and HTML changes refresh the page
template without rebuilding data. No audit means no red removal lines. These caches
are analysis-owned; the producer cache files are read-only inputs.

Use `--show-sources` to preview the selected inputs without building anything, or
`--build-only` to prepare the cache without starting a server. The first national
build can take considerable time; subsequent launches reuse the prepared data.
The launcher reserves the server port before starting that build, so a busy port
fails immediately. It also prevents concurrent viewers/builds from replacing the
same dataset's files while they are being served.

The separate Synpp build/export entry point is also available:

From the repository root:

```bash
uv run -m synpp config_dib_10.yml --run analysis.network.cleaning_process
```

Synpp resolves the two producer stages normally; check their cache validity before
running if you need to avoid rebuilding upstream stages. The analysis writes an
independent bundle to `<analysis_path>/network/cleaning_process/` and returns a
dictionary containing `html`, `database`, and `directory`. Producer caches are read-only.

Start the viewer from the repository environment:

```bash
uv run -m analysis.network.cleaning_server /path/to/analysis/network/cleaning_process --port 8765
```

Open `http://127.0.0.1:8765/`. On the cluster, forward that port or use your Jupyter
server's `/proxy/8765/` address. The server defaults to localhost. Opening `index.html`
directly explains how to start the server: national data are deliberately not embedded
in one enormous HTML file. Keep the SQLite file, manifest, HTML, and assets together.
The viewer server does not invoke Synpp, route trips, clean networks, or simulate.
Restart the viewer server after rebuilding the bundle so it reloads the manifest.

To build directly from explicit existing files, without invoking any pipeline stage:

```bash
uv run -m analysis.network.cleaning_process \
  --before /path/to/raw/converted_network.xml.gz \
  --before-geometry /path/to/raw/detailed_network.csv \
  --after /path/to/processed/converted_network.xml.gz \
  --after-geometry /path/to/processed/detailed_network.csv \
  --output /path/to/analysis/network/cleaning_process
```

## Controls and interpretation

- Default road checkboxes select motorway, trunk and primary roads and all their
  `_link` categories. Additional categories are discovered from the actual files.
- **Compare** draws cleaned links plus raw links audited as disconnected for car.
  Loops, duplicates and the outside-region filter are excluded from red lines.
  **Before** and **After**
  show the corresponding source. **Original segments of merged links** adds dashed
  raw segments to the comparison.
- **Retained**: matched by original link ID or single-member provenance.
  Attributes or modes can still have changed; inspect both versions.
- **Merged**: at least two original link IDs occur in a cleaned link's `old_link_id`.
  This includes the surviving first segment's ID. All mapped original segments have
  the merged status, so they are not incorrectly classified as removed.
- **Removed**: no cleaned ID or recorded `old_link_id` matches. Classification is
  based on available provenance, not similar OSM IDs or nearby geometry. Missing
  provenance in an upstream export cannot be reconstructed reliably here.
- The cleaner writes `cleaning_events.jsonl.gz` and includes its per-operation
  counts in `statistics_of_cleaning_network.json`. Every event records its
  operation link ID, original link IDs, reason and action. A merged chain removed
  later retains the removal reason for **every** original segment. Truck-only
  connectivity changes have action `mode_removed`; those links are retained.
  Missing audits hide removed links rather than guessing the cause.
- **Circle disconnected removals** draws circles enclosing the visible red links.
  Circle positions and sizes are prepared once; panning loads normal cached tiles.
- Overlapping reciprocal links use display curves offset **2 m to their own right**.
  Separated carriageways are left alone. Offsets are precomputed in LV95 and used
  for drawing, click selection and selection highlights. Simulation geometry and
  attributes remain unchanged; turn arrows follow the original network geometry.
- **Added**: a cleaned link has no matching original link provenance.
- Click anywhere near a line to list overlapping directed links, inspect XML fields
  and every custom attribute, and jump to before/after counterparts. Search accepts
  exact MATSim link IDs or OSM way IDs, including `old_osm_id` provenance.
- Turn arrows use the selected link's network and mode, independently of road display
  filters. Green is a permitted next movement in the stored data, red is an explicit
  single-next-link ban, and amber is a history-dependent multi-link prohibition.
  Entering the first link of a longer prohibition is not itself forbidden. Missing,
  discontinuous, or mode-incompatible references are reported, not invented.
  Arrows assume no route history before the selected link; restrictions that start
  on an earlier approach must be inspected by selecting that earlier link.
  Arrows reflect the exported restrictions, including any converter mistakes.

## Performance and basemap

XML and geometry CSVs are streamed into SQLite in batches; full national data never
become a single Python/GeoJSON/browser object. An RTree indexes complete shape bounds,
including curved links crossing the viewport whose endpoints lie outside. Link IDs,
node adjacency, provenance and OSM way IDs have separate indexes. Encoded polylines
are precomputed at 1.5 m and 40 m simplification, leaving full shapes and attributes
in SQLite for accurate inspection.

The browser loads transparent network tiles as the viewport changes. Each tile
streams **all** matching links from the spatial index: there is no feature-count
cutoff. Server rendering keeps national views out of the browser's vector scene;
tiles are cached in bounded memory and reused while panning. Full detailed geometry
is used at zoom 16 and above and for selected links; smaller scales use precomputed
simplifications. Click selection and turn arrows remain separate indexed requests.
The first uncached national view can take longer to render.
Bundle generation also builds `drawing.sqlite`, a compact derived drawing index,
beside the original database. For older bundles, the first server start builds it.
Subsequent starts reuse it. Large attribute records
are accessed only for inspection, not for every overview tile.
The index contains full offset display shapes, an RTree and precomputed circle
coordinates. Changes to source data, the cleaning audit or preprocessing code
invalidate this derived index. `cleaning_audit.sqlite` stores the original-ID
removal reasons; the compressed event log retains all operations, including merges.

The server prints clickable local/SSH-tunnel and Euler Jupyter proxy URLs. The
Euler proxy URL requires this server to run on the same node as the Jupyter server.

To rerun topology cleaning against already resolved upstream caches, use
`python -m scripts.replay_network_cleaning --config config_dib_10.yml
--stage-key <convert_osm cache key> --output <audit-run-directory> --viewer <bundle>`.
This replays geographic filtering, loop/duplicate removal, chain merging, and
restriction-aware car/truck connectivity. Toll boundaries and protected PT-stop
nodes use their resolved inputs. It verifies cleaned link IDs and merge membership
against the existing inspector; a mismatch prevents audit publication. It does not
run population generation, later speed/capacity corrections or a simulation.
Then `python -m analysis.network.refresh_cleaning <bundle> --run <audit-run-directory>`
imports the verified reasons and rebuilds only the display index. Restart the viewer
and refresh the page afterward. Upstream caches remain read-only.

OpenStreetMap is enabled by default and adapts to pan and zoom, with adjustable
background opacity and visible attribution. Tiles pass through the local server,
so the browser does not need direct access to the tile provider. Successfully
downloaded tiles are cached in `.tile_cache/osm/` for seven days; stale copies are
used if a download fails. Initial loading requires the server to reach OSM; no
permanent external availability is assumed. Failure leaves network inspection
usable. Leaflet libraries are bundled locally.

Optional config keys (defaults):

```yaml
analysis.network.cleaning_process.center: [46.252196, 6.138081]
analysis.network.cleaning_process.zoom: 13
analysis.network.cleaning_process.offline_radius_m: 8000  # legacy; no fixed snapshot now
analysis.network.cleaning_process.max_features: 10000  # legacy vector API only; tiles are uncapped
```

The geometry CRS is LV95 / EPSG:2056, following the converter's output contract.
Invalid/missing detailed shapes use directed endpoint fallback, counted in the manifest.
Database builds are atomic and are reused when source paths, sizes, modification
times and the importer code match. An interrupted database is preserved separately on retry.

## Comparing pt2matsim revisions

Set `pt2matsim_commit` to a full 40-character commit SHA to pin the runtime build.
The runtime checks out that exact commit and records it in `pt2matsim_commit.txt`.
Changing the pin gives the runtime and dependent stages distinct cache identities;
a cached branch name alone does not fetch newer commits automatically.

`scripts/run_network_revision.sh` runs the runtime and both network conversion
stages through Synpp using the named configuration. Required PT schedule dependencies
also use the rebuilt runtime. Capture the baseline first using
`python -m scripts.compare_network_revision --capture-baseline --baseline <report>/baseline.json`.
Then submit `sbatch scripts/run_network_revision.sh config_dib_10.yml <report>` from
the repository root. The job writes `<report>/comparison.json` with actual commit
SHAs, network sizes, cleaning statistics and the two investigated A1 OSM ways before
and after cleaning. This does not refresh an existing viewer bundle automatically.
