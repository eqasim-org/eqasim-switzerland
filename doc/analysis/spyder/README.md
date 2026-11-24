# Interactive selected-link Spyder viewer

From the repository root:

```bash
uv run -m analysis.spyder.run config_dib_10.yml
# Or pass the exported scenario directory, without a pipeline config:
uv run -m analysis.spyder.run /path/to/output/rda
# The simulation output directory itself also works:
uv run -m analysis.spyder.run /path/to/output/rda/simulation_output
```

The positional argument is recognized automatically: `.yml` and `.yaml` paths
(case insensitive) are pipeline configs; other paths must be directories.
The launcher reads YAML without running Synpp. With a config, it locates the final
`*output_events.xml[.gz|.zst]` and `*output_network.xml[.gz|.zst]` in
`config.output_path / config.output_id / config.simulation_directory`
(default simulation directory: `simulation_output`). Ambiguous inputs require
explicit `--events` or `--network` paths. These overrides must identify final
simulation outputs that correspond to one another.

With a directory, the launcher looks for its `simulation_output` subdirectory,
or reads results directly if the directory contains `*output_events.xml*`.
An output root containing exactly one `<scenario>/simulation_output` is also
recognized. Multiple scenarios require selecting a scenario or specifying
`--simulation PATH`; the launcher never selects an arbitrary run. Detailed
geometry and population discovery still search beside the simulation network
and in the scenario directory above it. Original scenario/result files remain
read-only inputs.

Rust Cargo and a Rust compiler version 1.85 or newer must be on `PATH` for the
first build or when Rust sources/dependencies change. Unchanged reruns reuse the
compiled executable without invoking Cargo.
The first build downloads the dependencies listed in the checked-in Cargo lockfile;
subsequent builds use Cargo's cache. The Python launcher uses the existing project
PyYAML dependency. It calls a compiled Rust executable, so no Python extension
installation or changes to the project environment are needed.

Population attributes are read in Rust from `*output_persons.csv[.gz|.zst]` beside the
simulation events. If absent, the launcher tries `*output_plans.xml[.gz|.zst]`,
`*output_population.xml[.gz|.zst]`, then the scenario's `*population.xml[.gz|.zst]` in that
directory or its parent. `--population PATH` explicitly selects either a persons
CSV or a population XML, with plain, gzip, or Zstandard input. All formats are
streamed directly in Rust, including concatenated compressed frames. CSV delimiters can be semicolons or commas. Multiple
candidates at the same priority require an explicit path.

Processed `network.json`, `trips.json`, `manifest.json`, `index.html`, and the Rust
build are stored under `working_directory/analysis_spyder/<input-path-hash>/`.
In directory mode, the default is `./analysis_spyder/<input-path-hash>/` under
the current working directory. `--work-dir PATH` overrides either destination;
point it at an existing processed-data directory to reuse that cache when
switching between config and directory invocation. Input paths, sizes, modification
times (including the population source), and the processed-data schema version determine cache reuse. `--rebuild` forces
reprocessing; `--prepare-only` stops after preprocessing. An unchanged rerun prints
`Loading processed data; raw events and population will not be reread.`
It checks input file metadata without decompressing or parsing raw inputs, then
the Rust server loads `network.json` and `trips.json` and rebuilds its in-memory
query index. Changed input paths/sizes/times or the processed-data schema, missing processed
files, and `--rebuild` trigger preprocessing. Display/API, HTML, and launcher-only updates do not invalidate processed data;
compatible older manifests are accepted. Original simulation
outputs and upstream stage caches are read-only inputs.

`highways.json` and `highways-manifest.json` cache OSM highway types separately.
`hierarchy.json` and its manifest store the display rank for each link, derived
from the cached highway types. Upgrading to hierarchy display reads only this
metadata, without rescanning raw network XML or events.
New preparation writes the types along with the processed network. Existing
caches need a one-time Rust scan of the network XML to populate this sidecar;
that upgrade leaves `network.json` and `trips.json` unchanged and never reads the
events or population. Subsequent launches reuse the highway sidecar when the
network path, size, and modification time match.

After the cached-data server is listening, the launcher prints separate clickable
local/SSH and Euler JupyterHub URLs. Keep the process running and open that
URL; opening `index.html` as a local file cannot query the Rust backend. Stop
with Ctrl+C. The viewer uses a canvas and a default offline geographic background. It needs
no CDN or external tile service.

Directory mode has no YAML `data_path`. Pass `--data-path /path/to/pipeline`
to load the same offline country/canton/lake background; this option also
overrides `config.data_path` in config mode. Without spatial inputs, the road
network and spiders still work on the plain background. Online street tiles
remain optional and off by default.

## Cluster use and Lausanne example

```bash
uv run -m analysis.spyder.run config_dib_10.yml \
  --simulation /cluster/scratch/dabdelkader/lausanne/simulation_output \
  --work-dir /tmp/lausanne-spyder --threads 8 --port 8765
```

For Euler JupyterHub, run the command in a terminal in your Jupyter session,
on the same node as its Jupyter server. Click the printed Euler link, for example:

```text
https://jupyter.euler.hpc.ethz.ch/user/<username>/proxy/8765/
```

The launcher uses `JUPYTERHUB_SERVICE_PREFIX` when available, including named
sessions, and otherwise uses `JUPYTERHUB_USER` or your Unix username. API requests
are relative to the viewer URL so they stay inside Jupyter's proxy prefix.
A server launched on a different Slurm node needs SSH forwarding to that node.
See [Euler JupyterHub documentation](https://docs.hpc.ethz.ch/services/jupyterhub/)
and the [Jupyter Server Proxy URL format](https://jupyter-server-proxy.readthedocs.io/en/latest/arbitrary-ports-hosts.html).

On your own computer, forward the port to the host running the process:

```bash
ssh -L 8765:127.0.0.1:8765 <compute-host>
```

Then open `http://127.0.0.1:8765/` locally. Use your usual cluster jump host if
required. The server binds to loopback by default. `--host` and `--port` change
the listener address. It has no authentication; keep loopback binding for SSH use.

The thread count is the minimum of process CPU affinity, available Slurm CPU
limits, Linux cgroup CPU quotas, and `--threads` (or the YAML `config.threads`).
It does not rely on the total number of CPUs in the server. Compressed XML is
streamed in Rust in event order; parallel workers aggregate matching trip paths
when a link is selected. Compilation is limited to the same allocation.

## Geometry and controls

The launcher searches beside the network and in its parent directory for
`*detailed_network.csv[.gz|.zst]`. Use `--geometry PATH` to choose a specific file.
Supported columns are `LinkId,Geometry` or `link_id,geometry`, with LINESTRING
WKT in the same coordinate system as the network. Processed geometries retain the network coordinates (EPSG:2056/LV95). For display,
Rust transforms a copy into Web Mercator using the
[swisstopo approximation](https://www.swisstopo.admin.ch/dam/en/sd-web/KLRCX9XIdXDu/ch1903wgs84-EN.pdf).
The viewer expects Swiss LV95 network coordinates.

Only matching link IDs are used. Reversed shapes are oriented from the network's
from-node to its to-node. Shapes whose endpoints disagree with the network by
more than five coordinate units fall back to straight node-to-node lines.
Absent or malformed shapes also fall back; preprocessing reports match totals.

The browser starts with a geographic background and loads just the network in
its viewport. `background.json` is prepared once from the configured
`data_path/spatial/country`, `canton`, and `lakes` shapefiles, simplified for display,
and cached separately. It shows borders, cantons, lakes, and major-city labels.
The optional OpenStreetMap street-tiles control contacts
`https://tile.openstreetmap.org` from the browser and sends tile coordinates for
the visible area; it is off by default. Network layers remain usable if tiles
cannot load. See the [tile usage policy](https://operations.osmfoundation.org/policies/tiles/).

Click a road or search an exact link ID. When multiple directed links overlap,
a chooser lists every nearby candidate (up to 60), its ID and compass direction,
and a small directed shape. The blue dot marks the from-end and the arrow the
to-end. Hover previews the direction in yellow on the map; click the desired row
to trace that link. Exactly reversed geometries remain separately selectable.
Zoom in if an intersection returns too many candidates. Pan by dragging and zoom
with the mouse wheel. The selected link is dark yellow with an endpoint arrow.
Upstream links are blue and downstream links orange; all colors are editable.
Selecting a new target automatically unchecks `Show road network`: the map then
shows only spider geometries, with no duplicate base road underneath. You can
check it to restore context; active spider links are omitted from that base
layer. `Clear target · show network` removes the analysis and restores road
selection. Spider strokes are solid colored lines, without
per-link white outlines or white endpoint caps. Rounded colored joins keep
connected links visually continuous. In the combined view each directed link
is painted once, using the side with the larger count (upstream wins ties).
Small flows draw first and dominant flows draw last, avoiding opposite-color
flecks across heavily used overlapping roads. Both counts remain available in
the tooltip and the separate upstream/downstream views. Overlapping
directions are chosen through the overlap picker; all spider strokes follow the
original geometry with no artificial sideways displacement at any zoom. Spider
links also come first in the
overlap chooser.
The chooser shows each link ID and allowed modes. Controls include:

- Population checkboxes: Swiss resident, French external, cross-border, freight,
  airport, and unknown/unmatched. Choose any combination, such as French plus
  cross-border. `All` selects every group; `Clear` selects none. A trip is counted
  once even when multiple groups are selected.
- Network mode from `vehicle enters traffic` events. The default is car plus
  truck; individual modes and all recorded modes are also available.
- Time window in hours, referring to entry onto the selected link, start inclusive
  and end exclusive. MATSim hours beyond 24 are accepted.
- Upstream, downstream, or both; minimum trip count; maximum width and exponent.

For a displayed side with `n` trips and `N` selected trips, width is
`1 + maximum_width * (n / N) ** exponent` pixels. These are raw simulated vehicle
trip counts, without sample expansion. They are not passenger counts.

## Nested traces with right-click

Select a main target, then right-click a visible spider branch and choose
`Complete spider at <link>`. Overlapping directed links have separate choices.

The first filter selects trips crossing the main target within the population,
mode and entry-time filters. The second filter keeps trips also crossing the
second link, in either order. Trips that use only the second link are excluded.
The nested spider shows these trips' complete observed origin-to-destination
paths, split at the **second** link: its origin side uses the chosen purple and
its destination side uses a darker shade of that color. Widths use the nested
cohort's trip total. For loops, the first crossing of the second link defines the
split; the time filter remains anchored at the main target's first qualifying
crossing. A trip counts once per side even if it repeats links.

The main counts and group bars remain unchanged. The parent spider fades to
25% opacity while the nested path stays opaque. Its opacity slider ranges from
0 to 50%; setting it to zero hides the parent paths. Fading applies uniformly
to the composed parent layer. The sidebar identifies both links, the nested
trip count and its share of the main target. Hover reports parent and nested
counts separately; nested “to/from second link” labels use the second link as
the reference point. Link-traffic percentages retain the all-recorded-times
denominator for the same population and mode filters.

One second link is shown at a time; choosing another replaces it.
Changing the main target or population/time/mode filters clears the nested trace.
Panning, zooming, and switching the displayed road classes keep it intact.

## Hover counts and population bars

After selecting a target, hover a visible spider stroke. The tooltip shows its
link ID and direction, trips going **to** the target (upstream), trips coming
**from** the target (downstream), and two percentages for each side:

- **Of selected target:** side count divided by the target trip count for the
  checked population groups, vehicle modes, and target-entry time window.
- **Of link traffic:** side count divided by all trips using the hovered link,
  restricted to the same population groups and vehicle modes, over **all recorded
  times**. This denominator deliberately includes trips whose entry to the
  hovered link is outside the target's time window, so every matching trip is
  included and percentages cannot exceed 100% merely because of travel time.

The tooltip labels the time scope explicitly. A trip counts once per side,
including on loops; one looping trip can appear on both sides. The target itself
shows its selected trip count. In the combined display, the dominant side
controls its color without changing either count. Hovering overlapping spider strokes can show up
to four directed links, with counts computed from the Rust trip index and cached
for subsequent hovers. Background-only roads do not obscure spider tooltips.

The sidebar bars describe the selected target, using the current time and
vehicle-mode filters. `Total · all groups` is the full 100% bar; each population
row reports its trip count and share of that total. Rows remain visible when
unchecked, with a muted color. `Selected groups` shows the sum for the checked
groups and matches the large target count. The population groups are disjoint,
so their target counts sum to the total.

The unchecked-by-default `Show all car/truck links` control keeps the base network
and both spider layers restricted to these `osm:way:highway` classes:
`motorway`, `trunk`, `primary`, `secondary`, `tertiary`, and all five corresponding
`_link` ramp classes. Residential, unclassified, service, other minor classes,
and missing highway types are hidden. Checking the control includes every
car/truck road, including those classes. Search respects the car/truck and major/all-road filter. Direction picking and
both spider layers retain all road classes permitted by the major/all-road
checkbox at every zoom. The zoom hierarchy affects only background roads.
Target/via markers follow the road-class checkbox and remain visible when zooming
out; their selections and counts remain in the sidebar if hidden by road class.

Only links with the exact allowed-mode token `car` **or** `truck` appear in the
road viewport, search, or direction chooser. Truck-only links are retained;
PT-only and bike-only links are hidden. A road that additionally allows buses
is still a road. This display filter does not remove anything from the cached
network or paths. The road-class display filter does not change vehicle counts
or the selected cohorts; it changes which path geometries are drawn. Choosing
all recorded vehicle modes can include bus trips on
these road links; the default car-plus-truck selection excludes them.

## Population classification

Groups describe the driver associated with each `vehicle enters traffic`
event's `person` attribute. Vehicle IDs are not treated as person IDs: a shared
vehicle can have a different driver on its next trip. Missing drivers or IDs
absent from the population remain unknown. This is a vehicle-trip analysis;
PT passengers are not assigned the bus or train driver's population group.

Classification uses the exported model attributes, with this precedence:

| Group | Rule |
| --- | --- |
| Freight | `isFreight=true` or `subpopulation=freight` |
| Airport | `subpopulation=airport` |
| Cross-border | `isCrossBorder=true` or `subpopulation=crossborder` |
| French external | `isExternalFR=true` |
| Swiss resident population | Ordinary/normal population with a Swiss canton or both external/cross-border flags explicitly false |
| Unknown/unmatched | No sufficient attributes, no driver, or driver not found |

These are model population groups, not citizenship labels. French external and
cross-border are distinct sources; cross-border takes precedence if both flags
are true. The pipeline's lower-case canton sentinel `fr` means France, while
`FR` means Fribourg. No group is guessed from an ID prefix, age, or nationality.
XML reading retains only person-level attributes and ignores similarly named
activity/leg attributes. A missing population source leaves all trips unknown;
it does not prevent viewing the unfiltered network. The sidebar target bars include unknown trips to make matching coverage visible.

## Counting semantics and scale

A vehicle trip begins at `vehicle enters traffic` and ends at
`vehicle leaves traffic`. This keeps multiple daily trips by one vehicle
separate. The initial traffic-entry link and later `entered link` events form
its directed path. Identical consecutive link/time entries are deduplicated.
If traffic-entry events are missing, paths are labeled `unknown` until a traffic
boundary. Trips still active at EOF are retained as observed partial paths.

A matching trip crosses the selected link inside the time/mode filter. For loops,
the first crossing inside the window anchors the upstream and downstream paths.
Each other link counts at most once per trip on each side, and the selected link
is excluded from both sides. A link can be both upstream and downstream on a
loop. These are full observed trip paths through the link, rather than only
immediately adjacent turns or straight origin/destination connectors.

The Rust server builds an inverted link-to-trip index at startup and scans only
matching candidates, using Rayon within its allocation-limited worker pool.
HTTP requests use a bounded worker pool (up to two handlers), so a large geometry
response does not serialize all hover, selection, and metadata requests. On a
four-CPU allocation, two threads are reserved for HTTP work and two for shared
Rayon computation. One-CPU runs remain serial in active computation. Geometry
JSON is serialized directly from borrowed arrays and is gzip-compressed when
the browser accepts gzip. Separate spatial grids and link lists
for each highway rank keep minor roads out of wide-view searches. Display
geometry preparation runs in parallel. Spider-only requests transfer only links
in the active parent/nested cohorts.

Roads are revealed by complete highway classes, including each class's `_link`
ramps, rather than sampling individual links:

| Web Mercator zoom | Classes visible through |
| --- | --- |
| Below 9 | Motorway |
| 9 to below 10 | Trunk |
| 10 to below 11 | Primary |
| 11 to below 12 | Secondary |
| 12 and above | Tertiary |
| 13 and above, with `Show all car/truck links` checked | All car/truck classes, including minor and unknown types |

This hierarchy applies only to the background network. Main and nested spiders
keep every permitted link in the current viewport at every zoom, with their full
detailed geometries. They bypass background rank cutoffs, density coarsening,
and geometric simplification. The major-road / all-car-and-truck checkbox still
applies to both traces. Enabling the background network returns its coarse roads
alongside the complete spider, and direction picking can select lower-rank spider
links even in a motorway-level overview. Counts and observed trip paths are not
changed by zoom.

If the background alone would return more than 40,000 links, the server lowers
its cutoff by whole classes and the badge requests a closer zoom. Spider links
are excluded from this density decision and are always retained. It never drops
arbitrary background links within a visible class. If even the highest available
background class exceeds that budget, that complete class is retained. Real gaps
already present in the simulation network remain gaps; no connecting links are
invented.

`Background opacity` controls the offline geographic backdrop and optional
street tiles independently of the flow colors. The browser keeps separate
cached bitmap layers for the background, network, parent spider, and nested
spider. Up to two Web Workers build cached `Path2D` geometry, rasterize the
network/spider layers, and perform hit testing outside the UI thread. A browser
without the required worker/canvas APIs uses the compatibility renderer. Incoming
and outgoing paths stay in the same render job to preserve their width/color
ordering. Coordinates use a local origin per link, avoiding numerical shifts at
high zoom. No path simplification or offsets are introduced for spiders.

Geometry fetching, JSON parsing, and bounding-box preparation also run in a
worker. Geometry is shared directly between the workers through a message
channel, avoiding large geometry copies during UI drawing. A padded viewport
cache reuses data across small pans, zooms, and network/spider switches. After
target selection, a background request warms the full spider geometry, including
minor roads for the optional all-roads setting. Nested traces reuse that parent
geometry. Cache keys include the parent overlay, filters, extent and detail;
data from another target or population selection is not reused accidentally.

The geometry cache is limited to eight entries and approximately 64 MiB. Worker
path caches target 48 MiB, retaining two active/warm working sets when needed;
bitmap caches target 64 MiB. A single large full spider can exceed these targets
because it is retained whole. Cache size estimates do not include all browser
engine overhead. The server uses a bounded request queue and immutable overlay
snapshots rather than holding a global lock during rendering or network writes.

During a gesture, the UI immediately reprojects the last bitmap and requests an
updated worker frame. Each rendering role keeps one active request and only the
latest pending camera view. Superseded requests are cancelled or skipped, rather
than queued behind obsolete zoom positions. Background geometry is refreshed
after movement settles. Hover and opacity controls reuse ready bitmaps. Parent
opacity is composited once for a uniform fade; changing a nested trace retains
the ready parent image while its replacement is prepared.

Local Lausanne measurements (headless Chromium, four allocated CPUs) reduced
UI-thread spider repaint work from about 71 ms to roughly 4–6 ms. A cold
network-plus-spider repaint measured approximately 0.3–4 ms of UI-thread work after
worker geometry preparation, compared with about 76 ms previously. Warm
network/spider switches measured approximately 6–15 ms with no new geometry
requests. Heavy
rasterization still takes time in the workers; these measurements describe UI
responsiveness, not an instantaneous end-to-end render guarantee.

Network and trip paths remain in memory; very large
national runs require memory proportional to all observed traversals. JSON
caches favor portability and inspection over minimum disk size. Viewport loading bounds browser transfer/rendering costs; regional exports
remain useful for reducing server memory requirements.

Focused validation on the Lausanne export processed 22,458 network links and
5,709,832 relevant events into 94,534 vehicle trips in approximately 15 seconds,
with 119 MiB peak preprocessing RAM in the development allocation. Performance
varies with storage and CPU availability.

Detailed geometry endpoints within five metres of their MATSim nodes are snapped
to those exact nodes, preserving intermediate curve vertices. This closes small
shape/node gaps without joining unrelated roads. Existing caches receive a
one-time network-only alignment on launch; processed events and populations are
reused. The alignment is recorded and subsequent launches skip it. Hidden minor
road classes or a minimum-flow threshold can still leave visible gaps: enable
`Show all car/truck links` and use minimum flow 1 to inspect those connectors.
Actual missing or disconnected event/network paths are not bridged artificially.
