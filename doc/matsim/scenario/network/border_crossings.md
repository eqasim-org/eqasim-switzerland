# Swiss-border link tagging and freight crossing GeoParquet

`matsim.scenario.network.convert_osm` runs border detection after network cleaning,
mode/connectivity correction and final attribute adjustments. It tags crossing
links in their copied attribute dictionaries with `ch_border_crossing: true` and
writes `border_crossing_links.parquet` in its Synpp cache directory. Its Python result
remains the completed network XML path. The tag is preserved in saved XML/pickle.

## Geometry and crossing definition

`get_network_geometry` in `utils/link_geometry.py` returns an EPSG:2056 GeoSeries
aligned with the input links' row order and index. It preserves directed detailed
curves, uses endpoint-node lines only for missing shapes, handles empty subsets,
and reprojects the network's declared CRS. Invalid supplied shapes fail explicitly.
Both toll exclusion and border detection consume this helper. Toll exclusion
continues to use the full geometry against the existing 10 m Swiss buffer.

The border detector uses exact line/boundary intersections and evaluates country
membership along the directed line on both sides. Pure tangencies and roads wholly
following the boundary are not crossings. A curved link with two foreign endpoints
can enter and leave Switzerland, producing two GeoParquet rows. Links split exactly at a
boundary node are included when incident link interiors lie on both sides. More
than one directed link may therefore identify the same physical road crossing.

The GeoParquet contains final link/node IDs, crossing index, highway, modes and tag.
Its `direction` is `in` (foreign → Switzerland) or `out` (Switzerland → foreign),
computed from country membership along each directed link. `border_x/border_y`
and `border_geometry` retain the exact intersection. `crossing_x/crossing_y` and
the active `geometry` contain the routing point shifted exactly one metre right
of the local from-node → to-node segment; `right_offset_m` records `1.0`.
At a curve vertex the forward segment defines right; at the line end the last
segment is used. Repeated vertices are ignored. Opposite overlapping link shapes
therefore receive points on opposite sides. The `link_geometry` column retains
the full directed curve. All three geometry columns declare EPSG:2056 and use
Zstandard-compressed GeoParquet.

Re-running tagging removes stale inherited border tags from non-crossing links.

## NPVM consumption

See [NPVM freight generation](../../../synthesis/freight/npvm/README.md).
`npvm_freight_use_border_crossings` defaults to `true`. Road hierarchy and all
freight eligibility/allocation rules live in `synthesis.freight.npvm.crossings`.
The network export preserves the original highway value and every link mode;
it does not filter or rank them. String mode lists are retained, and collection
mode values are stored as sorted comma-separated text.

For `npvm_freight_crossing_source: network`, the freight consumer chooses
truck-permitted roads if truck occurs in the export,
otherwise car-permitted roads, and excludes its lowest road category. It samples
separately for `in`/`out` among the best two adjacent classes within a zone, or the
two nearest distinct crossing points outside zones without crossings. Weights
per point are motorway/trunk 8, primary 4, secondary 2, tertiary 1. For nearest-two
fallbacks these are divided by `1 + distance to zone in km` before normalization. Full selection,
deduplication and probability details are in the NPVM freight guide above.

For that direct network source, the crossing-reader stage depends on conversion
and reads its GeoParquet. The default `route` source instead reads saved
`freight_routed_crossings.parquet` beside the NPVM matrices, without loading or
routing the network. With `npvm_freight_routing_overwrite: true`, it rebuilds those
OD-specific assignments from conversion's `network.pkl` and matching detailed curves. An explicit `npvm_freight_border_crossings_file` (relative to `data_path`
or absolute) removes that network dependency. Set
`npvm_freight_use_border_crossings: false` to preserve geometric placement instead.
Both default and disabled full MATSim dependency graphs are checked for cycles.
Older network crossing exports must be regenerated for the directional/offset schema.
An export missing directions or either truck-permitted direction fails explicitly.
Read-only validation exports made from existing outputs are kept outside upstream
cache directories.

## Routing scope

Foreign freight origins use the selected `in` point; foreign destinations use the
selected `out` point. Transit uses both. The one-metre offset separates coordinates
for opposite directed shapes; it does not bind an activity to the saved link ID.
The inspected cached Java `FreightAssignment` assigns nearby facilities by link
coordinate and replaces freight activity coordinates. Correct final link binding
therefore cannot be guaranteed from lateral coordinate offsets alone.

The optional `npvm_freight_crossing_source: interview` source is adapted in freight
and leaves this general network export unchanged. It uses scored interview road
references through `data.cross_border.road_candidates` and its entry/exit matcher.
See the NPVM guide above for source selection, score weighting and bidirectional
fallback behavior.
