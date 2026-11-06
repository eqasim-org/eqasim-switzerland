# PT route corrections

`matsim.scenario.network.corrected` compares mapped bus routes with the GTFS
shape of the trip, falling back to an official line reference for its operator.
Routes sharing a line number in Geneva and Lausanne are kept separate.
`analysis.pt.transit_schedule.visualize` uses the same references and can inspect
the mapped, corrected or simulated schedule.

The default operators are TPG (`881`) and TL (`151`). `pt_route_gtfs_shapes`
defaults to `true`; `false` uses only the official line references.
`pt_line_routes_path` accepts the existing single path or an agency-to-path
mapping. The default mapping uses `PT_line_routes/Geneva/TPG_LIGNES.shp` and
`PT_line_routes/Lausanne/Shape_lignes.shp`, relative to `data_path`. Missing
reference files are reported; routes with usable GTFS shapes can still be checked.
GTFS files come from the cleaned stage's `output/` directory.

Corrections consider stop order, U-turns, repeated links, stop access and deviation
from the shape. Existing roads may gain bus permission in the corrected network.
When `pt_route_add_links` is enabled (existing default: `true`), missing bus
connections can be added along the reference. These links are bus-only under the
default `pt_route_modes: [bus]`; real-road capacities are unchanged. Set
`pt_route_add_links: false` to restrict correction to existing connections and
inspect unresolved routes before accepting new links.

Official line references do not always serve the GTFS quays: TL's
`Shape_lignes.shp` can follow another street than the quay of the national GTFS
(e.g. line 17 at Port-Franc). `pt_route_platforms` gives, per operator, its stops
and the table of their DIDOK numbers; by default TL's
`PT_line_routes/Lausanne/Shape_Arrets.shp` and
`PT_line_routes/Lausanne/extrait_infos_arret_ATT5_2026.csv`. A DIDOK number is the
GTFS station id without the Swiss code `85`. When a stop of a route compared with
an official line reference lies outside its corridor, the correction can move it
to a platform of the same station inside the corridor, at most 300 m away; the
corrected schedule then has a stop such as `8592098:platform:PFRAN_E`, with the
platform's coordinates. A route with a stop more than 150 m from its reference and
no such platform takes another path than the reference (e.g. a newer route) and
keeps its mapping (`other_path` in the report). Routes compared with a GTFS shape,
such as TPG's, are not affected.

Some official line references skip stops of the GTFS: TL's line 32 reference is
a later route via Renens VD, 1er Août, without Renens VD, Avenir, so its routes
would keep their mapping (`other_path`). `pt_route_through_stops_lines` (default
none; e.g. `{"151": ["32"]}`) lists such lines by GTFS agency. For their routes, the
reference goes in straight lines through the stops more than 60 m from it,
between the stops before and after them on the reference, and the routes are
corrected against it as usual. The leg that has no road (line 32: the Rue du
Simplon underpass at Renens gare, which OSM tags `motor_vehicle=permit` without
bus access) then gets one straight bus-only link when `pt_route_add_links` is
enabled.

The stage writes a local schedule/network copy and `route_correction.csv`, with
before/after route metrics and added-link counts. Existing detailed curves are
preserved, reversed links receive reversed curves, and new links receive directed
geometry. The following `matsim.scenario.network.pt` stage must consume the
returned network, schedule and geometry together. Upstream mapped assets are
read-only.

Local validation covers operator identity, shape/stop order, rerouting a detour,
bus-only added links, capacity preservation, reversed geometry and the GTFS stage
path contract. National validation still requires rebuilding the corrected/PT
network and inspecting its report. This change does not alter PT utility
coefficients or cross-border donor timing restrictions.
