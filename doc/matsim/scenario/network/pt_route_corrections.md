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

A route deviates when it drives more than 50 m off its reference: off the lane of
a GTFS shape (links more than 8 m away), or off the road of an official line
reference (more than 20 m away; these lines follow the middle of the road and are
drawn less precisely). The share inside the 30 m corridor alone misses a short
detour on a parallel street of a long route, e.g. TL line 2 at Georgette, line 6
between Pont Marc Dufour and Languedoc, line 7 at Prilly-Malley, gare/aréna or line
9 via Aubépines.

A route with more than 50 m of pt2matsim artificial links also deviates. These are
straight links where pt2matsim found no road, e.g. TL line 32 under the tracks at
Renens VD, gare. For routes compared with an official line reference, an
intermediate stop is not served from a link more than 10 m away when a link in the
direction of travel is at least 10 m closer. This is usually the road behind the
platform, e.g. TL lines 46 and 64 at Epalinges, Croisettes. The correction then
uses the closer link, also for the TL platforms. GTFS shapes already give the lane,
and the rule does not apply at the termini.

Stations are recognised for both GTFS stop id formats: `8592098:0:10000` and the
Swiss SLOID `ch:1:sloid:92098:0:1164` (newer feeds), both station `8592098`.

A stop with no link within 30 m (e.g. a terminus loop missing in OSM, such as TL
Pully, Clergère or Lausanne, Grand Vennes) is reached by bus-only links along the
GTFS shape. Without a shape (TL), the part of the official line reference nearest
the stop is used, oriented in the direction of travel. When two routes share such
a terminus, one starting and one ending there, each gets the access link it needs.
The results then no longer depend on which route is corrected first (e.g. TPG line
54 at Versoix, Centre sportif). Stops are snapped to references densified to one
vertex per 10 m. A GTFS shape with sparse vertices (e.g. TPG line 83/84 at Viry,
P+R) can then receive links along it where the network lacks the road.

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
