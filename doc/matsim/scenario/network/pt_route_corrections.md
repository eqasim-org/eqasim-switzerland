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
