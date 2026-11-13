# Zones of the regional PT subscriptions

`synthesis.population.models.subscription_zones` runs after
`synthesis.population.models.subscriptions`. That stage says who has a regional
subscription (Verbundabo: `pt_subscription` 2, or 4 with the half-fare card), but
not which zones it covers. This stage finds them from the commute, with the method
of the Swiss PT explorer (`verbund_zones.py` of
[swiss-pt-explorer](https://github.com/AuroreSallard/swiss-pt-explorer)). The code
it needs is copied into `utils/gtfs_router/`: the timetable reader (`feed`), the
router (`csa`), the fares and stop zones (`pricing`), the zones of the stops abroad
and the ride tracks (`zones`, `tracks`), the Léman Pass (`lemanpass`) and the method itself
(`verbund`). Keep these copies in step with the Swiss PT explorer.

## Method

1. Each agent with a Verbundabo gets a commute destination: the place of education
   for students, else the workplace. The others have none.
2. The home and destination are placed in the zone polygons of
   `data.pt_pricing.zones` (built from `tarifverbundkarte.geojson`, the zones of the
   stops in `gtfs_zones.csv` and the French communes of Unireso). The ride tracks are
   cut with the same polygons, so the stage uses one set of zones throughout.
   The municipality outlines leave out the lakes and are coarse at the borders, so
   a home or destination in no zone but within 100 m of one (`SNAP_M` in
   `utils/gtfs_router/verbund.py`) is placed in the nearest zone, with the Z-Passes
   that share it. In the 5% population this concerns 86 Verbundabo homes, mostly on
   the shores of the Zürichsee; every home farther from a zone is in a municipality
   without a fare network.
3. The agents are grouped by zone OD. The subscription covers:
   - the home zone, without a destination or with one in the same zone;
   - the zones of the home network along the commute, within one network;
   - the Z-Pass zones along the commute, between two networks joined by a Z-Pass;
   - in these two cases, always the home and destination zones too (a journey can
     start or end with a walk to or from a stop of a neighbouring zone);
   - the zones of the home network until the first ride that leaves it, between
     two networks;
   - the zones of the destination network back from the destination, for a home
     in no network.
4. The commute of every agent whose home and destination are in different zones
   is routed on the timetable of one day, leaving at `subscription_zones_departure`,
   with the route-choice weights of the MATSim PT router
   (`calibration.pt_routing.pt_routing_parameters`, calibrated when
   `calibrate_pt_routing_params` is true),
   and the agent gets the zones of its own journey. An agent without a journey gets
   those of another agent of its zone OD, drawn with `random_seed`; without any
   journey in the OD, the home and destination zones are used. With
   `subscription_zones_samples` > 0, only that many commutes are routed per zone OD
   and the other agents copy one of them: in the 5% test population (2026-11-10),
   75% of the copying agents got the exact zones of their own journey and 16% missed
   at least one zone (39% for Z-Pass commutes of five or more zones).
5. A commute into the Léman Pass area abroad gets the cheapest annual Léman Pass
   product that covers it, when the tpg fare table (`tpg_<date>_Fare-system_Geneva.xlsx`)
   is in `subscription_zones_lemanpass_path`. A product of zones only (`Zone 10 + Zone 230`)
   covers every journey inside its zones, trains included: Genève - Lancy-Bachet by
   train and the bus to Saint-Julien gets `Zone 10 + Zone 230`. This rule is in the
   eqasim copy of `lemanpass.py`, not yet in the Swiss PT explorer.

## Inputs and options

| Option | Default | Meaning |
| --- | --- | --- |
| `subscription_zones_date` | empty | Routing date (YYYYMMDD); empty: `gtfs_date` if it is a date, else the first Tuesday of the feed |
| `subscription_zones_departure` | `07:30` | Departure time of the routed commutes |
| `subscription_zones_samples` | 0 | Commutes routed per zone OD; 0: every agent |
| `subscription_zones_lemanpass_path` | `pt_pricing/output` | Folder of the tpg fare table, relative to `data_path` |
| `subscription_zones_bus_tracks` | `true` | Buses follow their routes in `matsim.scenario.network.pt`; `false`: straight lines, without that dependency |
| `pt_zones_lines_file` | `pt_pricing/linie-mit-polygon.geojson` | SBB rail lines, for the train tracks |
| `subscription_zones_map_agents` | 400 | Agents drawn on the control map, spread over the cases; 0: no map |

The zone polygons use the `pt_zones_*` files of `data.pt_pricing.zones`. The
timetable is the output of `data.gtfs.cleaned`, and the fares those of
`data.pt_pricing.pt_pricing`. Routing uses `threads` processes.

The Léman Pass fares are read directly from the tpg fare table, as in the Swiss PT
explorer (sheets 1 and 3, the latest table if there are several). Reading it takes
about 10 s, so the result is cached in the stage folder (`lemanpass_<hash>.pkl`) and
read again only when the file changes. Without the table, the `lemanpass.csv` and
`lemanpass_places.csv` written by older versions are read if they are in the folder.

The stage also attaches the Léman Pass to the pricing (`pricing.lemanpass`), so
`Pricing.price` can use a Léman Pass single ticket. It covers the train part of a
journey into the area abroad and the bus and tram rides around it, and is kept when
it is cheaper than the usual tickets. A usual Unireso ticket with a zone abroad is
kept only if no Léman Pass ticket fits: Unireso itself sells only Zone 10. Single tickets have adult, child (6-15), youth
(16-25) and half-fare prices. The half-fare card applies only to tickets with Swiss
fares: SBB, a zone network with a stop in Switzerland, or a distance-based ticket with
no stop abroad. Under 16, half fare applies everywhere. Town zones sold only with their
station (Zone 300, Thonon) no longer include the other stations inside them.

In the current synthetic population, the work and education places of the Swiss
residents are in Switzerland, also for those who cross the border: their trips
abroad are other activities. Their commutes therefore rarely, if ever, need a Léman
Pass. The French residents of the external population are not covered by this stage.

## Outputs

The stage returns the persons of the subscriptions stage with these columns, filled
for the agents with a Verbundabo:

- `subscription_operator`: the fare network of the subscription;
- `subscription_zones`: its zones, e.g. `ZVV:110,121` (empty when none is found);
- `subscription_zones_method`: `own_route`, `group_route`, `home_zone`,
  `same_zone`, `fallback` or `none`;
- `subscription_product`: `Verbundabo` or `Léman Pass`;
- `lemanpass` and `lemanpass_annual_chf`: the Léman Pass product and its price.

`verbund_zones.csv` (one row per agent) and `od_matrix.csv` (agents by zone OD)
are written in the stage folder, with the cached timetable and tracks.

`subscription_zones_map.html` in the stage folder is a control map of the results.
It draws the homes of a sample of agents (drawn with `random_seed`, spread over the
cases), coloured by method, on the zone outlines. A click on a home outlines its
assigned zones in red and draws its destination and its own commute, routed again:
rides coloured by how they counted (in the network, partly in it, leaving it, after
leaving it) and walks dashed. Its popup compares the assigned zones with the zones
of that journey, which can differ for `group_route` agents (their zones come from
another agent of the same zone OD). A panel summarises all the agents by case,
method, product, number of zones and network.

`matsim.scenario.population` uses this stage instead of the subscriptions stage
(`use_subscription_zones`, default `true`). The Swiss synthetic persons with a
regional subscription get the person attribute `RegionalSubscriptionProducts`: the
zones per fare network, the networks joined by `+`, e.g. `ZVV:110,121` or
`ZVV:110,121+Unireso:10`. It is left out when no zones are found. A Léman Pass is
written like any other regional subscription (its product is not written), and
`subscriptions` keeps its values (`none`, `GA`, `VA`, `HT`, `VA+HT`). The external and cross-border agents get no zones. With
`use_subscription_zones: false`, the population is written as before.
