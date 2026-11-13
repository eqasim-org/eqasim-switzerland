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
3. The agents are grouped by zone OD. The subscription covers:
   - the home zone, without a destination or with one in the same zone;
   - the zones of the home network along the commute, within one network;
   - the Z-Pass zones along the commute, between two networks joined by a Z-Pass;
   - the zones of the home network until the first ride that leaves it, between
     two networks;
   - the zones of the destination network back from the destination, for a home
     in no network.
4. In each zone OD, `subscription_zones_samples` commutes are routed on the
   timetable of one day, leaving at `subscription_zones_departure`. The other
   agents of the OD get the zones of one of these journeys, drawn with
   `random_seed`. Without any journey, the home and destination zones are used.
5. A commute into the Léman Pass area abroad gets the cheapest annual Léman Pass
   product that covers it, when `lemanpass.csv` and `lemanpass_places.csv` are in
   `subscription_zones_lemanpass_path`. A product of zones only (`Zone 10 + Zone 230`)
   covers every journey inside its zones, trains included: Genève - Lancy-Bachet by
   train and the bus to Saint-Julien gets `Zone 10 + Zone 230`. This rule is in the
   eqasim copy of `lemanpass.py`, not yet in the Swiss PT explorer.

## Inputs and options

| Option | Default | Meaning |
| --- | --- | --- |
| `subscription_zones_date` | empty | Routing date (YYYYMMDD); empty: `gtfs_date` if it is a date, else the first Tuesday of the feed |
| `subscription_zones_departure` | `07:30` | Departure time of the routed commutes |
| `subscription_zones_samples` | 3 | Commutes routed per zone OD |
| `subscription_zones_lemanpass_path` | `pt_pricing/output` | Folder of the Léman Pass inputs, relative to `data_path` |
| `subscription_zones_bus_tracks` | `true` | Buses follow their routes in `matsim.scenario.network.pt`; `false`: straight lines, without that dependency |
| `pt_zones_lines_file` | `pt_pricing/linie-mit-polygon.geojson` | SBB rail lines, for the train tracks |
| `subscription_zones_map_agents` | 400 | Agents drawn on the control map, spread over the cases; 0: no map |

The zone polygons use the `pt_zones_*` files of `data.pt_pricing.zones`. The
timetable is the output of `data.gtfs.cleaned`, and the fares those of
`data.pt_pricing.pt_pricing`. Routing uses `threads` processes.

The Léman Pass inputs are in `pt_pricing/output` (copied from the Swiss PT explorer).
They are written from the tpg fare table with
`python -m utils.gtfs_router.lemanpass <fare table.xlsx> <folder>`.

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
