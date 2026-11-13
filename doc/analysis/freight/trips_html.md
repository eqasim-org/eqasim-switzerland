# Freight and LCV zone and trip explorer

Generate a report using the existing caches selected by any pipeline configuration:

```bash
uv run -m analysis.freight.trips_html config_dib_10.yml
```

Replace `config_dib_10.yml` with the configuration you want. The standalone command
resolves exact Synpp cache hashes from its `config` and reads its
`working_directory`; it does not execute pipeline stages or alter upstream caches.
It fails clearly if a matching source cache is missing. The report is written to
`<analysis_path>/freight_lcv_od.html`. Override that destination with `--output PATH`.
The configuration's original `run` list is not executed by this command.

The module is also a regular Synpp stage. Set the configuration's run list to:

```yaml
run:
  - analysis.freight.trips_html
```

Then run `uv run -m synpp your_config.yml`. Synpp resolves the declared trip and
zone dependencies normally, building missing dependencies if necessary. The stage
returns the exported HTML path. Both entry points use the same report generator.

## Using the report

1. **Demand:** choose Freight or LCV.
2. **Show:** choose Zones, Trips, Origins, or Destinations. Trips draws directional
   OD lines and endpoints; Origins and Destinations draw only those endpoints.
3. In **Zones** mode, choose **Outgoing** or **Incoming**, then click a polygon or
   choose a zone in the list. Outgoing highlights destination zones; Incoming
   highlights the origin zones where trips come from. The selected zone is purple.
   Connected origin zones are green and connected destination zones are orange;
   darker fill means more trips. The selected trips are drawn as lines.

The connected-zone table shows counts and shares for the selected zone, direction
and departure-hour range. Same-zone trips are included once. Click a table row to
filter to that OD pair and zoom to its connected polygon. Clear the connection or
zone selection to widen the view. Switching to Trips, Origins or Destinations shows
all trips for the time range without the zone filter; returning to Zones restores
the selected zone. Changing datasets clears the zone selection.

The trip table shows the first 100 filtered records; CSV downloads include every
filtered record. Trip details and CSV include source-zone IDs. Reset the time
filter to include all departure times, including unknown ones.

## Data and scope

LCV uses its cached NPVM zone IDs and the configured data directory's
`npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg`. With the default
`use_npvm_freight: true`, freight uses its sampled NPVM OD counts and the zone
result of `synthesis.freight.npvm.locations`. Additional NPVM zones without
published polygons are shown as circles at their original centroid coordinates.
With `use_npvm_freight: false`, freight uses NUTS regions from
`data.spatial.nuts`. Legacy freight drops its NUTS IDs after generating coordinates, so
the report recovers them by replaying the original multinomial OD draws using
cached GTE/GQGV demand inputs and the configured seed. It checks trip counts,
vehicle types and component-coordinate alignment with the final freight cache.
It does not infer foreign zones from border coordinates or rerun synthesis stages.
Foreign NUTS polygons retain their real location; associated trip endpoints remain
on the Swiss border as generated. NUTS levels overlap; use the zone list to select
any region. Polygons are simplified for display only (100 m for NUTS and 15 m for
NPVM); source-zone membership and trip coordinates are unchanged.

Every valid cached OD record is included, without additional sampling or expansion
to a full population. Counts refer to trip records, not distinct physical vehicles
or delivery tours. Source coordinates are EPSG:2056, transformed to WGS84 and
rounded to six decimal places. Invalid coordinates are excluded and counted.
Coincident endpoints are shown as points. OD lines are straight connections,
not road routes.

The HTML embeds Leaflet 1.9.4 and both datasets. Generation downloads Leaflet from
unpkg.com; opening the HTML needs internet access only for the attributed
swisstopo national map. The background has an opacity slider. Polygons, trip
visuals, tables, details and CSV remain usable if tiles fail. Swisstopo coverage
is limited beyond Switzerland. Source cache names, generation time and diagnostics
are recorded under **Source and scope**; private configuration contents are not
embedded.
