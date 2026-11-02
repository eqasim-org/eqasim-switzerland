# Routing organization

```text
mode_choice/routing/
  java/       server/client, requests, config, preparation, routing,
              PT components/pricing, population exports and SUMO bridges
  python/     Pandana routing, hourly linkstats, walk/bike factors, simple PT prices
  flash/      FLASH-TB build, timetable, native queries, PT adapter and benchmarks
  options.py  shared routing/pricing configuration and validation
  results.py  missing/invalid route handling
```

The Synpp stages in `mode_choice.routing_variables` are the public entry points.
They return trip IDs plus routing variables. PT pricing is selected independently:

```yaml
mode_choice_pt_routing_backend: flash  # java (default) or flash
# Omit pricing: java for Java routing, python for FLASH routing.
mode_choice_pt_price_backend: python  # explicit override: java or python
```

Java routing with Java pricing asks for both in one request. Python pricing always
uses `dmc.cost.pt.estimate_simple_cost`, whichever router produced the distance.
FLASH with Java pricing calls the server's `/cost` endpoint; raw PT inputs require
Java to compute its own itinerary internally. This can differ from the FLASH path.

Fare artifacts keep the existing `routing_variables.pt/cost.parquet` contract.
Routing implementations do not duplicate the Python fare model.
[FLASH documentation and benchmarks](flash/README.md) describe the experimental limits.
