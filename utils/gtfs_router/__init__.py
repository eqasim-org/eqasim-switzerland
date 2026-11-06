"""Journeys on a GTFS timetable and the fare zones they go through, copied from the Swiss PT explorer (gtfs_router,
github.com/AuroreSallard/swiss-pt-explorer) for synthesis.population.models.subscription_zones.

- feed: reads the GTFS timetable into a Network (build_network), and the network of one date (Network.on);
- csa: the Router, which finds the best Journey between two places (Connection Scan Algorithm);
- pricing: the fare networks and zones of the stops, and the zones of a ride (Pricing);
- zones and tracks: the zone polygons, and the real path of the rides and the zones they cross;
- lemanpass: the Léman Pass products around Geneva;
- verbund: the zones of the Verbundabos of a synthetic population.

Keep these files in step with the Swiss PT explorer when its method changes.
"""
