"""The zones of the regional PT subscriptions (Verbundabos) of the persons of the microcensus.

The Java pricing (RunComputeTransitPrices) prices a trip with a Verbundabo by the zones of the subscription
(RegionalSubscriptionProducts): without them, the whole trip is paid. The survey says who has a Verbundabo, not which
zones it covers, so they are found as for the synthetic population (synthesis.population.models.subscription_zones):
from the commute between the home and the place of education (else of work), routed on the timetable.

The stage returns person_id, subscription_zones ("ZVV:110,121"; empty when none is found), subscription_zones_method
and subscription_product for the persons with a Verbundabo, and writes them to verbund_zones.csv.
"""
import logging

import geopandas as gpd
import numpy as np

from synthesis.population.models.subscription_zones import configure_routing, routing_tools
from utils.gtfs_router.verbund import VERBUND_CODES, verbund_zones

logger = logging.getLogger("synpp")

# The places that can be commute destinations: Switzerland and its border regions, in LV95 (x, y)
LV95_BOX = ((2350000, 2950000), (1000000, 1400000))


def configure(context):
    configure_routing(context)
    context.config("random_seed")
    context.stage("data.microcensus.persons")


def execute(context):

    # Load the persons with a Verbundabo, their home and the places of their commute (LV95).

    survey = context.stage("data.microcensus.persons")
    holders = survey[survey["subscriptions_verbund"].fillna(False).astype(bool)]
    persons = holders[["person_id", "age", "home_x", "home_y"]].assign(pt_subscription = VERBUND_CODES[0])
    work, education = places(holders, "work"), places(holders, "education")
    logger.info(f"Microcensus: {len(persons)} persons with a Verbundabo, {len(work)} with a work place, {len(education)} with a place of education")

    # Route their commutes as for the synthetic population, each person on its own journey.

    tools = routing_tools(context)
    result = verbund_zones(persons, work, education, tools.tracks.zones, tools.router, tools.day, tools.pricing,
                           departure = context.config("subscription_zones_departure"), samples = 0,
                           seed = context.config("random_seed"), workers = max(1, int(context.config("threads"))),
                           lemanpass = tools.lemanpass)
    logger.info("Microcensus Verbundabos by method: " + ", ".join(f"{m} {n}" for m, n in result["method"].value_counts().items()))

    # Write and return the zones of each person.

    result.to_csv("%s/verbund_zones.csv" % context.path(), index = False)
    zones = result[["person_id", "verbund_zones", "method", "subscription"]].rename(columns = {
        "verbund_zones": "subscription_zones", "method": "subscription_zones_method", "subscription": "subscription_product"})
    zones["subscription_zones"] = zones["subscription_zones"].fillna("")
    return zones


def places(persons, kind):
    """The work or education places of persons (kind_x, kind_y in LV95) as points, for those who have one. The work
    places of data.microcensus.persons are converted from the survey even when they are missing codes (-99...), so
    only those of employed persons within LV95_BOX count."""
    if f"{kind}_x" not in persons:
        logger.warning(f"Microcensus: no {kind} places in data.microcensus.persons")
        return gpd.GeoDataFrame({"person_id": []}, geometry = gpd.points_from_xy([], []), crs = 2056)
    x, y = persons[f"{kind}_x"].to_numpy(dtype = float), persons[f"{kind}_y"].to_numpy(dtype = float)
    (x_min, x_max), (y_min, y_max) = LV95_BOX
    has = np.isfinite(x) & np.isfinite(y) & (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    if kind == "work":
        has &= persons["employed"].fillna(False).astype(bool).to_numpy()
    return gpd.GeoDataFrame({"person_id": persons["person_id"].to_numpy()[has]},
                            geometry = gpd.points_from_xy(x[has], y[has]), crs = 2056)
