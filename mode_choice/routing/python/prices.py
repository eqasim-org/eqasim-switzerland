"""Approximate PT fares using the existing DMC function and population attributes."""
import polars as pl
from dmc.parameters import Parameters
from dmc.cost.pt import estimate_simple_cost


def configure(context):
    context.stage('dmc.parameter_files')


def execute(context, trips, routes):

    # Attach routed distance while preserving trip identifiers and order.

    priced = trips.join(routes.select('trip_id', 'pt_in_vehicle_distance_km'), on='trip_id',
                        how='left', validate='1:1', maintain_order='left').to_pandas()
    if priced.empty:
        return trips.select('trip_id').with_columns(pl.lit(None, dtype=pl.Float64).alias('pt_cost_CHF'))

    # Map population subscriptions to the shared DMC fare model's input names.

    for field, codes in [('hasGeneralSubscription', [1]), ('hasRegionalSubscription', [2, 4]),
                         ('hasHalbtaxSubscription', [3, 4])]:
        priced[field] = priced.pt_subscription.isin(codes)
    priced['hasJuniorSubscription'] = False
    priced['hasGleis7Subscription'] = False
    radius = Parameters.from_context(context)['ptRegionalRadius_km']
    costs = estimate_simple_cost(priced, context, radius)

    # Return only the trip-keyed monetary output.

    return trips.select('trip_id').with_columns(pl.Series('pt_cost_CHF', costs))
