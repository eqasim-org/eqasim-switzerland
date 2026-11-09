"""Request PT fares from the shared Java server."""
from pathlib import Path
import logging
from mode_choice.routing.eqasim.client import RoutingServer
from mode_choice.routing.eqasim.server import ensure_server_started
from mode_choice.routing.eqasim.router import read_results


def configure(context):
    context.stage('mode_choice.routing.eqasim.server')
    context.stage('mode_choice.routing.eqasim.requests')


def execute(context, trips, routes):

    # FLASH itineraries cannot yet be priced directly by Java's saved-fare API.

    logging.getLogger(__name__).warning(
        'Java PT pricing for FLASH uses Java itineraries computed by the server; '
        'these can differ from FLASH itineraries and add routing work.')
    if trips.is_empty():
        import polars as pl
        return trips.select('trip_id').with_columns(pl.lit(None, dtype=pl.Float64).alias('pt_cost_CHF'))

    # Use the pricing endpoint; raw PT requests are routed internally by Java.

    ensure_server_started(context)
    output = Path(context.path()) / 'pt_prices.csv'
    RoutingServer.cost_csv(context.stage('mode_choice.routing.eqasim.requests'), output, 'pt')
    return read_results(output, 'identifier', {'cost': 'pt_cost_CHF'})
