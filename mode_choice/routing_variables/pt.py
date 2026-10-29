"""Select PT routing and pricing independently, keeping one trip-keyed contract."""
from mode_choice.routing import options
from mode_choice.routing.java.client import RoutingServer
from mode_choice.routing.java.router import save_costs
from mode_choice.routing.results import complete_routes
from mode_choice.data.frames import join_variables


def configure(context):
    options.configure_pt(context)
    if options.pt_backend(context) == 'flash':
        from mode_choice.routing.flash import pt as routing
    else:
        from mode_choice.routing.java import pt as routing
    routing.configure(context)
    if options.pt_price_backend(context) == 'python':
        from mode_choice.routing.python import prices
        prices.configure(context)
    elif options.pt_backend(context) == 'flash':
        from mode_choice.routing.java import prices
        prices.configure(context)


def execute(context):

    # Route alternatives using the selected implementation.

    try:
        if options.pt_backend(context) == 'flash':
            from mode_choice.routing.flash import pt as routing
        else:
            from mode_choice.routing.java import pt as routing
        routes = routing.execute(context)

        return price_routes(context, routes)
    except BaseException:
        RoutingServer.stop_server()
        raise


def price_routes(context, routes):
    """Apply the production fare contract to routes from an existing warm engine."""
    # Java routing already saves fares when Java pricing is selected.

    if options.pt_backend(context) == 'java' and options.pt_price_backend(context) == 'java':
        return routes
    if options.pt_price_backend(context) == 'python':
        from mode_choice.routing.python import prices
    else:
        from mode_choice.routing.java import prices
    trips = context.stage('mode_choice.data.prepare')
    costs = prices.execute(context, trips, routes)

    # Missing prices invalidate the alternative, then persist the normal cost artifact.

    identifiers = trips.select('trip_id')
    if costs.select('trip_id').join(identifiers, on='trip_id', how='anti').height:
        raise ValueError('PT pricing returned unknown trip identifiers')
    costs = identifiers.join(costs, on='trip_id', how='left', validate='1:1', maintain_order='left')
    combined = join_variables(routes, costs)
    combined = complete_routes(trips, combined, [c for c in routes.columns if c.endswith('_time_min')])
    return save_costs(context, combined, 'pt')
