"""Use the Java road router with either supported mode-choice package layout."""

from importlib.util import find_spec

# Keep the SUMO change independent of the separate mode-choice reorganization.
try:
    _client = find_spec('mode_choice.routing.java.client')
except ModuleNotFoundError:
    _client = None

if _client is None:
    from mode_choice.routing.RoutingServer import RoutingServer
    SERVER_STAGE = 'mode_choice.routing.server'
else:
    from mode_choice.routing.java.client import RoutingServer
    SERVER_STAGE = 'mode_choice.routing.java.server'
