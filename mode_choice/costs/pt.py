"""Read subscription-aware fares produced during PT routing."""
from mode_choice.costs.artifacts import read_costs


def configure(context):
    context.stage("mode_choice.routing_variables.pt")
    context.stage("mode_choice.defaults")


def execute(context):
    return read_costs(context, "pt")
