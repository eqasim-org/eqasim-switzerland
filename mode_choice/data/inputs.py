"""Assemble trip-keyed routing and cost outputs into the model input table."""
from mode_choice.defaults import ROUTING_VARIABLE_STAGES, COST_STAGES
from mode_choice.data.frames import join_variables
from mode_choice.model.availability import prepare


def configure(context):
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.defaults")
    for stage in (*ROUTING_VARIABLE_STAGES, *COST_STAGES):
        context.stage(stage)


def execute(context):
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips
    for stage in (*ROUTING_VARIABLE_STAGES, *COST_STAGES):
        trips = join_variables(trips, context.stage(stage))
    return prepare(trips)
