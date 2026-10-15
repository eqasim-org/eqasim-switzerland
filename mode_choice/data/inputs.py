"""Assemble trip-keyed routing and cost outputs into the model input table."""
from mode_choice.defaults import ROUTING_VARIABLE_STAGES, COST_STAGES, PERSON_AVAILABILITY_COLUMNS
from mode_choice.data.frames import join_variables


def configure(context):
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.availabilities")
    for stage in (*ROUTING_VARIABLE_STAGES, *COST_STAGES):
        context.stage(stage)


def execute(context):
    # we get the trips from the prepare stage, which is the last stage that modifies the trips table
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        raise ValueError("No trips eligible for mode choice")

    # we attach the routing variables and costs to the trips table
    for stage in (*ROUTING_VARIABLE_STAGES, *COST_STAGES):
        trips = join_variables(trips, context.stage(stage))

    # Replace the person-level ownership flags with the final mode availabilities.
    trips = trips.drop(PERSON_AVAILABILITY_COLUMNS)
    trips = join_variables(trips, context.stage("mode_choice.data.availabilities"))
    return trips
