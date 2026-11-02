"""Shared survey mobility features, prepared once for all expert folds."""
from .tools import mobility_features


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("data.spatial.ovgk")


def execute(context):

    # Compute home accessibility and subscription encoding once per person.

    persons = context.stage("data.microcensus.persons")
    features = mobility_features(persons, context.stage("data.constants"), survey=True,
                                 ovgk=context.stage("data.spatial.ovgk"))
    features["person_id"] = persons.person_id.to_numpy()

    # Share the immutable feature table across production and cross-fitted experts.

    return features
