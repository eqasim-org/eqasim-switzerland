"""Compatibility stage exposing zone locations with assigned foreign crossings."""


from synthesis.freight.npvm.zone_locations import read_zones, read_additional_centroids, enterprise_candidates


def configure(context):
    context.stage("synthesis.freight.npvm.crossing_points")


def execute(context):

    # Reuse the single crossing-selection stage for all freight analysis consumers.

    return context.stage("synthesis.freight.npvm.crossing_points")
