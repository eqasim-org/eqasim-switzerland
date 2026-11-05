"""Apply compact crossing overrides without caching another full population."""

import pandas as pd

STAGE = "data.cross_border.count_assignment"
FOREIGN = "data.cross_border.generate_od"
SWISS = "synthesis.population.models.cross_border"
REFERENCE = "data.cross_border.swiss_residents_od"


def configure_assignment(context):
    enabled = context.config("cross_border_count_assignment", True)
    include = context.config("include_cross_border", False)
    census = context.config("census", "statpop")
    # Joint shares require both populations. Other census producers retain their
    # existing assignment rather than silently using a different Swiss cohort.
    if enabled and include:
        if census != "statpop":
            raise ValueError("cross_border_count_assignment requires census=statpop; disable it for another census")
        context.stage(STAGE)


def active(context):
    return context.config("cross_border_count_assignment") and context.config("include_cross_border")


def configure_foreign(context):
    context.stage(FOREIGN)
    configure_assignment(context)


def configure_swiss(context):
    context.stage(SWISS)
    configure_assignment(context)


def configure_reference(context):
    context.stage(REFERENCE)
    configure_assignment(context)


def assigned_foreign(context):
    population = context.stage(FOREIGN)
    changes = context.stage(STAGE)["foreign"] if active(context) else pd.DataFrame()
    if changes.empty:
        return population
    result = population.copy()
    lookup = changes.set_index("cross_border_person_id")
    selected = result.cross_border_person_id.isin(lookup.index)
    points = result.loc[selected, "cross_border_person_id"].map(lookup.interview_geometry_point)
    result.loc[selected, "interview_point_id"] = result.loc[selected, "cross_border_person_id"].map(lookup.interview_point_id)
    result.loc[selected, "interview_geometry_point"] = points.to_numpy()
    # Projected origins are placeholders for the chosen entry. Keep the original
    # foreign residence unchanged; network_projection will refine the new entry.
    projected = selected & result.origin_is_projected.eq(True)
    for column, coordinate in (("origin_x", "x"), ("origin_y", "y")):
        result.loc[projected, column] = result.loc[projected, "interview_geometry_point"].map(lambda p: getattr(p, coordinate))
    result.loc[projected, "origin_point_id"] = result.loc[projected, "interview_point_id"]
    return result


def assigned_swiss(context):
    population = context.stage(SWISS)
    changes = context.stage(STAGE)["swiss"] if active(context) else pd.DataFrame()
    if changes.empty:
        return population
    # Copy only rewritten columns. The national table contains millions of
    # residents; sharing untouched columns avoids another multi-GB allocation
    # without relying on a particular pandas copy-on-write setting.
    result = population.copy(deep=False)
    columns = ("interview_point_id", "interview_geometry_point", "border_crossing_point", "cross_border_person_id")
    for column in columns:
        result[column] = population[column].copy()
    lookup = changes.set_index("person_id")
    selected = result.person_id.isin(lookup.index)
    for column in columns:
        result.loc[selected, column] = result.loc[selected, "person_id"].map(lookup[column]).to_numpy()
    return result


def assigned_reference(context):
    reference = context.stage(REFERENCE)
    aliases = context.stage(STAGE)["swiss_facilities"] if active(context) else pd.DataFrame()
    # A survey donor can serve different residents/directions at different
    # crossings. Append donor+point facility aliases, leaving the raw donor pool
    # and all original records intact. Never reuse one facility at two places.
    return pd.concat([reference, aliases], ignore_index=True) if len(aliases) else reference
