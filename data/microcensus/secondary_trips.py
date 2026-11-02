"""Derived diary purposes for secondary assignment; observed MZ trips stay intact."""

import pandas as pd


EDUCATION_PURPOSES = ["education", "education_main", "education_secondary"]


def recode_nonstudent_education(trips, nonstudent_ids,
                               purpose_columns=("purpose", "origin_purpose")):
    """Route nonstudent education visits through ``other``, retaining provenance.

    A reported education visit does not establish enrolment or a particular
    provider. The ordinary secondary destination pool is our agreed proxy for
    these visits. Recode both trip endpoints to keep the activity chain coherent.
    Person status, trip keys, coordinates, modes and times are not changed.
    """
    result = trips.copy()
    nonstudents = result["person_id"].isin(nonstudent_ids)
    for column in purpose_columns:
        original = "original_" + column
        if original not in result:
            result[original] = result[column].copy()
        selected = nonstudents & result[column].isin(EDUCATION_PURPOSES)
        if selected.any():
            # Some callers use categorical purposes without an ``other`` level.
            if isinstance(result[column].dtype, pd.CategoricalDtype):
                if "other" not in result[column].cat.categories:
                    result[column] = result[column].cat.add_categories(["other"])
            result.loc[selected, column] = "other"
    return result


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")


def execute(context):
    persons = context.stage("data.microcensus.persons")
    # MZ has no independent is_student column: 0/1 are nonstudents, while 2/3
    # include students and employed students. Unknown status stays unreclassified.
    status = pd.to_numeric(persons["employment_status"], errors="raise")
    nonstudents = persons.loc[status.isin([0, 1]), "person_id"]
    return recode_nonstudent_education(context.stage("data.microcensus.trips")[0], nonstudents)
