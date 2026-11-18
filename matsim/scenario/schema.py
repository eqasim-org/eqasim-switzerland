"""MATSim-specific attribute names with compatibility aliases for scenario fields."""

from synthesis.prepared.schema import (
    HOME_DESTINATION_ID,
    VEHICLE_FIELDS,
    PERSON_FIELDS,
    ACTIVITY_FIELDS,
    PERSONS_DTYPES,
    BOOLEAN_PERSON_COLUMNS,
)

ACTIVITY_ATTRIBUTES_TO_SAVE = dict(
    municipalityType="municipality_type",
    municipalityId="municipality_id",
    employeeDensity="employee_density",
    companiesDensity="companies_density",
    populationDensity="population_density",
    ovgk="ovgk",
)

SUBSCRIPTIONS = ["none", "GA", "VA", "HT", "VA+HT"]
