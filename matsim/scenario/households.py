"""Write cached household attributes and memberships as MATSim XML."""

import gzip
import io
from itertools import groupby
import matsim.writers

# Preserve helper imports used by existing local callers.
from synthesis.prepared.households import (
    get_income,
    INCOME_VALUES,
    FIELDS,
)


def configure(context):
    context.stage("synthesis.prepared.households")


def execute(context):

    # Read globally sorted household and member tables from preparation.

    data = context.stage("synthesis.prepared.households")
    output = "%s/households.xml.gz" % context.path()
    groups = iter(
        groupby(
            data.members.itertuples(index=False), key=lambda member: member.household_id
        )
    )

    # Stream one household and its member IDs at a time without mutating cached data.

    with gzip.open(output, "wb") as stream:
        with io.BufferedWriter(stream, buffer_size=2 * 1024**3) as buffered:
            writer = matsim.writers.HouseholdsWriter(buffered)
            writer.start_households()
            for household in context.progress(
                data.households.itertuples(index=False), total=len(data.households)
            ):
                household_id, members = next(groups)
                if household_id != household.household_id:
                    raise ValueError(
                        "Prepared households and memberships are not aligned"
                    )
                writer.start_household(household.household_id)
                writer.add_members([member.person_id for member in members])
                writer.add_income(household.income)
                writer.start_attributes()
                attributes = (
                    ("incomeClass", "java.lang.Integer", household.income_class),
                    (
                        "numberOfCars",
                        "java.lang.String",
                        _car_class_text(
                            household.number_of_cars_class, data.maximum_car_class
                        ),
                    ),
                    (
                        "municipalityType",
                        "java.lang.String",
                        household.municipality_type,
                    ),
                    ("spRegion", "java.lang.Integer", household.sp_region),
                    ("ovgk", "java.lang.String", household.ovgk),
                    ("cantonName", "java.lang.String", household.canton_name),
                    (
                        "incomePerCapita",
                        "java.lang.Double",
                        household.income_per_capita,
                    ),
                    ("cantonId", "java.lang.Double", household.canton_id),
                )
                for name, kind, value in attributes:
                    writer.add_attribute(name, kind, str(value))
                writer.end_attributes()
                writer.end_household()
            writer.end_households()

    # Preserve the original stage result expected by scenario/output consumers.

    return output


def _car_class_text(value, maximum):
    """Encode the upper car-count category using MATSim's existing plus notation."""
    return "%d+" % maximum if value == maximum else str(value)


def write_number_of_cars_class(value, constants):
    """Compatibility entry point for existing callers of the household writer."""
    return _car_class_text(value, constants.MAX_NUMBER_OF_CARS_CLASS)
