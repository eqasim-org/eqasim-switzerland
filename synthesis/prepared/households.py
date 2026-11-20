"""Assemble final synthetic household attributes and memberships."""

import logging
import pandas as pd
from .tables import Households

logger = logging.getLogger("synpp")
FIELDS = [
    "household_id",
    "person_id",
    "income_class",
    "age",
    "number_of_cars_class",
    "municipality_type",
    "sp_region",
    "canton_id",
    "ovgk",
    "canton_name",
    "income_per_capita",
]
INCOME_VALUES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.constants")
    context.stage("data.spatial.cantons")

    if context.config("include_cross_border", False):
        context.stage("data.cross_border.generate_cross_border_traffic")

    context.config("include_external_population", default=True)
    if context.config("include_external_population"):
        context.stage("data.external_population.read_outputs")
        context.stage("data.external_population.constants")


def execute(context):

    # Combine resident groups and compute household income and regional attributes.

    c = context.stage("data.constants")

    df_persons = context.stage("synthesis.population.enriched").sort_values(
        by=["household_id", "person_id"]
    )
    df_cantons = context.stage("data.spatial.cantons")[
        ["canton_id", "canton_name_en"]
    ].copy()

    # Attach canton name to agent (TODO: do it in previous stages, keep track of canton name)
    df_cantons = df_cantons.rename(columns={"canton_name_en": "canton_name"})
    df_persons["canton_id"] = df_persons["canton_id"].astype(int)
    df_persons = pd.merge(
        df_persons, df_cantons, left_on="canton_id", right_on="canton_id", how="left"
    )
    assert (
        df_persons.canton_name.notnull().all()
    ), "Not all persons have a canton name assigned. Check the canton data."

    # Attach real average income per person per household
    df_persons[["income", "income_per_capita"]] = get_income(df_persons, c)

    _require_cols(
        df_persons,
        [
            "household_id",
            "person_id",
            "income_class",
            "number_of_cars_class",
            "municipality_type",
            "sp_region",
            "canton_id",
            "ovgk",
            "canton_name",
            "income_per_capita",
        ],
        "df_persons",
    )

    # Keep only the fields you need, but don't crash if you later add extras elsewhere
    df_persons = df_persons[[c for c in FIELDS if c in df_persons.columns]]

    if context.config("include_external_population"):
        external_persons = context.stage("data.external_population.read_outputs")[
            0
        ].copy()
        ex_constants = context.stage("data.external_population.constants")

        external_persons["municipality_type"] = ex_constants.municipality_type
        external_persons["sp_region"] = ex_constants.sp_region
        external_persons["canton_id"] = ex_constants.canton_id
        external_persons["ovgk"] = ex_constants.ovgk

        external_persons["canton_name"] = ex_constants.canton_name
        if (
            "income_per_capita" not in external_persons.columns
            or "income" not in external_persons.columns
        ):
            external_persons[["income", "income_per_capita"]] = get_income(
                external_persons, c, "French"
            )

        external_persons = external_persons[
            [c for c in FIELDS if c in external_persons.columns]
        ]
        df_persons = pd.concat([df_persons, external_persons])

    if context.config("include_cross_border"):
        cross_border_persons = context.stage(
            "data.cross_border.generate_cross_border_traffic"
        )[0].copy()
        cross_border_persons = cross_border_persons.sort_values(by="person_id")

        cross_border_persons["household_id"] = cross_border_persons["person_id"].values

        cross_border_persons["municipality_type"] = "crossborder"
        cross_border_persons["sp_region"] = -1
        cross_border_persons["canton_id"] = 0
        cross_border_persons["ovgk"] = "crossborder"

        cross_border_persons["canton_name"] = "outsideCH"
        cross_border_persons["income_per_capita"] = 0

        cross_border_persons = cross_border_persons[
            [c for c in FIELDS if c in cross_border_persons.columns]
        ]
        df_persons = pd.concat([df_persons, cross_border_persons])

    # Group the combined populations globally, so each household is written once.

    df_persons = df_persons.sort_values(["household_id", "person_id"])
    if df_persons.person_id.isna().any() or df_persons.person_id.duplicated().any():
        raise ValueError("Household membership requires unique, non-missing person IDs")
    if df_persons.household_id.isna().any():
        raise ValueError("Household IDs must be present before export")
    members = df_persons[["household_id", "person_id"]].copy()
    households = (
        df_persons.drop_duplicates("household_id")
        .drop(columns=["person_id", "age"], errors="ignore")
        .copy()
    )

    # Resolve household income while retaining the original car-count category.

    households["income_class"] = households.income_class.astype(int)
    households["income"] = households.income_class.map(dict(enumerate(INCOME_VALUES)))
    if households.income.isna().any():
        raise ValueError("Household income class is outside the defined income range")
    households["canton_id"] = households.canton_id.fillna(-1)
    return Households(
        households.reset_index(drop=True),
        members.reset_index(drop=True),
        maximum_car_class=int(c.MAX_NUMBER_OF_CARS_CLASS),
    )


def _require_cols(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def _na_to_default(x, default):
    return default if pd.isna(x) else x


def get_income(df_persons, cst, population="Swiss"):
    cols = ["household_id", "income_class", "age"]
    cols = cols + ["household_size"] if "household_size" in df_persons.columns else cols
    df = df_persons[cols].copy()

    # transform income class into income
    df["income"] = df["income_class"].astype(int).map(cst.INCOME_CLASS_MAP)

    # Calculate income per capita using the OECD equivalence scale: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Equivalised_income
    if len(df) == df["household_id"].nunique():
        logger.warning(
            f"There is no household structure for this population:{population}"
        )
        # If household structure is not kept, we use an average household structure (average is also very near to median here)
        num_children = 1
        num_adults = 2
    else:
        if "household_size" not in df.columns:
            # If household_size is not present, we can calculate it by counting the number of persons per household
            df["household_size"] = df.groupby("household_id")["age"].transform("count")

        df["is_child"] = df["age"] < 14
        num_children = df.groupby("household_id")["is_child"].transform("sum")
        num_adults = (df["household_size"] - num_children).clip(0, 8)
        if not ((num_adults >= 1).all()):
            sel = num_adults < 1
            logger.warning(
                f"There are {sel.sum()} households with no adult in this population: {population}"
            )
            logger.warning(
                f"\t Minimum age in these households is: { df[sel].age.min() }"
            )
            num_adults[sel] = 1
            num_children[sel] -= 1

    equvalent_size = 1 + 0.5 * (num_adults - 1) + 0.3 * num_children
    df["income_per_capita"] = df["income"] / equvalent_size

    return df[["income", "income_per_capita"]]


def validate(context):
    from .tables import implementation_token

    return implementation_token(__file__)
