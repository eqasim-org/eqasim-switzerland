import pandas as pd


def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    part_1 = f"{data_path}/statpop/STATPOP_PP_2023_TEIL_1.csv"
    part_2 = f"{data_path}/statpop/STATPOP_PP_2023_TEIL_2.csv"

    fields = {
        "personPseudoID": int,
        "SEX": int,
        "AGE": int,
        "MARITALSTATUS": int,
        "NATIONALITYCATEGORY": int,
        "GEOCOORDN": float,
        "GEOCOORDE": float,
        "POPULATIONTYPE": int,
        "TYPEOFRESIDENCE": int,
        "REPORTINGMUNICIPALITYID": int,
        "INDIC_EGID": int,
    }

    renames = {
        "personPseudoID": "person_id",
        "SEX": "sex",
        "AGE": "age",
        "MARITALSTATUS": "marital_status",
        "NATIONALITYCATEGORY": "nationality",
        "GEOCOORDN": "home_y",
        "GEOCOORDE": "home_x",
        "POPULATIONTYPE": "population_type",
        "TYPEOFRESIDENCE": "type_of_residence",
        "REPORTINGMUNICIPALITYID": "municipality_id",
        "INDIC_EGID": "statpop_geocoding_indicator",
    }

    with open(part_1, mode="rb") as f1, open(part_2, mode="rb") as f2:
        df1 = data.utils.read_csv(context, f1, fields, renames, total=None)
        df2 = data.utils.read_csv(context, f2, fields, renames, total=None)

    # Combine them into one DataFrame
    df = pd.concat([df1, df2], ignore_index=True)

    # Legacy flag used by matching to restrict donors to home-only activity chains.
    # It detects administrative-centre/unknown coordinates, NOT verified residence
    # in an institution: geocoded dorm/care-home residents can have this flag False.

    municipality_centers = pd.read_excel(f"{data_path}/spatial/municipality_centers/be-b-00.03-gg22.xlsx", sheet_name = "g1g22")
    municipality_centers = list(zip(municipality_centers["E_CNTR"], municipality_centers["N_CNTR"]))

    df["coords"] = list(zip(df["home_x"], df["home_y"]))

    df["collective_housing_resident"] = False
    df.loc[df["coords"].isin(municipality_centers),     "collective_housing_resident"] = True
    df.loc[(df["home_x"] == -9) & (df["home_y"] == -9), "collective_housing_resident"] = True # Unknown home location?

    del df["coords"]

    return df
