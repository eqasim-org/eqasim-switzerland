import pandas as pd
def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    part_1 = f"{data_path}/statpop/STATPOP_PP_2023_TEIL_1_2/STATPOP_PP_2023_TEIL_1.csv"
    part_2 = f"{data_path}/statpop/STATPOP_PP_2023_TEIL_1_2/STATPOP_PP_2023_TEIL_2.csv"

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
        #"FEDERALBUILDINGID": int,
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
       # "FEDERALBUILDINGID": "federal_building_id",
    }

    with open(part_1, mode="rb") as f1, open(part_2, mode="rb") as f2:
        df1 = data.utils.read_csv(context, f1, fields, renames, total=None)
        df2 = data.utils.read_csv(context, f2, fields, renames, total=None)

    # Combine them into one DataFrame
    df = pd.concat([df1, df2], ignore_index=True)

    return df
