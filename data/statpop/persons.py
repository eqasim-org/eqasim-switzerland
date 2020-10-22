def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    with xz.open("%s/statpop/STATPOP_2012_Personen.csv.xz" % data_path) as f:
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
            "FEDERALBUILDINGID": int,
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
            "FEDERALBUILDINGID": "federal_building_id",
        }

        return data.utils.read_csv(context, f, fields, renames, total=8261094)
