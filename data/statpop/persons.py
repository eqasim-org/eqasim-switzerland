import pandas as pd

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils
    
#     d1 = pd.read_csv("%s/updated_data/statpop_2017/STATPOP_2017_Bestand_1.csv" % data_path, sep = ";")
    
#     d1 = d1[["personPseudoID",
#             "SEX",
#             "AGE",
#             "MARITALSTATUS",
#             "NATIONALITYCATEGORY",
#             "GEOCOORDN",
#             "GEOCOORDE",
#             "POPULATIONTYPE",
#             "TYPEOFRESIDENCE",
#             "REPORTINGMUNICIPALITYID",
#             "FEDERALBUILDINGID"]]
    
# #     d2 = pd.read_csv("%s/updated_data/statpop_2017/STATPOP_2017_Bestand_2.csv" % data_path, sep = ";")
    
# #     d2 = d2[["personPseudoID",
# #             "SEX",
# #             "AGE",
# #             "MARITALSTATUS",
# #             "NATIONALITYCATEGORY",
# #             "GEOCOORDN",
# #             "GEOCOORDE",
# #             "POPULATIONTYPE",
# #             "TYPEOFRESIDENCE",
# #             "REPORTINGMUNICIPALITYID",
# #             "FEDERALBUILDINGID"]]
    
# #     df = pd.concat([d1, d2])
    
#     df = df[["personPseudoID",
#             "SEX",
#             "AGE",
#             "MARITALSTATUS",
#             "NATIONALITYCATEGORY",
#             "GEOCOORDN",
#             "GEOCOORDE",
#             "POPULATIONTYPE",
#             "TYPEOFRESIDENCE",
#             "REPORTINGMUNICIPALITYID",
#             "FEDERALBUILDINGID"]]
            
#     df.columns = ["person_id", "sex", "age", "marital_status", "nationality", "home_y", "home_x", "population_type", "type_of_residence", "municipality_id", "federal_building_id"]
            
#     for col in df.columns:
#         df[col] = df[col].astype(int)
        
#     return df

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


