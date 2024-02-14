import pandas as pd

def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    with xz.open("%s/statpop/STATPOP_2012_Link_Pers_HH.csv.xz" % data_path) as f:
       fields = {
           "personPseudoID" : int,
            "householdIdNum" : int,
            "REPORTINGMUNICIPALITYID" : int
       }

       renames = {
           "personPseudoID" : "person_id",
            "householdIdNum" : "household_id",
            "REPORTINGMUNICIPALITYID" : "municipality_id"
       }

       return data.utils.read_csv(context, f, fields, renames, total = 8689634)
    
    #df = pd.read_csv("%s/updated_data/statpop_2017/STATPOP_2017_Link_Person_Haushalten.csv" % data_path, sep = ";")
    #df = df[["personPseudoID", "HOUSEHOLDID", "REPORTINGMUNICIPALITYID" ]]
    
    df.columns = ["person_id", "household_id", "municipality_id"]
    
    #for col in df.columns:
    #    df[col] = df[col].astype(int)
        
    #return df
