def configure(context):
    context.config("raw_data_path")
    # require.cache = False

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    import lzma as xz
    import data.utils

    with xz.open("%s/statpop/STATPOP_2012_Link_Pers_HH.csv.xz" % raw_data_path) as f:
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

        return data.utils.read_csv(f, fields, renames, total = 8261094)
