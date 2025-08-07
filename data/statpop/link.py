def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    with open("%s/statpop/STATPOP_2023_LINK_CH.csv" % data_path, mode="rb") as f:
        fields = {
            "personPseudoId" : int,
            "HOUSEHOLDID" : int,
            "REPORTINGMUNICIPALITYID" : int
        }

        renames = {
            "personPseudoId" : "person_id",
            "HOUSEHOLDID" : "household_id",
            "REPORTINGMUNICIPALITYID" : "municipality_id"
        }

        return data.utils.read_csv(context, f, fields, renames, total = 9191057)
