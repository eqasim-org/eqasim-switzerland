def configure(context):
    context.config("raw_data_path")
    # require.cache = False

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    import lzma as xz
    import data.utils

    with xz.open("%s/statpop/STATPOP_2012_PHH.csv.xz" % raw_data_path) as f:
        fields = {
            "householdIdNum" : int,
            "Plausibel" : int
        }

        renames = {
            "householdIdNum" : "household_id",
            "Plausibel" : "plausible"
        }

        return data.utils.read_csv(f, fields, renames, total = 3488739)
