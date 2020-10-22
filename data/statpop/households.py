def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    with xz.open("%s/statpop/STATPOP_2012_PHH.csv.xz" % data_path) as f:
        fields = {
            "householdIdNum": int,
            "Plausibel": int
        }

        renames = {
            "householdIdNum": "household_id",
            "Plausibel": "plausible"
        }

        return data.utils.read_csv(context, f, fields, renames, total=3488739)
