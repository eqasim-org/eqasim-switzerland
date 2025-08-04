def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    with open("%s/statpop/DSV250221_STATPOP_HH_2021-2023/STATPOP_2023_HOUSEHOLD_CH_K.csv" % data_path, mode="rb") as f:
        fields = {
            "HOUSEHOLDID": int
            #"Plausibel": int
        }

        renames = {
            "HOUSEHOLDID": "household_id"
            #"Plausibel": "plausible"
        }

        return data.utils.read_csv(context, f, fields, renames, total=3964840)
