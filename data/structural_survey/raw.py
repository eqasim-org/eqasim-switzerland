import pandas as pd


def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    data_frames = []

    for path, weight_column, total, sep in [
        ("%s/structural_survey/se_zpers_2012_CH.csv.xz" % data_path, "WEIGHT2012", 286016, ","),
        ("%s/structural_survey/se_zpers_2011_CH.csv.xz" % data_path, "WEIGHT2011", 282750, ";"),
        ("%s/structural_survey/se_zpers_2010_CH.csv.xz" % data_path, "WEIGHT2010", 317222, ","),
    ]:
        with xz.open(path) as f:
            fields = {
                "RES_MUN": int,
                "RES_QUARTER": int,
                "COMPANY_MUN": int,
                "COMPANY_QUARTER": int,
                "COMPANY_CTRY": int,
                "MAINMODETRANSPWORK": int,
                weight_column: float,
                "SCHOOL_MUN": int,
                "SCHOOL_QUARTER": int
            }

            renames = {
                "RES_MUN": "home_municipality",
                "RES_QUARTER": "home_quarter",
                "COMPANY_MUN": "work_municipality",
                "COMPANY_QUARTER": "work_quarter",
                "COMPANY_CTRY": "work_country",
                "MAINMODETRANSPWORK": "mode",
                weight_column: "weight",
                "SCHOOL_MUN": "education_municipality",
                "SCHOOL_QUARTER": "education_quarter"
            }

            data_frames.append(data.utils.read_csv(context, f, fields, renames, total=total, sep=sep))

    return pd.concat(data_frames, sort=True)
