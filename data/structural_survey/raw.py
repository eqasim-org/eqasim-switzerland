import pandas as pd


def configure(context):
    context.config("data_path")

# Structural survey interviews all (e.g., unemployed or those going to school are also present) individuals age 15 and over
# The weights of all interviewed individuals sumup to the STATPOP values of the same year for those individuals
# Last 3 or 5 instances of this survey can be combined in order to ahve more detailed representation of flows
# we currently only use transport mode, but age is also available along with other variables

def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    data_frames = []

    for path, weight_column, total, sep in [
        ("%s/structural_survey/se_zpers_2021_ch.csv" % data_path, "WEIGHT2021", 286016, ";"),
        ("%s/structural_survey/se_zpers_2022_ch.csv" % data_path, "WEIGHT2022", 282750, ";"),
        ("%s/structural_survey/se_zpers_2023_ch.csv" % data_path, "WEIGHT2023", 317222, ";"),
    ]:
        with open(path, mode="rb") as f:
            fields = {
                "RES_MUN": int,
                "RES_QUARTER": int,
                "COMPANY_MUN": int,
                "COMPANY_QUARTER": int,
                "COMPANY_CTRY": int,
                "MAINMODETRANSPWORK": int,
                weight_column: float,
                "SCHOOL_MUN": int,
                "SCHOOL_QUARTER": int,
                "AGE": int,
                "SEX": int,
                "CURRACTIVITYSTATUSI": int,
                "STATUSINEMPL_DETAIL": int,
                "RES_CANTON": int,
                "RES_DISTRICT": int,
                "CURRACTIVITY_STUDENT": int,
                "NATIONALITYCAT": int,
                "ONGOINGEDUCATION": int,
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
                "SCHOOL_QUARTER": "education_quarter",
                "CURRACTIVITY_STUDENT": "IS_STUDENT",
                "AGE": "age",
                "SEX": "sex",
                "RES_CANTON": "canton_id",
                "RES_DISTRICT": "district_id",
                "CURRACTIVITYSTATUSI": "employed", # 1: yes, 2: no, 3: inactive
                "STATUSINEMPL_DETAIL": "job_position",
                "CURRACTIVITY_STUDENT": "is_student",
                "NATIONALITYCAT" : "nationality",
                "ONGOINGEDUCATION" : "current_education",
            }

            data_frames.append(data.utils.read_csv(context, f, fields, renames, total=total, sep=sep))

    return pd.concat(data_frames, sort=True)
