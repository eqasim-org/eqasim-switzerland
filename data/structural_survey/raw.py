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
                "COMPANYLOCUNNOGA": str,
                "COMPANYENPNOGA": str,
                "MAINMODETRANSPWORK": int,
                weight_column: float,
                "STATYEAR": int,
                "SCHOOL_MUN": int,
                "SCHOOL_QUARTER": int,
                # School commute can start at a term-time/secondary residence.
                # Keep its geography separate from the registered RES_* fields.
                "DEPFORSCHOOL": int,
                "DEPFORSCHOOLMUN": int,
                "DEPFORSCHOOLQUARTER": int,
                "DEPFORSCHOOLCTRY": int,
                "MAINRESIDENCECATEGORY": int,
                "SCHOOL_ADDRESSCTRY": int,
                "SCHOOLLOCUNNOGA": str,
                "SCHOOLENPNOGA": str,
                "MAINMODETRANSPSCHOOL": int,
                "FREQPERWEEK_TOSCHOOL": float,
                "COMMUTERTOSCHOOL_DETAIL": int,
                "AGE": int,
                "SEX": int,
                "CURRACTIVITYSTATUSI": int,
                "STATUSINEMPL_DETAIL": int,
                "RES_CANTON": int,
                "RES_DISTRICT": int,
                "CURRACTIVITY_STUDENT": int,
                "NATIONALITYCAT": int,
                "ONGOINGEDUCATION": int,
                "IN_EDUCATION": int,
                "DISTTOWORK_AIR": float,
                "DISTTOSCHOOL_AIR":float,                       
                "FREQPERWEEK_TOWORK": float,
                "COMMUTERTOWORK_DETAIL": int,                
                "STARTWORK": int,
                "HIGHESTCOMPLEDU": int,
                "CURROCCUPATIONISCO": int
            }

            renames = {
                "RES_MUN": "home_municipality",
                "RES_QUARTER": "home_quarter",
                "COMPANY_MUN": "work_municipality",
                "COMPANY_QUARTER": "work_quarter",
                "COMPANY_CTRY": "work_country",
                "COMPANYLOCUNNOGA": "work_noga",
                "COMPANYENPNOGA": "work_enterprise_noga",
                "MAINMODETRANSPWORK": "mode",
                weight_column: "weight",
                "STATYEAR": "survey_year",
                "SCHOOL_MUN": "education_municipality",
                "SCHOOL_QUARTER": "education_quarter",
                "DEPFORSCHOOL": "education_departure_code",
                "DEPFORSCHOOLMUN": "education_departure_municipality",
                "DEPFORSCHOOLQUARTER": "education_departure_quarter",
                "DEPFORSCHOOLCTRY": "education_departure_country_code",
                "MAINRESIDENCECATEGORY": "main_residence_category",
                "SCHOOL_ADDRESSCTRY": "education_country_code",
                "SCHOOLLOCUNNOGA": "education_noga",
                "SCHOOLENPNOGA": "education_enterprise_noga",
                "MAINMODETRANSPSCHOOL": "education_mode_code",
                "FREQPERWEEK_TOSCHOOL": "education_freq_per_week",
                "COMMUTERTOSCHOOL_DETAIL": "commute_to_education",
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
                "IN_EDUCATION": "in_education",
                "DISTTOWORK_AIR": "crowfly_distance_to_work",
                "DISTTOSCHOOL_AIR": "crowfly_distance_to_school",
                "FREQPERWEEK_TOWORK": "freq_per_week",
                "COMMUTERTOWORK_DETAIL": "commute_to_work", #0: no displacement or no fixed work location, 2: in the same commune, ...6: abroad
                "STARTWORK": "start_work", #1: a domicile, 2: different locations, 3: employer adress, >4: other adresses
                "HIGHESTCOMPLEDU": "highest_completed_education", #1: None, ... 11: Bachelor, 12: Master, 13: Doctorate       
                "CURROCCUPATIONISCO": "isco_code"            
            }            
            data_frames.append(data.utils.read_csv(context, f, fields, renames, total=total, sep=sep))

    return pd.concat(data_frames, sort=True)
