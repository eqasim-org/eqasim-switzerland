import numpy as np
import pandas as pd
import geopandas as gpd
from random import choices

import data.spatial.urbanisation_level
import data.constants as c
import data.microcensus.income

import SNN.models as dcm

"""
This stage fuses sampled STATPOP data with microcensus data.
"""

dic_noga_to_letter = {"acc_and_food": "I",
                          "admin": "N",
                          "arts": "R",
                          "construction": "F",
                          "education": "P",
                          "el_gas_steam": "D",
                          "finance_insurance": "K",
                          "health": "Q",
                          "it": "J",
                          "manufact": "C",
                          "other_services": "S",
                          "public_admin": "O",
                          "real_estate": "L",
                          "retail": "G",
                          "scientific": "M",
                          "transport": "H"}

dic_noga_to_number = {"acc_and_food": "9",
                          "admin": "14",
                          "arts": "18",
                          "construction": "6",
                          "education": "16",
                          "el_gas_steam": "4",
                          "finance_insurance": "11",
                          "health": "17",
                          "it": "10",
                          "manufact": "3",
                          "other_services": "19",
                          "public_admin": "15",
                          "real_estate": "12",
                          "retail": "7",
                          "scientific": "13",
                          "transport": "8"}

dic_isco_to_number = {"clerical": 4,
                      "craft": 7,
                      "elementary": 9,
                      "machine_operators": 8,
                      "manager": 1,
                      "professional": 2,
                      "service_sales": 5,
                      "technician": 3}

dic_cat_company_size = {"0-4": 2, "5-9": 7, "10-19": 15, "20-29": 25, "30-49": 40, "50-74": 62,
                        "75-99": 87, "100-149": 125, "150-199": 175, "200-249": 225, "250-299": 275,
                        "300-499": 400, "500-999": 750, "1000+": 1877}


def configure(context):
    context.stage("synthesis.population.enriched")

    if context.config("run_snn"):
        context.config("run_snn")
        context.config("snn_day")
        context.config("data_path")
        context.config("snn_var_path")
        context.config("output_path")
        context.config("random_seed")
        context.stage("data.microcensus.persons")
        context.stage("data.microcensus.households")
        context.stage("data.microcensus.household_persons")
        context.stage("data.statent.statent")
        context.stage("data.spatial.urbanisation_level")
        context.stage("synthesis.population.spatial.home.locations")
        context.stage("synthesis.population.spatial.primary.work.locations")


def impute_workplace_attributes(df_persons, home, work, companies, df_municipality_types):

    print("  INFO: imputing available information from work locations and statent.")

    # Compute commute distance
    work.columns = ["person_id", "work_dest_id", "work_geometry"]
    home.columns = ["household_id", "home_dest_id", "home_geometry"]

    link         = df_persons[["person_id", "household_id"]]
    link_commute = pd.merge(link, work, on = "person_id", how = "right")
    link_commute = pd.merge(link_commute, home, on = "household_id", how = "inner")
    work_gdf = gpd.GeoDataFrame(link_commute["work_geometry"], geometry="work_geometry", crs='epsg:2056')
    home_gdf = gpd.GeoDataFrame(link_commute["home_geometry"], geometry="home_geometry", crs='epsg:2056')

    link_commute.loc[:, "commute_distance"] = work_gdf.distance(home_gdf)
    link_commute.loc[link_commute["commute_distance"]<=0.1, "commute_distance"] = 0.1
    link_commute.loc[:, "log_commute_km"]   = [np.log(x / 1000) for x in link_commute["commute_distance"]]

    link_commute = link_commute[["person_id", "household_id", "work_dest_id", "log_commute_km"]]

    companies    = companies[["enterprise_id", "number_employees", "noga08_section", "municipality_type", "x", "y", "municipality_id"]]
    companies    = companies.rename(columns = {"municipality_type": "work_municipality_type"})

    df_persons = pd.merge(df_persons, link_commute, how = "left", on = ["person_id", "household_id"])
    df_persons["log_commute_km"] = df_persons["log_commute_km"].fillna(np.nan)

    df_persons = pd.merge(df_persons, companies, left_on = "work_dest_id", right_on = "enterprise_id", how = "left")

    print("  INFO: Imputing spatial information")

    df_persons = df_persons.merge(df_municipality_types, on="municipality_id", how = "left")
    df_persons.loc[:, "wk_urbanization_high"]   = df_persons["urbanisation_level"] == "high"
    df_persons.loc[:, "wk_urbanization_medium"] = df_persons["urbanisation_level"] == "medium"
    df_persons.loc[:, "wk_urbanization_low"]    = df_persons["urbanisation_level"] == "low"

    return df_persons


def impute_hhl_attributes(df_persons, hhl_info, data_path):

    df_persons = pd.merge(df_persons, hhl_info, left_on = "mz_head_id", right_on="household_id", how = "left")

    df_mz_households = pd.read_csv(
        "%s/microcensus/haushalte.csv" % data_path,
        sep = ",", encoding = "latin1"
    )

    df_mz_households["mz_head_id"] = df_mz_households["HHNR"]
    columns = ["mz_head_id"]

    #df_mz_households["income_class"] = df_mz_households["F20601"] -1
    #df_mz_households["income_class"] = np.maximum(-1, df_mz_households["income_class"])  # Make all "invalid" entries -1
    #df_mz_persons = data.microcensus.income.impute(df_mz_persons)
    #columns.extend(["income_class"])

    # Parking at home
    df_mz_households.loc[:, "parking_home"] = (df_mz_households["f31100"] > 0)
    columns.extend(["parking_home"])

    # Secondary residence
    df_mz_households.loc[:, "re_2nd_ch"]  = (df_mz_households["ZW1_LND"] == 8100)
    df_mz_households.loc[:, "re_2nd_out"] = (df_mz_households["ZW1_LND"] >= 8100)
    columns.extend(["re_2nd_ch", "re_2nd_out"])

    # Household type
    df_mz_households["hhl_type"] = df_mz_households["hhtyp"]
    columns.extend(["hhl_type"])    

    df_mz_households = df_mz_households[columns]
    df_persons = df_persons.merge(df_mz_households, on = "mz_head_id", how = "left")

    # Fix children and marital status information
    df_persons.loc[:, "has_children"] = (df_persons["n_children"]>1) & (df_persons["hhl_type"]>=220)
    df_persons.loc[:, "marital_status_divorced"] = (df_persons["marital_status"]==2) & (df_persons["hhl_type"]>=220)
    df_persons.loc[:, "marital_status_married"]  = (df_persons["marital_status"]==1) & (df_persons["hhl_type"].isin([220, 210]))
    df_persons.loc[:, "marital_status_married_sep"]  = (df_persons["marital_status"]==1) & (~df_persons["hhl_type"].isin([220, 210]))

    # Income. Convert to an income in 1000CHF/month
    df_persons.loc[:, "hh_income"] = [(2*x+1)*1000 for x in df_persons["income_class"]]

    return df_persons


def impute_personal_attributes(df_persons, data_path):

    df_mz_persons = pd.read_csv(
        "%s/microcensus/zielpersonen.csv" % data_path,
        sep = ",", encoding = "latin1", parse_dates = ["USTag"]
    )

    df_mz_persons["mz_person_id"] = df_mz_persons["HHNR"]
    columns = ["mz_person_id"]

    # Parking at workplace
    df_mz_persons["parking_work"] = "unknown"
    df_mz_persons.loc[df_mz_persons["f41300"] == 1, "parking_at_work"] = "free"
    df_mz_persons.loc[df_mz_persons["f41300"] == 2, "parking_at_work"] = "paid"
    df_mz_persons.loc[df_mz_persons["f41300"] == 3, "parking_at_work"] = "no"
    df_mz_persons["parking_at_work"] = df_mz_persons["parking_at_work"].astype("category")

    df_mz_persons.loc[:, "parking_work"] = (df_mz_persons["parking_at_work"] == "free") | (df_mz_persons["parking_at_work"] == "paid")
    columns.extend(["parking_work"])

    # Highest education
    df_mz_persons["highest_education"] = np.nan
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([1, 2, 3, 4]), "highest_education"] = "primary"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([5, 6, 7, 8, 9, 10, 11, 12]), "highest_education"] = "secondary"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([13, 14, 15, 16, 17, 18, 19]), "highest_education"] = "tertiary"
    df_mz_persons["highest_education"] = df_mz_persons["highest_education"].astype("category")

    df_mz_persons.loc[:, "education_higher"]    = (df_mz_persons["highest_education"] == "tertiary")
    df_mz_persons.loc[:, "education_mandatory"] = (df_mz_persons["highest_education"] == "primary")
    df_mz_persons.loc[:, "education_secondary"] = (df_mz_persons["highest_education"] == "secondary")
    columns.extend(["education_higher"])
    columns.extend(["education_mandatory"])
    columns.extend(["education_secondary"])

    # Is a leader at workplace
    df_mz_persons.loc[:, "is_leader"] = (df_mz_persons["f41102"] == 1)
    columns.extend(["is_leader"])

    # Is working full time
    df_mz_persons.loc[:, "wk_full_time"] = (df_mz_persons["f40900"] == 1)
    columns.extend(["wk_full_time"])

    # ISCO
    for key, value in dic_isco_to_number.items():
        df_mz_persons.loc[:, "isco_" + key] = (df_mz_persons["ISCO_08"] == value)
        columns.extend(["isco_" + key])
    columns.extend(["ISCO_08"])

    # Work from home
    df_mz_persons.loc[:, "wfa_1"] = (df_mz_persons["f81300"] == 1) | (df_mz_persons["f81300"] == 2)
    df_mz_persons.loc[:, "wfh"]   = [max(0, x) for x in df_mz_persons["f81400"]]
    df_mz_persons["wfh"] = [round(w // 20) for w in df_mz_persons["wfh"]]
    columns.extend(["wfa_1", "wfh"])

    # Work schedule
    df_mz_persons.loc[:, "wk_schedule_fixed"]    = (df_mz_persons["f81200"]==1)
    df_mz_persons["wk_schedule_fixed"] = df_mz_persons["wk_schedule_fixed"].fillna(False)
    df_mz_persons.loc[:, "wk_schedule_flexible"] = (df_mz_persons["f81200"]==4)

    # Employment status
    df_mz_persons.loc[df_mz_persons["f40900"]==1, "emp"] = "full_time"
    df_mz_persons.loc[df_mz_persons["f40900"]==2, "emp"] = "part_time"
    df_mz_persons.loc[df_mz_persons["f40900"]==3, "emp"] = "mult_part_time"
    columns.extend(["emp"])

    # NOGA
    df_mz_persons["wiabt_08"] = df_mz_persons["wiabt_08"].astype(np.int32)
    columns.extend(["wiabt_08", "wk_schedule_fixed", "wk_schedule_flexible"])
    for key, value in dic_noga_to_number.items():

        df_mz_persons.loc[:, "wk_noga_"+key] = (df_mz_persons["wiabt_08"] == value)
        columns.extend(["wk_noga_"+key])

    # Merge
    df_mz_persons = df_mz_persons[columns]
    df_persons = df_persons.merge(df_mz_persons, how = "left", on = "mz_person_id")

    df_persons["wk_schedule_fixed"]    = df_persons["wk_schedule_fixed"].fillna(False)
    df_persons["wk_schedule_flexible"] = df_persons["wk_schedule_flexible"].fillna(False)

    return df_persons


def impute_missing_attributes(df_persons, companies):

    # Work schedule
    filter = (~df_persons["wk_schedule_fixed"]) & (~df_persons["wk_schedule_flexible"])
    proba_annual = 0.252873563218391
    proba_other  = 0.0266457680250784
    proba_shift  = 0.114420062695925
    prob_vector = [proba_annual, proba_other, proba_shift]
    prob_vector = prob_vector / np.sum(prob_vector)
    schedule_choices = ["annual", "other", "shift"]
    len_missing = len(df_persons[filter])
    df_persons.loc[filter, "work_model"] = choices(schedule_choices, weights=prob_vector, k=len_missing)
    df_persons.loc[:, "wk_schedule_annual"] = (df_persons["work_model"] == "annual")
    df_persons.loc[:, "wk_schedule_other"]  = (df_persons["work_model"] == "other")
    df_persons.loc[:, "wk_shiftwork"]       = (df_persons["work_model"] == "shift")

    # Company car
    proba_company_car = 0.118599791013584
    df_persons.loc[:, "has_company_car"] = choices([1, 0], weights = [proba_company_car, 1-proba_company_car], k = len(df_persons))

    # Housing type
    proba_single_house = 0.258620689655172
    proba_appt         = 0.644200626959248
    prob_vector = [proba_single_house, proba_appt, 1 - proba_single_house - proba_appt]
    housing_choices = ["single house", "appt", "other"]
    df_persons.loc[:, "housing_type"] = choices(housing_choices, weights = prob_vector, k = len(df_persons))
    df_persons.loc[:, "re_type_apartment"]    = (df_persons["housing_type"] == "single house")
    df_persons.loc[:, "re_type_single_house"] = (df_persons["housing_type"] == "appt")

    # Number of employees per company
    companies = companies[["enterprise_id", "number_employees", "noga08_section"]]
    for key, value in dic_cat_company_size.items():
        if key != "1000+":
            bin_inf = int(key.split("-")[0])
            bin_sup = int(key.split("-")[1])
        else:
            bin_inf = 1000
            bin_sup = 15000

        companies.loc[(companies["number_employees"] >= bin_inf) & (companies["number_employees"] <= bin_sup), "sum_nb_empl"] = value
        companies.loc[(companies["number_employees"] >= bin_inf) & (companies["number_employees"] <= bin_sup), "cat_company_size"] = key
    
    g = companies.groupby(["cat_company_size", "noga08_section"])["sum_nb_empl"].sum().reset_index()

    alphabet = "ABCDEFGHIJKLMNOPQRS"
    for number in range(21):
        if number > 0 and number  < len(alphabet):
            letter = alphabet[int(number)]
            comp_sector = g[g["noga08_section"]==letter]
            comp_sector["sum_nb_empl"] = comp_sector["sum_nb_empl"] / np.sum(comp_sector["sum_nb_empl"])
            choices_values = comp_sector["cat_company_size"].values.tolist()
            choice_weights = comp_sector["sum_nb_empl"].values.tolist()

            filter = df_persons["wiabt_08"]==number
            df_persons.loc[filter, "size_company"] = choices(choices_values, choice_weights, k = len(df_persons[filter]))

    df_persons.loc[df_persons["employed"], "wk_firm_size_1_9"]    = (df_persons["size_company"].isin(["0-4", "5-9"]))
    df_persons.loc[df_persons["employed"], "wk_firm_size_10_49"]  = (df_persons["size_company"].isin(["10-19", "20-29", "30-49"])) 
    df_persons.loc[df_persons["employed"], "wk_firm_size_50_249"] = (df_persons["size_company"].isin(["50-74", "75-99", "100-149", "150-199", "200-249"]) )
    df_persons.loc[df_persons["employed"], "wk_firm_size_250_inf"] = (df_persons["size_company"].isin(["250-299", "300-499", "500-999", "1000+"]))   

    return df_persons


def check_and_select(df_persons, snn_var_path):

    variables = pd.read_excel(snn_var_path)
    cols_sel = variables["variable"].values.tolist()
    cols_sel.append("emp")

    copy_persons = df_persons.copy()
    copy_persons.loc[: , "ID"] = copy_persons["person_id"]
    copy_persons = copy_persons[copy_persons["employed"]]
    copy_persons = copy_persons[cols_sel]

    for variable in cols_sel:

        if variable not in ["ID", "age", "hh_income", "log_commute_km", "n_adults", "n_small_children", "wfh", "emp"]:
            copy_persons[variable] = copy_persons[variable].astype(int)


    copy_persons["hh_income"] = copy_persons["hh_income"] / 1000

    return df_persons, copy_persons


def execute(context):

    df_persons = context.stage("synthesis.population.enriched")

    if context.config("run_snn"):
        hhl_info = context.stage("data.microcensus.household_persons")[1].copy()

        days_of_the_week = context.config("snn_day")
        days_of_the_week = days_of_the_week.split("-")
        for day in days_of_the_week:
            assert day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


        data_path = context.config("data_path")
        companies = context.stage("data.statent.statent")
        home      = context.stage("synthesis.population.spatial.home.locations")
        work      = context.stage("synthesis.population.spatial.primary.work.locations")

        snn_var_path = context.config("snn_var_path")

        df_municipality_types = context.stage("data.spatial.urbanisation_level")

        print("INFO: starting to add attributes to the synthetic population.")
        persons_original = df_persons.copy()
        df_persons = df_persons
        
        # Gender
        df_persons.loc[:, "sex_male"] = df_persons["sex"] == 0
        #del df_persons["sex"]

        # Nationality
        df_persons.loc[:, "swiss"] = df_persons["nationality"] == 1
        #del df_persons["nationality"]

        # Driving license
        df_persons.loc[:, "is_driver"] = df_persons["driving_license"]
        #del df_persons["driving_license"]

        # Urbanisierungsgrad
        df_persons.loc[:, "municipality_id"] = df_persons["home_municipality_id"]
        df_persons = data.spatial.urbanisation_level.impute(df_persons, df_municipality_types)
        df_persons.loc[:, "re_urbanization_high"]   = df_persons["urbanisation_level"] == "high"
        df_persons.loc[:, "re_urbanization_medium"] = df_persons["urbanisation_level"] == "medium"
        df_persons.loc[:, "re_urbanization_low"]    = df_persons["urbanisation_level"] == "low"
        del df_persons["municipality_id"]
        del df_persons["urbanisation_level"]#
        del df_persons["home_municipality_id"]

        print("INFO: adding workplace attributes to the synthetic population.")
        df_persons    = impute_workplace_attributes(df_persons, home, work, companies, df_municipality_types)

        print("INFO: adding information about household to the synthetic population.")
        df_persons    = impute_hhl_attributes(df_persons, hhl_info, data_path)

        print("INFO: adding additional personal attributes.")
        df_persons    = impute_personal_attributes(df_persons, data_path)

        print("INFO: impute unknown attributes.")
        df_persons    = impute_missing_attributes(df_persons, companies)

        print("INFO: check before mode choice and WFH imputations")
        df_persons, red = check_and_select(df_persons, snn_var_path) 

        # Separating employment status from the rest of the dataframes as it doesn't comply with JSON requirements
        emp_status = red[["ID", "emp"]]
        emp_status = emp_status.fillna("full_time")
        del red["emp"]

        print("INFO: work from home ability and willingness.")
        imputed_persons = dcm.wfh_status(red)

        print("INFO: mode tool ownership")
        imputed_persons = dcm.mob_tool_ownership(imputed_persons) 

        print("INFO: work_from_home on the specific day")
        choices_set = days_of_the_week
        number_of_days = len(choices_set)
        proba = 1.0 / number_of_days
        imputed_persons.loc[: , "weekday"] = choices(choices_set, weights=[proba for _ in range(number_of_days)], k = len(imputed_persons))

        imputed_persons = pd.merge(imputed_persons, emp_status, on = "ID")
        imputed_persons = dcm.weekday_probability(imputed_persons) 

        imputed_persons = imputed_persons[["ID", "wfh_today", "wfa_1", "wfh", "has_ga", "has_ca", "has_ht", "has_re", "has_bi", "has_cs", "wfh_the_days"]]

        df_persons = persons_original.merge(imputed_persons, how="left", right_on = "ID", left_on = "person_id")

        # Replace mobility tool ownership in population
        df_persons.loc[~df_persons["has_ga"].isna(), "subscriptions_ga"]      = df_persons[~df_persons["has_ga"].isna()]["has_ga"]
        df_persons.loc[~df_persons["has_ht"].isna(), "subscriptions_halbtax"] = df_persons[~df_persons["has_ht"].isna()]["has_ht"]
        df_persons.loc[~df_persons["has_re"].isna(), "subscriptions_verbund"] = df_persons[~df_persons["has_re"].isna()]["has_re"]

        # Car availability
        df_persons.loc[(~df_persons["has_ca"].isna()) & (df_persons["has_ca"]==0) & (df_persons["car_availability"]==2), "car_availability"] = 2 # never
        df_persons.loc[(~df_persons["has_ca"].isna()) & (df_persons["has_ca"]==0) & (df_persons["car_availability"]==1), "car_availability"] = 2
        df_persons.loc[(~df_persons["has_ca"].isna()) & (df_persons["has_ca"]==0) & (df_persons["car_availability"]==0), "car_availability"] = 2 
        df_persons.loc[(~df_persons["has_ca"].isna()) & (df_persons["has_ca"]==1), "car_availability"] = 0 # always

        # Looks fine!
        df_persons["wfh_today"] = df_persons["wfh_today"].fillna(False)

        # Adjust mobility tool ownership for the unemployed adult population
        random = np.random.RandomState(context.config("random_seed"))
        filter = ~(df_persons["employed"]) & (df_persons["age"]>=18)

        filter_car = filter &  (df_persons["car_availability"] <= 1)
        probability = (0.851-0.665)/0.851
        df_persons.loc[filter_car, "car_availability"] = 2*(random.random_sample(size=(len(df_persons[filter_car]),)) < probability  )

        filter_GA = filter & df_persons["subscriptions_ga"].isna() & (df_persons["subscriptions_ga"]==1)
        probability = (1-0.095 - 1 + 0.118 ) / (1-0.095)
        df_persons.loc[filter_GA, "subscriptions_ga"] = random.random_sample(size=(len(df_persons[filter_GA]),)) < probability  

        filter_HT = filter & df_persons["subscriptions_halbtax"].isna() & (df_persons["subscriptions_halbtax"]==1)
        probability = (1-0.404 - 1 + 0.59) / (1-0.404)
        df_persons.loc[filter_HT, "subscriptions_halbtax"] = random.random_sample(size=(len(df_persons[filter_HT]),)) < probability  

        filter_regio = filter & df_persons["subscriptions_verbund"].isna() & (df_persons["subscriptions_verbund"]==1)
        probability = (1-0.116 - 1 + 0.172) / (1-0.116)
        df_persons.loc[filter_regio, "subscriptions_verbund"] = random.random_sample(size=(len(df_persons[filter_regio]),)) < probability

        return df_persons

    return df_persons
