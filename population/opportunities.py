import pandas as pd


def configure(context):
    context.stage("data.statent.statent")

def execute(context):
    df_opportunities = pd.DataFrame(context.stage("data.statent.statent")[[
        "enterprise_id", "x", "y", "noga"
    ]], copy = True)
    df_opportunities.columns = ["location_id", "location_x", "location_y", "noga"]

    df_opportunities.loc[:, "offers_work"] = True
    df_opportunities.loc[:, "offers_service"] = True

    # 85 = education
    df_opportunities.loc[:, "offers_education"] = df_opportunities["noga"].str.startswith("85")

    # 90 = arts, entertainment, leisure; 56 = gastronomy
    df_opportunities.loc[:, "offers_leisure"] = df_opportunities["noga"].str.startswith("90") | df_opportunities["noga"].str.startswith("56")

    # 47 = retail
    df_opportunities.loc[:, "offers_shop"] = df_opportunities["noga"].str.startswith("47")

    del df_opportunities["noga"]
    return df_opportunities
