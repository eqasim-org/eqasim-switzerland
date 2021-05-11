import pandas as pd

import data.constants as c


def impute(df):
    df_head = pd.DataFrame(df)

    # 1) Only consider persons that have reached a certain age (remember, there are no complete underage households anymore!)
    df_head = df_head[df_head["age"] >= c.MINIMUM_AGE_PER_HOUSEHOLD]

    # 2) For households with at least one person under ACTIVE_AGE, filter out all persons above that age.
    # For all other houesholds, keep all persons:

    df_head.loc[:, "is_in_active_age"] = df_head["age"] <= c.ACTIVE_AGE

    df_filter = df_head[["household_id", "is_in_active_age"]].groupby("household_id").sum().reset_index()
    df_filter["active_count"] = df_filter["is_in_active_age"]

    df_head = pd.merge(df_head, df_filter[["household_id", "active_count"]])
    df_head = df_head[(df_head["active_count"] == 0) | (df_head["age"] <= c.ACTIVE_AGE)]

    # 3) TODO: Not completely sure: I think now we should keep all persons that are married in households where there are ANY
    # married persons. For other households, we keep all persons:

    df_head.loc[:, "is_married"] = df_head["marital_status"] == c.MARITAL_STATUS_MARRIED
    df_filter = df_head[["household_id", "is_married"]].groupby("household_id").sum().reset_index()
    df_filter["married_count"] = df_filter["is_married"]
    df_head = pd.merge(df_head, df_filter[["household_id", "married_count"]])
    df_head = df_head[(df_head["married_count"] == 0) | df_head["is_married"]]

    # 4) Now sort by age (oldest first), and by gender (male first):
    df_head = df_head.sort_values(by = ["household_id", "age", "sex", "person_id"], ascending = [True, False, True, True])

    # 5) Finally, get the heads of houeshold:
    df_head = df_head[["person_id", "household_id"]].groupby("household_id").first().reset_index()
    df_head["head_id"] = df_head["person_id"]
    del df_head["person_id"]

    # 6) Merge into main data frame
    df = pd.merge(df, df_head[["household_id", "head_id"]])
    df.loc[:, "is_head"] = df["head_id"] == df["person_id"]
    del df["head_id"]
    return df
