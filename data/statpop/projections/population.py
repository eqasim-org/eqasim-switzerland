import pandas as pd
import numpy as np
import data.constants as c

def configure(context, require):
    require.config("raw_data_path")
    require.config("scaling_year")
    # require.cache = False

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    # load excel data
    df = pd.read_csv("%s/projections/population/px-x-0104020000_101.csv" % raw_data_path, sep=";", encoding="latin1", skiprows=1)

    # rename columns
    renames = {"Kanton": "canton_id",
               "Geschlecht": "sex",
               "Alter": "age",
               "Jahr": "year",
               "Bevölkerungsstand am 1. Januar": "weight"}
    df = df.rename(renames, axis=1)

    # replace canton names with ids
    df = df.replace({"Zürich": 1,
                         "Bern / Berne": 2,
                         "Luzern": 3,
                         "Uri": 4,
                         "Schwyz": 5,
                         "Obwalden": 6,
                         "Nidwalden": 7,
                         "Glarus": 8,
                         "Zug": 9,
                         "Fribourg / Freiburg": 10,
                         "Solothurn": 11,
                         "Basel-Stadt": 12,
                         "Basel-Landschaft": 13,
                         "Schaffhausen": 14,
                         "Appenzell Ausserrhoden": 15,
                         "Appenzell Innerrhoden": 16,
                         "St. Gallen": 17,
                         "Graubünden / Grigioni / Grischun": 18,
                         "Aargau": 19,
                         "Thurgau": 20,
                         "Ticino": 21,
                         "Vaud": 22,
                         "Valais / Wallis": 23,
                         "Neuchâtel": 24,
                         "Genève": 25,
                         "Jura": 26})

    # turn sex and nationality into an actual 0-based class
    df = df.replace({"Mann": 0, "Frau": 1}).replace({"Schweiz": 0, "Ausland": 1})

    # turn age into integer
    df["age"] = df["age"].str.split("Jahr", expand=True)[0].astype(int)

    # Get the age class
    df["age_class"] = np.digitize(df["age"], c.AGE_CLASS_UPPER_BOUNDS)

    # aggregate
    df = df[["canton_id", "sex", "age_class", "year", "weight"]]
    df = df.groupby(["canton_id", "sex", "age_class", "year"]).sum().reset_index()

    # select year in the future to project to (default = 2018)
    scaling_year = np.max([c.BASE_YEAR, context.config["scaling_year"]])

    # create lists of cantons, household sizes and years from data
    canton_ids = list(df["canton_id"].unique())
    sexes = list(df["sex"].unique())
    age_classes = list(df["age_class"].unique())
    years = list(df["year"].unique())

    # fill data
    if (scaling_year in years):
        df = df[df["year"] == scaling_year].drop("year", axis=1).reset_index().drop("index", axis=1)
    else:
        # build empty data array
        index = np.arange(0, len(canton_ids) * len(sexes) * len(age_classes))
        columns = ["canton_id", "sex", "age_class", "weight"]
        data = np.zeros((len(index), len(columns)), dtype=object)

        # for years not between 2015 and 2045, the data is linearly interpolated
        i = 0
        for canton_id in canton_ids:
                for sex in sexes:
                    for age_class in age_classes:
                        data[i][0] = canton_id
                        data[i][1] = sex
                        data[i][2] = age_class

                        # interpolate value for future year
                        temp = df[(df["canton_id"] == canton_id) &
                                  (df["sex"] == sex) &
                                  (df["age_class"] == age_class)]

                        # linear fit over last 5 years
                        xp = temp["year"].values[-5:]
                        fp = temp["weight"].values[-5:]
                        coeff = np.polyfit(xp, fp, 1)
                        value = coeff[0] * scaling_year + coeff[1]

                        # values cannot be negative
                        data[i][3] = np.max([0, round(value)])

                        i += 1

        df = pd.DataFrame(data=data, index=index, columns=columns)

    return df
