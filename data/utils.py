import numpy as np
import pandas as pd


def fix_marital_status(df, c):
    """ Makes young people, who are separated, be treated as single! """
    f = ((df["marital_status"] == c.MARITAL_STATUS_SEPARATE) & 
         (df["age"] < c.SEPARATE_SINGLE_THRESHOLD))
    df.loc[f, "marital_status"] = c.MARITAL_STATUS_SINGLE
    df["marital_status"] = df["marital_status"].astype(np.int)


def assign_household_class(df, census):
    """
        Combines all houeshold sizes above 5 into one class.

        Attention! Here KM also says that houesholds with at least one married person
        have a minimum size of 2. Technically, this doesn't need be true in reality, and
        I'm not sure if it has any implications later on. (TODO)
    """

    if census == "statpop":
        df["household_size_class"] = np.minimum(5, df["household_size"]) - 1

    elif census == "are_synpop":
        df["household_size_class"] = "5+"
        df.loc[df["household_size"]==1, "household_size_class"] = "1"
        df.loc[df["household_size"]==2, "household_size_class"] = "2"
        df.loc[df["household_size"]==3, "household_size_class"] = "3-4"
        df.loc[df["household_size"]==4, "household_size_class"] = "3-4"


def read_csv(context, fp, fields, renames=None, sep=";", total=None, encoding="latin1", limit=None, label="Reading csv file..."):
    if renames is None:
        renames = {}
    header = None
    data = []

    count = 0

    for line in context.progress(fp, total=total, label=label):
        line = line.decode(encoding).strip().split(sep)

        if header is None:
            header = line
        else:
            data.append([
                field_function(line[header.index(field_name)])
                for field_name, field_function in fields.items()
            ])

        count += 1

        if limit is not None and count == limit:
            break

    columns = [
        renames[field_name] if field_name in renames else field_name
        for field_name in fields.keys()
    ]

    return pd.DataFrame.from_records(data, columns=columns)
