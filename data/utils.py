import numpy as np
import pandas as pd

BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "off", ""})


def coerce_boolean_series(values, *, default=False, name=None):
    """Return a strict bool Series from bool, 0/1, or common string encodings.

    ``Series.astype(bool)`` treats every non-empty string—including ``"False"``—as
    true. CSV and GeoJSON readers can infer the same logical field differently
    depending on whether missing values are present, so pipeline boundaries must
    parse values rather than rely on their inferred dtype.
    """
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    result = pd.Series(pd.NA, index=series.index, dtype="boolean", name=series.name)

    non_missing = series.notna()
    normalized = series.astype("string").str.strip().str.lower()
    result.loc[non_missing & normalized.isin(TRUE_VALUES)] = True
    result.loc[non_missing & normalized.isin(FALSE_VALUES)] = False

    invalid = non_missing & result.isna()
    if invalid.any():
        examples = series.loc[invalid].drop_duplicates().head(5).tolist()
        field = name or series.name or "boolean field"
        raise ValueError(f"{field} contains values that are not boolean: {examples!r}")

    return result.fillna(bool(default)).astype(bool)


def to_base62(n):
    """
    Encodes a non-negative integer as a base62 string (0-9, A-Z, a-z).
    Keep identical to eqasim-france's copy (data/id_utils.py).
    """
    n = int(n)
    if n == 0:
        return "0"

    digits = []
    while n > 0:
        n, remainder = divmod(n, 62)
        digits.append(BASE62_ALPHABET[remainder])

    return "".join(reversed(digits))


def fix_marital_status(df, c):
    """ Makes young people, who are separated, be treated as single! """
    f = ((df["marital_status"] == c.MARITAL_STATUS_SEPARATE) & 
         (df["age"] < c.SEPARATE_SINGLE_THRESHOLD))
    df.loc[f, "marital_status"] = c.MARITAL_STATUS_SINGLE
    df["marital_status"] = df["marital_status"].astype(int)


def assign_household_class(df):
    """
        Combines all houeshold sizes above 5 into one class.

        Attention! Here KM also says that houesholds with at least one married person
        have a minimum size of 2. Technically, this doesn't need be true in reality, and
        I'm not sure if it has any implications later on. (TODO)
    """
    df["household_size_class"] = np.minimum(5, df["household_size"]) - 1


def prepare_education_locations(df_persons, df_statent, c):

    census = c.census

    filters_persons   = []
    filters_locations = []
    query_sizes       = None
    education_types   = None 

    if census == "statpop":
        age_bounds      = [(-np.inf, 6), (6, 12), (12, 16), (16, np.inf)]
        education_types = ["kindergarten", "primary", "secondary", "tertiary"]
        query_sizes     = (1, 2, 7, 20)

        for (lower_bound, upper_bound), type in zip(age_bounds, education_types):
            filter_persons  = (df_persons["age"] > lower_bound) & (df_persons["age"] <= upper_bound)
            filter_location = df_statent["education_type"] == type

            filters_persons.append(filter_persons)
            filters_locations.append(filter_location)

    elif census == "are_synpop":
        query_sizes     = (1, 5, 20)
        education_types = ["kindergarten", "primary and secondary", "tertiary"]

        f_under_6  = df_persons["age_class"] == 0
        f_6_17     = df_persons["age_class"] == 1
        f_above_18 = df_persons["age_class"] >= 2

        f_kindergarden      = df_statent["education_type"] == "kindergarten"
        f_primary_secondary = (df_statent["education_type"] == "primary") | (df_statent["education_type"] == "secondary")
        f_tertiary          = df_statent["education_type"] == "tertiary"

        filters_persons = [f_under_6, f_6_17, f_above_18]
        filters_locations = [f_kindergarden, f_primary_secondary, f_tertiary]

    return filters_persons, filters_locations, education_types, query_sizes


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
