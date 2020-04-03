import numpy as np
import pandas as pd

import data.constants as c

INDEX_RENAMES = {0: "total",
                 1: "truck",
                 2: "delivery_van"}


def configure(context):
    context.config("data_path")
    context.config("scaling_year")


def execute(context):
    data_path = context.config("data_path")

    # Select year in the future to project to
    scaling_year = np.max([c.BASE_SCALING_YEAR, context.config("scaling_year")])

    # Load excel for projections
    df = pd.read_excel(
        "%s/projections/are/freight/Verkehrsperspektiven_2040_Ergebnisse_Gueterverkehr_de.xlsx" % data_path,
        sheet_name="Fahrzeugkilometer_Referenz", header=9,
        index_col=None, nrows=3
    ).dropna(axis=1)[[2010,2020,2030,2040]].rename(index=INDEX_RENAMES).reset_index().rename(columns={"index":"type"})

    # Convert to long format
    df = df.melt(
        id_vars = "type", value_vars = [2010, 2020, 2030, 2040],
        value_name = "vehicle_km", var_name = "year"
    )

    # Pivot years to columns
    df = df.pivot_table(
        index = ["type"], columns = ["year"]
    )

    # Add new interpolated column
    df[("vehicle_km", scaling_year)] = df.apply(
        lambda x: max(0, 1e6 * np.round(
            np.dot(
                np.polyfit(
                    [2010, 2020, 2030, 2040],
                    [x["vehicle_km", 2010], x["vehicle_km", 2020], x["vehicle_km", 2030], x["vehicle_km", 2040]],
                    1
                ),
                [scaling_year, 1]
            )
        ).astype(int)),
        axis=1)

    # Reformat
    df = df[("vehicle_km", scaling_year)].reset_index()
    df.columns = ["type", "vehicle_km"]

    return df
