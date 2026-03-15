import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("synpp")

CANTON_TO_ID: dict[str, int] = {
    "Zürich": 1,
    "Bern": 2,
    "Luzern": 3,
    "Uri": 4,
    "Schwyz": 5,
    "Obwalden": 6,
    "Nidwalden": 7,
    "Glarus": 8,
    "Zug": 9,
    "Freiburg": 10,
    "Solothurn": 11,
    "Basel-Stadt": 12,
    "Basel-Landschaft": 13,
    "Schaffhausen": 14,
    "Appenzell A.Rh.": 15,
    "Appenzell I.Rh.": 16,
    "St. Gallen": 17,
    "Graubünden": 18,
    "Aargau": 19,
    "Thurgau": 20,
    "Tessin": 21,
    "Waadt": 22,
    "Wallis": 23,
    "Neuenburg": 24,
    "Genf": 25,
    "Jura": 26
}

def _normalize_canton(name: str) -> str:
    """Fix minor spelling/spacing variants to match mapping keys."""
    if not isinstance(name, str):
        return name
    s = name.strip()
    s = s.replace("A. Rh.", "A.Rh.").replace("A. Rh", "A.Rh.")  # Appenzell A.Rh.
    s = s.replace("I. Rh.", "I.Rh.").replace("I. Rh", "I.Rh.")  # Appenzell I.Rh.
    s = s.replace("Sankt Gallen", "St. Gallen")
    return s


def configure(context):
    context.config("data_path")
    context.config("scaling_year")
    context.config("enable_scaling")
    context.stage("data.constants")


def execute(context):
    if not context.config("enable_scaling"):
        logger.info("Skipping projecting households as scaling is disabled!")
        return
    data_path = context.config("data_path")
    c         = context.stage("data.constants")

    # Select year in the future to project to
    scaling_year = np.max([c.BASE_SCALING_YEAR, context.config("scaling_year")])

    df = pd.read_excel("%s/projections/households/su-d-01.03.03.01.xlsx" % data_path, sheet_name=0, header=1)

    # Fail fast if year not present
    year_cols = [c for c in df.columns if isinstance(c, (int, float))]
    if scaling_year not in year_cols:
        raise ValueError(f"Year {scaling_year} not found. Available years: {sorted(year_cols)}")

    # Filter out the Switzerland total row in column A
    df = df[df.iloc[:, 0] != "Schweiz"].copy()

    # Normalize names and map to canton_id
    df["canton_name"] = df.iloc[:, 0].map(_normalize_canton)
    df["canton_id"] = df["canton_name"].map(CANTON_TO_ID)

    # Keep only rows that successfully mapped (and warn if any didn’t)
    missing = df[df["canton_id"].isna()].iloc[:, 0].unique().tolist()
    if missing:
        logger.warning("Unmapped canton names (skipped): %s", missing)
    df = df[~df["canton_id"].isna()].copy()

    # Build the output
    out = df[["canton_id"]].copy()
    out["weight"] = df[scaling_year].astype("Int64")
    return out.sort_values("canton_id").reset_index(drop=True), scaling_year
