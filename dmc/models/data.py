"""Shared estimation features and conditional passenger history."""
import logging
import numpy as np
import pandas as pd
from dmc.models.specification import MODES

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, ignore_car_passenger: bool):
    """Code covariates and use uninterrupted starting sequences for history."""
    required = {'originPtQuality', 'origin_work', 'passenger_origin_home'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing current-model features {sorted(missing)}; rebuild DMC survey preparation and training data')
    df = prepare_sequences(df, ignore_car_passenger)
    modes = list(MODES)
    if ignore_car_passenger:
        df = df[[column for column in df.columns if 'car_passenger' not in column]]
        modes.remove('car_passenger')

    for mode in modes:
        df[f"{mode}_availability"] = df[f"{mode}_availability"].astype(int)

    integer_columns = [
        "driving_license", "destination_work", "origin_home", "destination_home",
        "destination_education", "destination_shopping", "destination_leisure",
        "destination_other", "homePtQuality", "destinationPtQuality",
        "low_income", "high_income", "originPtQuality", "origin_work", "passenger_origin_home", "has_car", "destination_zurich",
        "destination_geneva", "destination_basel", "destination_lausanne",
        "destination_luzern", "destination_bern",
    ]
    for column in integer_columns:
        df[column] = df[column].astype(int)

    df["mode"] = df["mode"].map(modes.index)
    df["person_weight"] = len(df) * df["person_weight"] / df["person_weight"].sum()
    if not ignore_car_passenger:
        df["is_car_passenger"] = df["is_car_passenger"].astype(int)

    municipalities = ["rural", "suburban", "urban", "urbancore"]
    for column in ["home_municipality", "origin_municipality", "destination_municipality"]:
        df[column] = df[column].map(municipalities.index).astype(int)
    df["urban_destination"] = (df["destination_municipality"] == 2).astype(int)
    df["urbancore_destination"] = (df["destination_municipality"] == 3).astype(int)
    df["region_0"] = (df["ms_region"] == 0).astype(int)
    df["region_1"] = (df["ms_region"] == 1).astype(int)
    df["region_2"] = (df["ms_region"] == 2).astype(int)
    for column in ["short_distance", "long_distance", "very_long_distance", "car_ownership_ratio"]:
        df[column] = df[column].astype(float)

    result = df.drop(columns=[
        "person_id", "trip_id", "home_municipality", "origin_municipality",
        "sp_region", "ms_region", "ovgk", "pt_egress_time_min", "income_class",
        "pt_access_time_min", "actual_parking_duration_min", "region_0", "purpose",
    ])
    return result.select_dtypes(include=['number', 'bool']).reset_index(drop=True), modes



def prepare_sequences(df, ignore_car_passenger):
    """Retain selected trip keys and construct history using the estimation rules."""
    if df.duplicated(['person_id', 'trip_id']).any():
        raise ValueError('Duplicate person/trip identifiers in DMC training data')
    modes = [mode for mode in MODES if not (ignore_car_passenger and mode == 'car_passenger')]
    result = df.loc[df['mode'].isin(modes)].sort_values(['person_id', 'trip_id'], kind='stable').copy()
    if not ignore_car_passenger:
        consecutive = result.trip_id.eq(result.groupby('person_id', sort=False).cumcount() + 1)
        prefix = consecutive.groupby(result.person_id, sort=False).cummin()
        logger.info('Passenger history keeps %d/%d uninterrupted starting-prefix trips', prefix.sum(), len(result))
        result = result.loc[prefix].copy()
        result['passenger_history'] = result['mode'].eq('car_passenger').groupby(result.person_id).shift(1, fill_value=False).astype(int)
    else:
        result['passenger_history'] = 0
    if result.empty:
        raise ValueError('No usable DMC training sequences')
    return result.reset_index(drop=True)
