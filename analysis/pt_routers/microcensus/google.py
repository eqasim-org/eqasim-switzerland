"""Read the already routed Google alternatives; no API requests are made."""
from pathlib import Path
import polars as pl

MAPPING = {'travelTime_pt': ('pt_total_travel_time_min', 60.),
           'inVehicleTime_pt': ('pt_in_vehicle_time_min', 60.),
           'inVehicleDistance_pt': ('pt_in_vehicle_distance_km', 1000.),
           'transferTime_pt': ('pt_transfer_time_min', 60.), 'transfers_pt': ('pt_transfers', 1.)}


def configure(context):
    context.stage('analysis.pt_routers.microcensus.inputs')
    context.stage('dmc.data.clean_routed_data')


def execute(context):

    # Convert a copy and join only the verified original survey identifiers.

    trips = context.stage('analysis.pt_routers.microcensus.inputs')
    source = context.stage('dmc.data.clean_routed_data')
    source = source if isinstance(source, pl.DataFrame) else pl.from_pandas(source)
    result = prepare_reference(trips, source)
    path = Path(context.path()) / 'routes.parquet'
    result.write_parquet(path)
    return {'routes': str(path), 'timing': {'scope': 'saved Google API results; original runtime unavailable'}}


def prepare_reference(trips, source):
    source = source.rename({'trip_id': 'google_trip_id'}).with_columns(pl.col('person_id', 'google_trip_id').cast(pl.Int64))
    if source.select('person_id', 'google_trip_id').is_duplicated().any():
        raise ValueError('Google reference has duplicate person/original-trip identifiers')
    columns = [pl.col(original).cast(pl.Float64) / divisor for original, (_, divisor) in MAPPING.items()]
    source = source.select('person_id', 'google_trip_id',
                           *[expression.alias(name) for expression, (name, _) in zip(columns, MAPPING.values())],
                           ((pl.col('accessTime_pt') + pl.col('egressTime_pt')) / 60).alias('pt_access_egress_time_min'),
                           pl.col('expectedModeUsed_pt').cast(pl.Boolean).alias('expected_pt_used'),
                           pl.col('types_pt').cast(pl.String).alias('google_vehicle_types'),
                           pl.lit(True).alias('reference_present'))
    return trips.select('trip_id', 'person_id', 'google_trip_id').join(
        source, on=['person_id', 'google_trip_id'], how='left', validate='m:1', maintain_order='left')
