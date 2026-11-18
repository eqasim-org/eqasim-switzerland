"""Build airport municipality pools with overlap for similar access distances."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.airports._shared.common import file_digest
from data.airports.geneva.model import distances


def configure(context):
    context.stage('data.airports.geneva.model')
    context.config('airport_catchment_distance_ratio', default=1.3)


def execute(context):

    # Reuse exactly the municipal reference points and airport coordinates of gravity.

    model = context.stage('data.airports.geneva.model')

    # Permit the farther airport only within the configured ratio of the nearest.

    return assign_municipalities(model.municipalities, model.airports,
                                 context.config('airport_catchment_distance_ratio'))


def assign_municipalities(municipalities, airports, ratio_threshold=1.3):
    """Always allow the nearest airport; allow both if farther/nearer <= threshold."""
    if (isinstance(ratio_threshold, (bool, np.bool_))
            or not isinstance(ratio_threshold, (int, float))
            or not np.isfinite(ratio_threshold) or ratio_threshold < 1):
        raise ValueError('airport_catchment_distance_ratio must be finite and at least 1')
    columns = ['municipality_id', 'municipality_name', 'canton_id', 'x', 'y']
    frame = municipalities[columns].copy().sort_values('municipality_id').reset_index(drop=True)
    if frame.municipality_id.isna().any() or frame.municipality_id.duplicated().any():
        raise ValueError('Airport catchments require unique, non-missing municipality IDs')
    if frame.empty or not np.isfinite(frame[['x', 'y']].to_numpy(dtype=float)).all():
        raise ValueError('Airport catchments require finite municipal reference coordinates')
    codes = sorted(airports)
    if codes != ['GVA', 'ZRH']:
        raise ValueError('Airport catchments currently cover GVA and ZRH')
    distance = np.column_stack([distances(frame, airports[code]) for code in codes])
    if not np.isfinite(distance).all():
        raise ValueError('Airport catchment distances must be finite')
    for index, code in enumerate(codes):
        frame['distance_' + code.lower() + '_km'] = distance[:, index]
    nearest, farther = distance.min(axis=1), distance.max(axis=1)
    frame['nearest_airport'] = np.asarray(codes)[distance.argmin(axis=1)]
    frame['distance_ratio'] = np.divide(farther, nearest, out=np.full(len(frame), np.inf), where=nearest > 0)
    frame.loc[farther == 0, 'distance_ratio'] = 1.
    for index, code in enumerate(codes):
        frame['eligible_' + code.lower()] = distance[:, index] <= ratio_threshold * nearest
    frame['is_shared'] = frame.eligible_gva & frame.eligible_zrh
    settings = {'airports': airports, 'ratio_threshold': float(ratio_threshold)}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    digest.update(file_digest(Path(__file__)).encode())
    frame.attrs = {'signature': digest.hexdigest(), 'method': 'Allow an airport within distance_ratio_threshold times the nearest airport distance',
                   'distance_ratio_threshold': float(ratio_threshold), 'exact_ties': 'Both airports eligible; nearest_airport uses GVA for deterministic reporting',
                   'candidate_municipalities': {code: int(frame['eligible_' + code.lower()].sum()) for code in codes},
                   'shared_municipalities': int(frame.is_shared.sum()),
                   'scope': 'Gravity-model municipalities with residents or accommodation; inactive water/uninhabited polygons excluded'}
    return frame
