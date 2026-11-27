"""Estimate relative facility-choice coefficients from weekday Microcensus trips."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.spatial import cKDTree

from . import subtypes


def configure(context):
    context.config("data_path")
    context.config(subtypes.OPTION, default=False)
    context.stage("data.statent.statent")
    context.stage("data.gtfs.cleaned")
    context.stage("data.spatial.swiss_border")


def execute(context):

    # Restrict the choice sets to accurately located domestic trips with usable radii.

    folder = Path(context.config("data_path")) / "microcensus"
    visits = read_visits(folder)
    length = np.hypot(visits.x - visits.ox, visits.y - visits.oy)
    visits = visits.loc[visits.S_QAL.eq(1) & visits.Z_QAL.eq(1) & visits.S_LND.eq(8100)
                        & length.between(250., 30000.) & ~holdout(visits.HHNR)]
    border = context.stage("data.spatial.swiss_border").to_crs(2056).geometry.union_all()
    stops = Path(context.path("data.gtfs.cleaned")) / "output/stops.txt"
    catalog = subtypes.facility_catalog(context.stage("data.statent.statent"), stops, border)
    used, features, allowed = examples(visits, catalog)

    # Estimate subtype preferences relative to the pooled choice model on the same input wave.

    if used.empty or not np.isfinite(used.WP).all() or not used.WP.gt(0).all():
        raise ValueError("The facility calibration requires eligible trips with positive survey weights")
    pooled = fit(features, allowed, used.WP.to_numpy())
    selected = subtypes.selected_purposes(context.config(subtypes.OPTION))
    coefficients = pd.DataFrame(0., index=sorted(catalog.sector.unique()), columns=subtypes.PURPOSES.values())
    for code, label in subtypes.PURPOSES.items():
        if label in selected:
            mask = used.wzweck1.eq(code)
            if not mask.any():
                raise ValueError(f"No eligible survey observations for {label}")
            coefficients[label] = fit(features[mask], allowed[mask], used.loc[mask, "WP"].to_numpy()) - pooled
    coefficients.rename_axis("sector").to_csv(Path(context.path()) / "facility_preferences.csv")
    used.groupby("wzweck1").agg(trips=("HHNR", "size"), respondents=("HHNR", "nunique")).to_csv(
        Path(context.path()) / "training_support.csv")
    return coefficients


def holdout(ids):
    """Reserve entire respondents across both facility estimation and geographic tuning."""
    return pd.util.hash_pandas_object(ids.astype("int64"), index=False).mod(5).eq(0)


def validate(context):
    folder = Path(context.config("data_path")) / "microcensus"
    metadata = {"facility_catalog_source": hashlib.sha256(Path(subtypes.__file__).read_bytes()).hexdigest()}
    for name in ("zielpersonen.csv", "wege.csv"):
        stat = (folder / name).stat()
        metadata[name] = [stat.st_size, stat.st_mtime_ns]
    return metadata


def read_visits(folder):
    options = dict(encoding='latin1')
    day = 'tag'
    r = pd.read_csv(folder / 'zielpersonen.csv', usecols=['HHNR', 'WP', 'alter', day], **options)
    r = r[r[day].between(1, 5) & r.alter.ge(6)]
    d = pd.read_csv(folder / 'wege.csv',
                    usecols=[
                        'HHNR', 'WEGNR', 'wzweck1', 'wzweck2', 'pseudo', 'S_X', 'S_Y', 'Z_X', 'Z_Y', 'S_QAL',
                        'Z_QAL', 'S_LND', 'Z_LND', 'Z_KANTON'
                    ],
                    **options)
    d = d[d.wzweck1.isin(subtypes.PURPOSES) & d.wzweck2.eq(1) & d.pseudo.eq(1) & d.Z_LND.eq(8100)]
    d = d.merge(r, on='HHNR', validate='many_to_one')
    project = Transformer.from_crs(4326, 2056, always_xy=True)
    d['ox'], d['oy'] = project.transform(d.S_X, d.S_Y)
    d['x'], d['y'] = project.transform(d.Z_X, d.Z_Y)
    return d


def examples(d, catalog):
    # Match at the reported destination and at 15 equal-angle alternatives on the same trip radius.
    origins = d[['ox', 'oy']].to_numpy()
    targets = d[['x', 'y']].to_numpy()
    delta = targets - origins
    angles = np.arctan2(delta[:, 1], delta[:, 0])[:, None] + 2 * np.pi * np.arange(16) / 16
    sites = origins[:, None, :] + np.linalg.norm(delta, axis=1)[:, None, None] * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1)
    possible = catalog[['x', 'y']].drop_duplicates().to_numpy()
    distance, idx = cKDTree(possible).query(sites)
    # An alternative needs a catalogued site within 100 m. Count coincident snapped sites once.
    allowed = distance <= 100.
    for j in range(1, 16):
        allowed[:, j] &= ~((idx[:, :j] == idx[:, j, None]) & allowed[:, :j]).any(axis=1)
    use = allowed[:, 0] & (allowed.sum(axis=1) >= 4)
    sites = possible[idx[use]]
    features = []
    for label, g in catalog.groupby('sector', sort=True):
        tree = cKDTree(g[['x', 'y']].to_numpy())
        features.append(np.exp(-tree.query(sites)[0] / 50.))
    return d.iloc[np.flatnonzero(use)].copy(), np.stack(features, axis=-1), allowed[use]


def fit(x, allowed, weights):
    total = weights.sum()

    def objective(b):
        score = np.where(allowed, x @ b, -np.inf)
        errors = softmax(score, axis=1)
        errors[:, 0] -= 1
        loss = np.average(logsumexp(score, axis=1) - score[:, 0], weights=weights)
        grad = np.einsum('i,ij,ijk->k', weights, errors, x) / total
        return loss + .005 * b.dot(b), grad + .01 * b

    result = minimize(objective, np.zeros(x.shape[-1]), jac=True, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(result.message)
    return result.x
