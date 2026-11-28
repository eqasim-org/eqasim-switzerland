"""Weight force-field attractors by surrounding STATENT employment (LV95)."""
import numpy as np
import shapely
from scipy.ndimage import gaussian_filter, map_coordinates

PREFIX = "secondary_activities.force_model."
PURPOSES = ("shop", "leisure", "other")
CELL_M = 250.
BANDWIDTH_M = 2000.


def configure(context):
    context.stage("synthesis.population.spatial.secondary.force_model.attractors.uniform")
    context.stage("data.statent.statent")
    context.stage("data.spatial.cantons")
    ids = context.config(PREFIX + "canton_ids", default=list(range(1, 27)))
    if not isinstance(ids, (list, tuple)) or not ids or any(
            type(code) is not int or not 1 <= code <= 26 for code in ids):
        raise ValueError("force_model.canton_ids requires Swiss canton numbers 1 to 26")
    for purpose in PURPOSES:
        beta = context.config(PREFIX + "density_beta." + purpose, default=0.)
        if isinstance(beta, bool) or not isinstance(beta, (float, int)) or not np.isfinite(beta):
            raise ValueError(f"Density exponent for {purpose} must be finite")


def execute(context):

    # Start from the same catalogue, ordering and unit weights as the baseline.

    uniform = context.stage("synthesis.population.spatial.secondary.force_model.attractors.uniform")
    betas = {p: context.config(PREFIX + "density_beta." + p) for p in PURPOSES}
    if not any(betas.values()):
        return uniform

    # Limit direct reweighting to configured Swiss cantons and activity purposes.

    cantons = context.stage("data.spatial.cantons").to_crs(2056)
    region = cantons[cantons.canton_id.isin(context.config(PREFIX + "canton_ids"))].geometry.union_all()
    shapely.prepare(region)
    active = uniform.activity_type.isin([p for p in PURPOSES if betas[p]])
    selected = uniform.loc[active]
    inside = shapely.intersects_xy(region, selected.geometry.x, selected.geometry.y)
    selected = selected.loc[inside]
    weighted = uniform.copy()

    # Preserve total attractor mass for each affected purpose inside the region.

    density = employment_density(context.stage("data.statent.statent"), selected.geometry.x, selected.geometry.y)
    for purpose, beta in betas.items():
        mask = selected.activity_type.eq(purpose)
        if mask.any():
            weighted.loc[selected.index[mask], "weight"] = relative_weights(density[mask], beta)
    return weighted


def employment_density(establishments, x, y):
    """Gaussian-smoothed jobs per square kilometre, sampled at destination coordinates."""
    coords = establishments[["x", "y"]].to_numpy(dtype=float)
    jobs = establishments["number_employees"].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or not np.isfinite(jobs).all() or (jobs < 0).any():
        raise ValueError("STATENT locations and employment must be finite with nonnegative employment")
    if not len(jobs):
        raise ValueError("Employment density requires STATENT locations")
    # Align to a fixed national raster and pad by the truncated Gaussian support.
    cells = np.floor(coords / CELL_M).astype(int)
    padding = int(np.ceil(4 * BANDWIDTH_M / CELL_M)) + 1
    lower = cells.min(axis=0) - padding
    shape = cells.max(axis=0) - lower + padding + 1
    counts = np.zeros(shape)
    np.add.at(counts, tuple((cells - lower).T), jobs)
    density = gaussian_filter(counts, BANDWIDTH_M / CELL_M, mode="constant") * (1000 / CELL_M)**2
    positions = np.column_stack((x, y)) / CELL_M - lower - 0.5
    return map_coordinates(density, positions.T, order=1, mode="constant")


def relative_weights(density, beta):
    log_density = np.log1p(density)
    reference = log_density.max() if beta >= 0 else log_density.min()
    with np.errstate(over="ignore"):
        weight = np.exp(beta * (log_density - reference))
    return weight / weight.mean()
