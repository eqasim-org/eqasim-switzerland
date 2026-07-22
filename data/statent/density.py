import numpy as np
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.statent.statent")

def execute(context):
    df_statpop = context.stage("data.statent.statent")[["x","y","number_employees"]].reset_index(drop=True)
    density_coordinates = np.vstack([df_statpop["x"], df_statpop["y"]]).T
    kd_tree = KDTree(density_coordinates)
    employee_weights = df_statpop["number_employees"].to_numpy()

    return {
        "kd_tree": kd_tree,
        "employee_weights": employee_weights
    }

def impute(context, df, x="x", y="y", radius= 500, point_type="", chunk_size=1e5,
           measure="companies", output_column=None):
    measure = _normalize_measure(measure)
    kd_tree, employee_weights = _unpack_density_data(context)
    output_column = _get_output_column(measure, output_column)
    logger.info("Imputing %s density within %d m of %d %s coordinates...", measure, radius, len(df), point_type)
    counts = []
    chunk_count = max(1, int(np.ceil(len(df) / chunk_size)))
    indices = np.array_split(np.arange(len(df)), chunk_count)
    for idx in context.progress(indices, 
                                  total=chunk_count,
                                  label="Imputing {} density...".format(measure)):
        chunk = df.iloc[idx]
        coordinates = np.vstack([chunk[x], chunk[y]]).T
        counts.extend(_query_density(kd_tree, coordinates, radius, measure, employee_weights))
    
    df[output_column] = counts

    return df

def impute_parallel(context, df, x="x", y="y", radius=500, point_type="", chunk_size=1e4,
                    n_jobs=10, measure="companies", output_column=None):
    measure = _normalize_measure(measure)
    kd_tree, employee_weights = _unpack_density_data(context)
    output_column = _get_output_column(measure, output_column)

    total_points = len(df)
    logger.info("Imputing %s density within %d m of %d %s coordinates...", measure, radius, total_points, point_type)

    # Split DataFrame into roughly equal chunks
    chunk_count = max(1, int(np.ceil(total_points / chunk_size)))
    indices = np.array_split(np.arange(total_points), chunk_count)
    df_splits = [df.iloc[idx] for idx in indices]

    def process_chunk(chunk):
        coords = np.vstack([chunk[x], chunk[y]]).T
        return _query_density(kd_tree, coords, radius, measure, employee_weights)

    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk)
        for chunk in context.progress(df_splits, total=chunk_count, label="Imputing {} density...".format(measure))
    )

    # Flatten list of arrays
    counts = np.concatenate(results)
    df[output_column] = counts
    return df



############# HELP FUNCTIONS #############

def _normalize_measure(measure):
    if measure is None:
        return "companies"
    return str(measure).strip().lower()


def _unpack_density_data(context):
    kd_tree = context.stage("data.statent.density")
    if isinstance(kd_tree, dict):
        return kd_tree["kd_tree"], kd_tree.get("employee_weights")
    return kd_tree, None


def _get_output_column(measure, output_column):
    if output_column is not None:
        return output_column
    return "companies_density" if measure == "companies" else "employees_density"


def _query_density(kd_tree, coords, radius, measure, employee_weights):
    coords = np.array(coords)
    nan_mask = np.isnan(coords).any(axis=1)
    valid_coords = coords[~nan_mask]

    if measure == "companies":
        result = np.zeros(len(coords), dtype=int)
        if len(valid_coords) > 0:
            result[~nan_mask] = kd_tree.query_radius(valid_coords, radius, count_only=True)
        return result

    if measure == "employees":
        if employee_weights is None:
            raise ValueError(
                "employee_weights are required to impute employees density. "
                "Pass the object returned by execute(context) as kd_tree argument."
            )
        result = np.zeros(len(coords), dtype=float)
        if len(valid_coords) > 0:
            indices = kd_tree.query_radius(valid_coords, radius, count_only=False)
            result[~nan_mask] = np.array([employee_weights[index].sum() for index in indices])
        return result

    raise ValueError("Unknown density measure '{}'. Use 'companies' or 'employees'.".format(measure))
