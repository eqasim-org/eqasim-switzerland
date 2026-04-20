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
    for chunk in context.progress(np.array_split(df, chunk_count), 
                                  total=chunk_count,
                                  label="Imputing {} density...".format(measure)):
        
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
    df_splits = np.array_split(df, chunk_count)

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
    if measure == "companies":
        return kd_tree.query_radius(coords, radius, count_only=True)

    if measure == "employees":
        if employee_weights is None:
            raise ValueError(
                "employee_weights are required to impute employees density. "
                "Pass the object returned by execute(context) as kd_tree argument."
            )

        indices = kd_tree.query_radius(coords, radius, count_only=False)
        return np.array([employee_weights[index].sum() for index in indices])

    raise ValueError("Unknown density measure '{}'. Use 'companies' or 'employees'.".format(measure))
