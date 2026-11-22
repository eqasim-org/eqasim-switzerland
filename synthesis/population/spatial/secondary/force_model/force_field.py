"""Exact inverse-square attraction grids, adapted from eqasim-france.

Reference: eqasim-org/eqasim-france 7d219a070917919625c795e5b9fd0071184f6b67
(GPL-2.0). Row batches and a serial Numba kernel bound worker memory without
approximating or truncating the French model's long-range attraction.
"""
import numpy as np
from numba import njit


def configure(context):
    context.config("threads", volatile=True)
    attractors = context.config("secondary_activities.force_model.attractors", default="uniform")
    context.config("secondary_activities.force_model.cell_size", default=250.)
    context.config("secondary_activities.force_model.grid_margin", default=2000.)
    if attractors != "uniform":
        raise ValueError("secondary_activities.force_model.attractors currently supports 'uniform' only")
    context.stage("synthesis.population.spatial.secondary.force_model.attractors.uniform", alias="attractors")
    for key, positive in [("cell_size", True), ("grid_margin", False)]:
        value = context.config("secondary_activities.force_model." + key)
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
            raise ValueError("Force-model %s must be finite and %s" % (key, "positive" if positive else "nonnegative"))


def execute(context):

    # Construct the same projected grid and uniform attractors as the French model.

    attractors = context.stage("attractors").to_crs(2056)
    if attractors.empty:
        raise ValueError("The force model needs shopping, leisure or other destination attractors")
    points = np.column_stack((attractors.geometry.x, attractors.geometry.y, attractors.weight))
    if not np.isfinite(points).all() or (points[:, 2] < 0).any():
        raise ValueError("Force-model attractors need finite coordinates and nonnegative weights")
    cell = float(context.config("secondary_activities.force_model.cell_size"))
    margin = float(context.config("secondary_activities.force_model.grid_margin"))
    lower, upper = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    nx, ny = np.ceil((upper - lower + 2 * margin) / cell).astype(int) + 1
    grid = dict(xmin=float(lower[0] - margin), ymin=float(lower[1] - margin),
                cell_size=cell, nx=max(2, int(nx)), ny=max(2, int(ny)))

    # Sum every attractor exactly; split grid rows rather than duplicating full grids.

    processes = max(1, min(int(context.config("threads")), grid["nx"], 24))
    edges = np.linspace(0, grid["nx"], min(processes * 4, grid["nx"]) + 1, dtype=int)
    batches = list(zip(edges[:-1], edges[1:]))
    fields = {}
    for purpose in attractors.activity_type.unique():
        selected = points[attractors.activity_type.to_numpy() == purpose]
        field = np.empty((grid["nx"], grid["ny"], 2))
        with context.progress(label="Computing %s force field" % purpose, total=grid["nx"]) as progress:
            with context.parallel(processes=processes, data=dict(grid=grid, attractors=selected)) as parallel:
                for start, partial in parallel.imap_unordered(process, batches):
                    field[start:start + len(partial)] = partial
                    progress.update(len(partial))
        fields[purpose] = field

    # Return the upstream solver contract; fields are guidance, not facility eligibility.

    return grid, fields


def process(context, rows):
    grid = context.data("grid")
    return int(rows[0]), calculate_field(grid, context.data("attractors"), int(rows[0]), int(rows[1]))


def calculate_field(grid, attractors, start=0, end=None):
    end = grid["nx"] if end is None else end
    return _field_rows(float(grid["xmin"]), float(grid["ymin"]), float(grid["cell_size"]),
                       int(start), int(end), int(grid["ny"]), np.asarray(attractors, dtype=float))


@njit(cache=True)
def _field_rows(xmin, ymin, cell, start, end, ny, attractors):
    field = np.zeros((end - start, ny, 2))
    for i in range(start, end):
        x = xmin + i * cell
        for j in range(ny):
            y = ymin + j * cell
            fx, fy = 0., 0.
            for k in range(len(attractors)):
                dx, dy = attractors[k, 0] - x, attractors[k, 1] - y
                norm = np.sqrt(dx * dx + dy * dy)
                if norm < 1.:
                    norm = 125.  # Upstream near-zero regularization (metres).
                scale = attractors[k, 2] / norm ** 3
                fx += dx * scale
                fy += dy * scale
            field[i - start, j, 0] = fx
            field[i - start, j, 1] = fy
    return field
