"""Offline cell data with an optional swisstopo WMTS background."""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


def write(output, cells, info):
    output = Path(output)
    values = pd.to_numeric(cells.mean_distance_km, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    cells = cells.loc[cells.supported.fillna(False).astype(bool) & np.isfinite(values)].copy()
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)
    size = info["grid_size_m"]
    squares, indices = [], {}
    for x, y in cells[["cell_x", "cell_y"]].drop_duplicates().itertuples(index=False, name=None):
        indices[(x, y)] = len(squares)
        lon, lat = transformer.transform([x, x + size, x + size, x], [y, y, y + size, y + size])
        squares.append([[float(a), float(b)] for a, b in zip(lat, lon)])
    records = [[indices[(r.cell_x, r.cell_y)], r.dataset, r.purpose, r.scope, float(r.mean_distance_km), int(r.n_contributors)] for r in cells.itertuples()]
    outlines = []
    for geom in info["outlines"]:
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"] if geom["type"] == "MultiPolygon" else []
        for polygon in polygons:
            for ring in polygon:
                points = np.asarray(ring, dtype=float)
                if not len(points):
                    continue
                lon, lat = transformer.transform(points[:, 0], points[:, 1])
                outlines.append([[float(a), float(b)] for a, b in zip(lat, lon)])
    data = json.dumps(dict(squares=squares, rows=records, outlines=outlines,
                          methods=info.get("algorithms", sorted(set(cells.dataset) - {"Survey"}))), allow_nan=False).replace("</", "<\\/")
    summary = cells.groupby(["scope", "dataset", "purpose"]).agg(supported_cells=("mean_distance_km", "size"),
        min_cell_mean_km=("mean_distance_km", "min"), max_cell_mean_km=("mean_distance_km", "max")).reset_index()
    template = Path(__file__).with_name("interactive_distances.html.template").read_text()
    page = template.replace("__TITLE__", html.escape(info.get("region", "Switzerland")))
    page = page.replace("__DATA__", data).replace("__TABLE__", summary.to_html(index=False, float_format=lambda x: f"{x:.2f}"))
    (output / "interactive_distances.html").write_text(page)
