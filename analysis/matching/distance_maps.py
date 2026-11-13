"""Incoming diary-leg distances by home grid cell; no synthetic destinations inferred."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from synthesis.population.matching.V3.inputs import scenario_mask

PURPOSES = ("shop", "leisure", "other")
LABELS = {"shop": "Shopping", "leisure": "Leisure", "other": "Other"}
DISTANCE_PURPOSES = ("work", "education", "shop", "leisure", "other")
DISTANCE_LABELS = {**LABELS, "work": "Work", "education": "Education"}


def aggregate(people, summaries, size, dataset, scope, min_people):
    """Compress recipient/donor reuse before joining purpose-specific distance totals."""
    people = people.copy()
    x = pd.to_numeric(people.home_x, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    y = pd.to_numeric(people.home_y, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    quality = dict(dataset=dataset, scope=scope, residents=len(people),
                   valid_home_coordinates=int(valid.sum()), invalid_home_coordinates=int((~valid).sum()))
    people = people.loc[valid].copy()
    people["cell_x"] = np.floor(x[valid] / size) * size
    people["cell_y"] = np.floor(y[valid] / size) * size
    grouped = people.groupby(["cell_x", "cell_y", "mz_person_id"], sort=True).agg(
        resident_weight=("weight", "sum"), n_residents=("person_id", "size")).reset_index()
    results = []
    for purpose, summary in summaries.items():
        frame = grouped.join(summary, on="mz_person_id").fillna(0)
        frame["distance_weighted_km"] = frame.resident_weight * frame.distance_km
        frame["trip_weight"] = frame.resident_weight * frame.valid_trips
        frame["invalid_trip_weight"] = frame.resident_weight * frame.invalid_trips
        frame["n_contributors"] = frame.n_residents.where(frame.valid_trips.gt(0), 0)
        frame["distinct_donors"] = frame.valid_trips.gt(0).astype(int)
        frame["donor_weight_squared"] = frame.trip_weight ** 2
        cells = frame.groupby(["cell_x", "cell_y"], as_index=False).agg(
            distance_weighted_km=("distance_weighted_km", "sum"), trip_weight=("trip_weight", "sum"),
            invalid_trip_weight=("invalid_trip_weight", "sum"), n_residents=("n_residents", "sum"),
            n_contributors=("n_contributors", "sum"), distinct_donors=("distinct_donors", "sum"),
            donor_weight_squared=("donor_weight_squared", "sum"))
        cells["mean_distance_km"] = cells.distance_weighted_km / cells.trip_weight.where(cells.trip_weight.gt(0))
        cells["donor_effective_n"] = (cells.trip_weight ** 2 / cells.donor_weight_squared.where(cells.donor_weight_squared.gt(0))).fillna(0)
        cells["supported"] = cells.n_contributors.ge(min_people) & cells.trip_weight.gt(0)
        cells["dataset"], cells["scope"], cells["purpose"] = dataset, scope, purpose
        results.append(cells)
    return pd.concat(results, ignore_index=True), quality


def prepare(output, mappings, roster, persons, trips, const, day, *, cantons=None, grid_size=5000., min_people=5,
            canton_id=None, region="Switzerland"):
    """Save map statistics and outlines so rendering never needs to rematch anyone."""
    size = float(grid_size)
    if not np.isfinite(size) or size <= 0 or int(min_people) != min_people or min_people < 1:
        raise ValueError("Distance maps require positive grid size and minimum contributor count")
    for frame in (roster, persons):
        if not {"home_x", "home_y"}.issubset(frame):
            raise ValueError("Distance maps require EPSG:2056 home_x/home_y on population and survey")
    if "crowfly_distance" not in trips:
        raise ValueError("Distance maps require the survey trip crowfly_distance in metres")
    if trips.duplicated(["person_id", "trip_id"]).any():
        raise ValueError("Duplicate survey trip keys in distance maps")
    purpose = trips.purpose.astype(str).str.replace(r"_secondary$|_main$", "", regex=True).replace({"shopping": "shop"})
    distance = pd.to_numeric(trips.crowfly_distance, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(distance) & (distance >= 0)
    legs = pd.DataFrame({"mz_person_id": trips.person_id, "purpose": purpose,
        "distance_km": np.where(valid, distance / 1000., 0), "valid_trips": valid.astype(int),
        "invalid_trips": (~valid).astype(int)})
    summaries = {p: legs.loc[legs.purpose.eq(p)].groupby("mz_person_id")[["distance_km", "valid_trips", "invalid_trips"]].sum() for p in PURPOSES}
    eligible = roster.age.ge(const.MZ_AGE_THRESHOLD) if const.census == "statpop" else roster.age_class.gt(0)
    if canton_id is not None:
        eligible &= pd.to_numeric(roster.canton_id, errors="coerce").eq(canton_id)
    population = roster.loc[eligible, ["person_id", "home_x", "home_y"]].copy()
    if population.person_id.duplicated().any() or persons.person_id.duplicated().any():
        raise ValueError("Distance maps require unique population and survey identifiers")
    survey_weights = pd.to_numeric(persons.person_weight, errors="coerce")
    reference = scenario_mask(persons, day) & ~coerce_boolean_series(persons.is_crossing_the_border, name="is_crossing_the_border")
    reference &= persons.age.ge(const.MZ_AGE_THRESHOLD) & np.isfinite(survey_weights) & survey_weights.gt(0)
    if canton_id is not None:
        reference &= pd.to_numeric(persons.canton_id, errors="coerce").eq(canton_id)
    survey = persons.loc[reference, ["person_id", "home_x", "home_y"]].copy()
    survey["mz_person_id"] = survey.person_id
    survey["weight"] = survey_weights.loc[reference].to_numpy()
    common = pd.Index(population.person_id)
    for mapping in mappings.values():
        if mapping.person_id.duplicated().any():
            raise ValueError("Distance maps require unique mapping recipients")
        common = common.intersection(mapping.person_id, sort=False)
    tables, quality, distributions = [], [], []
    for scope in ("full", "common"):
        residents = population if scope == "full" else population.loc[population.person_id.isin(common)]
        datasets = {"Survey": survey}
        for name, mapping in mappings.items():
            frame = residents.merge(mapping[["person_id", "mz_person_id"]], on="person_id", validate="one_to_one")
            frame["weight"] = 1.
            datasets[name] = frame
        for name, frame in datasets.items():
            table, audit = aggregate(frame, summaries, size, name, scope, int(min_people))
            tables.append(table)
            quality.append(audit)
            distributions.extend(distance_distributions(frame, legs, name, scope))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(output / "distance_map_cells.csv", index=False)
    pd.DataFrame(quality).to_csv(output / "distance_map_coverage.csv", index=False)
    pd.DataFrame(distributions).to_csv(output / "distance_distributions.csv", index=False)
    # Store simplified outlines in the same metric CRS, independent of GIS tools during re-rendering.
    outlines = []
    if cantons is not None and "geometry" in cantons:
        codes = pd.to_numeric(cantons.canton_id)
        selected = codes.between(1, 26) if canton_id is None else codes.eq(canton_id)
        swiss = cantons.loc[selected].to_crs("EPSG:2056").copy()
        swiss.geometry = swiss.geometry.simplify(200, preserve_topology=True)
        outlines = [feature["geometry"] for feature in json.loads(swiss.to_json())["features"] if feature["geometry"]]
    metadata = {"crs": "EPSG:2056", "grid_size_m": size, "min_contributing_people": int(min_people),
        "region": region, "canton_id": canton_id, "algorithms": list(mappings),
        "distance": "Survey straight-line origin-to-destination leg distance, in kilometres, grouped by destination purpose.",
        "geography": "Cells locate recipient homes for V1/V3 and respondent homes for Survey. Synthetic activity destinations are not inferred.",
        "mean": "Trip-weighted arithmetic mean: sum of valid incoming leg distances divided by valid incoming trip count; survey person weights apply to every reported leg.",
        "support": "Cells need the minimum number of contributing people, before survey weighting. Non-travellers and missing/negative distances do not contribute zero lengths. Distinct donor counts and donor contribution ESS are exported.",
        "outlines": outlines}
    (output / "distance_map_manifest.json").write_text(json.dumps(metadata))


def distance_distributions(people, legs, dataset, scope):
    """Trip-weighted distributions include residents with unavailable map coordinates."""
    weights = people.groupby("mz_person_id").weight.sum()
    event_weights = legs.mz_person_id.map(weights).fillna(0).to_numpy()
    rows = []
    for purpose in DISTANCE_PURPOSES:
        selected = legs.purpose.eq(purpose).to_numpy()
        valid = selected & legs.valid_trips.eq(1).to_numpy()
        values = legs.distance_km.to_numpy()[valid]
        w = event_weights[valid]
        # Shared bins depend on all survey legs, not the selected matching method.
        maximum = max(200., float(values.max()) + 1.e-6) if len(values) else 200.
        edges = np.r_[0., np.geomspace(.1, maximum, 61)]
        hist, _ = np.histogram(values, bins=edges, weights=w)
        mass = float(w.sum())
        probability = hist / mass if mass else np.zeros_like(hist)
        cumulative = probability.cumsum()
        invalid = float(event_weights[selected & ~valid].sum())
        zero = float(w[values == 0].sum() / mass) if mass else 0.
        for k, p in enumerate(probability):
            rows.append(dict(scope=scope, dataset=dataset, purpose=purpose, left_km=edges[k], right_km=edges[k + 1],
                probability=p, density=p / (edges[k + 1] - edges[k]), cdf=cumulative[k],
                valid_trip_weight=mass, invalid_trip_weight=invalid, zero_distance_probability=zero))
    return rows


def _rings(geometry):
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon


def render(output, save):
    """Draw three matched colour-scale maps and a paired difference, from saved cells."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize, TwoSlopeNorm

    output = Path(output)
    info = json.loads((output / "distance_map_manifest.json").read_text())
    cells = pd.read_csv(output / "distance_map_cells.csv")
    from .interactive_maps import write as write_interactive
    write_interactive(output, cells, info)
    size = info["grid_size_m"]
    rings = []
    for geom in info["outlines"]:
        for ring in _rings(geom):
            vertices = np.asarray(ring, dtype=float)
            if vertices.size == 0:
                continue
            if vertices.ndim != 2 or vertices.shape[1] < 2:
                raise ValueError("Canton outline rings must contain X/Y coordinate pairs")
            # swissBOUNDARIES3D can include elevation; maps and extents use X/Y.
            # Normalize during reading so existing saved 3D outlines also work.
            rings.append(vertices[:, :2])
    if rings:
        points = np.concatenate(rings)
    elif len(cells):
        points = np.vstack([cells[["cell_x", "cell_y"]].to_numpy(), cells[["cell_x", "cell_y"]].to_numpy() + size])
    else:
        points = np.array([[0., 0.], [size, size]])
    low, high = points.min(axis=0) - size, points.max(axis=0) + size

    def draw(ax, rows, values, norm, cmap, title):
        if rings:
            ax.add_collection(PolyCollection(rings, facecolors="#e5e5e5", edgecolors="none", zorder=0))
        vertices = [np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]]) for x, y in rows[["cell_x", "cell_y"]].itertuples(index=False, name=None)]
        collection = PolyCollection(vertices, array=np.asarray(values, dtype=float), cmap=cmap, norm=norm, edgecolors="none", rasterized=True)
        ax.add_collection(collection)
        for ring in rings:
            ax.plot(ring[:, 0], ring[:, 1], color="#555555", linewidth=.35)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), title=title)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if rows.empty:
            ax.text(.5, .5, "No supported cells", transform=ax.transAxes, ha="center", fontsize=9)
        return collection

    for scope in ("full", "common"):
        for purpose in PURPOSES:
            frame = cells.loc[cells.scope.eq(scope) & cells.purpose.eq(purpose) & cells.supported].copy()
            values = frame.mean_distance_km.dropna().to_numpy()
            vmax = max(float(np.quantile(values, .98)), .1) if len(values) else 1.
            norm = Normalize(0, vmax)
            algorithms = info.get("algorithms", [name for name in cells.dataset.unique() if name != "Survey"])
            datasets = ["Survey", *sorted(algorithms)]
            comparisons = [(name, "V1") for name in sorted(algorithms) if name != "V1"]
            panels = len(datasets) + len(comparisons)
            if panels <= 4:
                fig, axes = plt.subplots(1, panels, figsize=(17, 5.5), layout="constrained")
            else:
                fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
            axes = np.asarray(axes).reshape(-1)
            for ax, dataset in zip(axes, datasets):
                rows = frame.loc[frame.dataset.eq(dataset)]
                collection = draw(ax, rows, rows.mean_distance_km, norm, "viridis", dataset)
            fig.colorbar(collection, ax=list(axes[:len(datasets)]), orientation="horizontal", fraction=.06, pad=.04,
                         label="Mean incoming trip distance (km)", extend="max")
            for position, (method, baseline) in enumerate(comparisons, start=len(datasets)):
                paired = frame.loc[frame.dataset.isin([baseline, method])].pivot(index=["cell_x", "cell_y"], columns="dataset", values="mean_distance_km")
                paired = paired.reindex(columns=[baseline, method]).dropna()
                paired["difference"] = paired[method] - paired[baseline]
                bound = max(float(np.quantile(np.abs(paired.difference), .98)), .1) if len(paired) else 1.
                rows = paired.reset_index()
                collection = draw(axes[position], rows, rows.difference, TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), "RdBu_r", f"{method} minus {baseline}")
                fig.colorbar(collection, ax=axes[position], orientation="horizontal", fraction=.06, pad=.04,
                             label="Difference in mean distance (km)", extend="both")
            for ax in axes[panels:]:
                ax.axis("off")
            fig.suptitle(f"{info.get('region', 'Switzerland')} — {LABELS[purpose]}: distance from the previous activity — {scope} population")
            caption = (f"Each square is {size / 1000:g} km wide and locates a person's home, not the trip origin or activity destination. "
                f"Colour shows the trip-weighted mean straight-line distance into {LABELS[purpose].lower()} from the preceding activity. "
                f"Cells with fewer than {info['min_contributing_people']} contributing people are uncoloured (grey within the mapped region); this does not mean zero distance. "
                "Survey uses person weights; Population methods use the assigned survey legs and synthetic recipients' homes. Non-travellers do not contribute artificial zero distances. "
                "All absolute-distance panels share a colour scale. Colour limits use the 98th percentile of supported cell values; underlying CSV means are not clipped. "
                "The difference is shown only where both methods have support. Positive differences mean longer distances than V1. Full uses each retained population; common uses the same recipients in the compared methods. "
                "These maps describe assigned diary lengths before synthetic activity locations are assigned. They are not maps of final simulated routes.")
            save(fig, f"{scope}_distance_{purpose}", caption)


def render_distributions(output, save):
    """Render density and binned cumulative-distance curves from saved histograms."""
    import matplotlib.pyplot as plt
    output = Path(output)
    table = pd.read_csv(output / "distance_distributions.csv")
    info = json.loads((output / "distance_map_manifest.json").read_text())
    colors = {"Survey": "#333333", "V1": "#dd8452", "V2": "#55a868", "V3": "#4c72b0"}
    datasets = ["Survey", *sorted(set(table.dataset) - {"Survey"})]
    for scope in ("full", "common"):
        for metric in ("density", "cdf"):
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            title = "Incoming-distance density" if metric == "density" else "Cumulative incoming-distance distribution"
            fig.suptitle(f"{info.get('region', 'Switzerland')} — {title} — {scope} population")
            for ax, purpose in zip(axes.flat, DISTANCE_PURPOSES):
                available = False
                for dataset in datasets:
                    color = colors[dataset]
                    rows = table.loc[table.scope.eq(scope) & table.purpose.eq(purpose) & table.dataset.eq(dataset)]
                    if rows.empty or rows.valid_trip_weight.iloc[0] <= 0:
                        ax.plot([], [], color=color, label=f"{dataset} (no valid trips)")
                        continue
                    available = True
                    if metric == "density":
                        edges = np.r_[rows.left_km.iloc[0], rows.right_km.to_numpy()]
                        ax.stairs(rows.density, edges, color=color, label=dataset)
                    else:
                        ax.step(np.r_[0, rows.right_km], np.r_[rows.zero_distance_probability.iloc[0], rows.cdf],
                            where="post", color=color, label=dataset)
                ax.set_xscale("symlog", linthresh=1.)
                ax.set(xlabel="Incoming straight-line distance (km)",
                    ylabel="Density (per km)" if metric == "density" else "Cumulative trip fraction",
                    title=DISTANCE_LABELS[purpose], xlim=(0, None))
                active = table.loc[table.scope.eq(scope) & table.purpose.eq(purpose) & table.probability.gt(0)]
                ax.set_xlim(0, float(active.right_km.max()) if len(active) else 1.)
                if metric == "cdf":
                    ax.set_ylim(0, 1.02)
                if not available:
                    ax.text(.5, .5, "No valid trips", transform=ax.transAxes, ha="center")
            axes.flat[0].legend(fontsize=8)
            axes.flat[-1].axis("off")
            save(fig, f"{scope}_distance_{metric}",
                "Distances run from the preceding activity to the activity named in each panel. "
                "The curves include all valid incoming survey legs assigned to residents of this region, even when home coordinates are missing. "
                "Survey legs use respondent weights; population legs use donor reuse counts. Non-travellers do not add zero-length trips; genuinely zero-length legs remain included. "
                "Invalid/negative distances are excluded and counted in the CSV. The horizontal axis is linear up to 1 km and logarithmic thereafter. "
                "Density uses common unequal-width bins and integrates to one when trips are available. Cumulative curves show the fraction at or below each bin's upper edge; higher curves indicate shorter trips. "
                "The horizontal range stops at the last occupied shared bin; no observations are clipped. Full and common use the same population definitions as the other report figures.")
