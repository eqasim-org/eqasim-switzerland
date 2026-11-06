"""Metrics and figures for paired secondary-location assignment results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from numba import njit

from synthesis.population.spatial.secondary.opening_hours import encode_schedules, eligible_candidates


CITIES = {"Lausanne": (6.6323, 46.5197), "Geneva": (6.1432, 46.2044), "Zurich": (8.5417, 47.3769)}
PROJECT = Transformer.from_crs(4326, 2056, always_xy=True)
CENTRES = {name: PROJECT.transform(*location) for name, location in CITIES.items()}
COLORS = {"without_hours": "#2878b5", "with_hours": "#e48726"}


@njit(cache=True)
def check_stays(schedules, indices, starts, ends, tolerance_seconds=600.):
    result = np.empty(len(indices), dtype=np.bool_)
    for index in range(len(indices)):
        result[index] = eligible_candidates(schedules, indices[index:index+1], starts[index], ends[index], tolerance_seconds)[0]
    return result


def activity_metrics(trips, primary, secondary, hours, radius, tolerance_seconds=600.):
    keys = ["person_id", "activity_index"]
    activity = trips[["person_id", "trip_index", "preceding_purpose", "departure_time"]].rename(
        columns={"trip_index": "activity_index", "preceding_purpose": "purpose", "departure_time": "end"})
    last = trips.groupby("person_id", sort=False).tail(1)
    final = pd.DataFrame(dict(person_id=last.person_id, activity_index=last.trip_index + 1,
                             purpose=last.following_purpose, end=np.maximum(108000., last.arrival_time + 60.)))
    activity = pd.concat([activity, final], ignore_index=True)
    arrivals = trips[["person_id", "trip_index", "arrival_time"]].rename(
        columns={"trip_index": "activity_index", "arrival_time": "start"})
    arrivals["activity_index"] += 1
    activity = activity.merge(arrivals, on=keys, how="left", validate="one_to_one")
    activity["start"] = activity.start.fillna(0.)
    assigned = secondary.rename(columns={"trip_index": "activity_index"})
    activity = activity.merge(assigned[keys + ["destination_id", "geometry"]], on=keys, how="left", validate="one_to_one")
    geometry = gpd.GeoSeries(activity.geometry, crs=2056)
    activity["x"], activity["y"] = geometry.x, geometry.y
    anchors = primary.set_index("person_id")
    for purpose in ("home", "work", "education"):
        selection = activity.purpose.eq(purpose)
        geometry = gpd.GeoSeries(anchors[purpose], crs=2056)
        activity.loc[selection, "x"] = activity.loc[selection, "person_id"].map(geometry.x)
        activity.loc[selection, "y"] = activity.loc[selection, "person_id"].map(geometry.y)
    activity = activity.sort_values(keys)
    previous = activity.groupby("person_id", sort=False)[["x", "y"]].shift()
    activity["incoming_km"] = np.hypot(activity.x - previous.x, activity.y - previous.y) / 1000.
    home = gpd.GeoSeries(anchors.home, crs=2056)
    activity["home_x"] = activity.person_id.map(home.x)
    activity["home_y"] = activity.person_id.map(home.y)
    activity = activity.loc[activity.purpose.isin(["shop", "leisure"])].copy()
    if activity.destination_id.isna().any() or activity[["x", "y"]].isna().any().any():
        raise ValueError("Missing RDA assignments for shopping/leisure activities")
    activity["known_hours"], activity["open_for_stay"] = False, False
    catalogue = hours.set_index("destination_id")
    for purpose in ("shop", "leisure"):
        selected = activity.purpose.eq(purpose)
        schedules = catalogue[purpose + "_open_intervals"]
        positions = catalogue.index.get_indexer(activity.loc[selected, "destination_id"])
        if (positions < 0).any():
            raise ValueError("Assigned destination absent from opening-hours catalogue")
        encoded = encode_schedules(schedules)
        activity.loc[selected, "known_hours"] = schedules.iloc[positions].notna().to_numpy()
        activity.loc[selected, "open_for_stay"] = check_stays(encoded, positions,
            activity.loc[selected, "start"].to_numpy(), activity.loc[selected, "end"].to_numpy(), tolerance_seconds)
    for city, (x, y) in CENTRES.items():
        activity[city] = np.hypot(activity.home_x - x, activity.home_y - y) <= radius
    return activity.drop(columns="geometry").set_index(keys).sort_index()


def generate(variants, convergence, radius, output, tolerance_seconds=600.):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    before, after = variants["without_hours"], variants["with_hours"]
    if not before.index.equals(after.index) or not before.purpose.equals(after.purpose):
        raise ValueError("RDA variants do not contain identical activity keys/purposes")
    paired = before[["purpose", "destination_id", "x", "y", "incoming_km"]].join(
        after[["destination_id", "x", "y", "incoming_km"]], lsuffix="_without", rsuffix="_with")
    paired["changed_destination"] = paired.destination_id_without.ne(paired.destination_id_with)
    paired["relocation_km"] = np.hypot(paired.x_with - paired.x_without, paired.y_with - paired.y_without) / 1000.
    paired["incoming_change_km"] = paired.incoming_km_with - paired.incoming_km_without
    paired.reset_index().to_parquet(output / "paired_changes.parquet", index=False)
    rows = []
    for variant, data in variants.items():
        for region in ("All", *CENTRES):
            for purpose in ("shop", "leisure"):
                selection = data.purpose.eq(purpose)
                if region != "All":
                    selection &= data[region]
                group = data.loc[selection]
                known = group.loc[group.known_hours]
                rows.append(dict(variant=variant, region=region, purpose=purpose, activities=len(group),
                    mean_incoming_km=group.incoming_km.mean(), median_incoming_km=group.incoming_km.median(),
                    p90_incoming_km=group.incoming_km.quantile(.9), missing_incoming_distances=int(group.incoming_km.isna().sum()),
                    known_hours_share=group.known_hours.mean(), known_hours_compliance=known.open_for_stay.mean(),
                    changed_destination_share=paired.loc[selection, "changed_destination"].mean(),
                    median_relocation_km=paired.loc[selection, "relocation_km"].median()))
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(convergence).to_csv(output / "convergence.csv", index=False)
    plot_distributions(variants, paired, output)
    plot_city_changes(variants, radius, output)
    (output / "metadata.json").write_text(json.dumps({
        "city_radius_m": radius, "city_scope": "Residents whose home is within the radius of the city centre; circles are not administrative boundaries",
        "opening_hours_tolerance_seconds": tolerance_seconds,
        "distances": "Euclidean incoming leg distances, including changed preceding secondary destinations",
        "maps": "All assigned destinations inside city circles, regardless of home location; 500m cells",
        "comparison": "Same population, trip times, primary anchors and initial worker seeds. Changed draws/retries can propagate to other assignments.",
        "unknown_hours": "Unrestricted in RDA; compliance is reported separately for known schedules.",
        "centres_wgs84": CITIES,
    }, indent=2))
    return {"output": str(output), "summary": str(output / "summary.csv"),
            "figures": [str(path) for path in sorted(output.glob("*.png"))]}


def plot_distributions(variants, paired, output):
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), constrained_layout=True)
    fig2, axes2 = plt.subplots(4, 2, figsize=(13, 15), constrained_layout=True)
    for row, region in enumerate(("All", *CENTRES)):
        for column, purpose in enumerate(("shop", "leisure")):
            for variant, data in variants.items():
                selection = data.purpose.eq(purpose)
                if region != "All":
                    selection &= data[region]
                distances = np.sort(data.loc[selection, "incoming_km"].dropna().to_numpy())
                # ECDF covers the complete distribution; histograms show 0–50km.
                if len(distances):
                    step = max(1, len(distances) // 5000)
                    indices = np.unique(np.r_[np.arange(0, len(distances), step), len(distances)-1])
                    axes[row, column].plot(distances[indices], (indices+1)/len(distances),
                                            label=variant.replace('_', ' '), color=COLORS[variant])
                    weights = np.full(len(distances), 100./len(distances))
                    axes2[row, column].hist(distances, bins=np.arange(0,51,1), weights=weights,
                        histtype="step", color=COLORS[variant], label=variant.replace('_', ' '))
            for ax in (axes[row,column], axes2[row,column]):
                ax.set(title=f"{region} — {purpose}", xlabel="Incoming distance (km)")
                ax.grid(alpha=.2)
            axes[row,column].set(xlim=(0,50), ylim=(0,1), ylabel="Cumulative share (tail above 50km omitted from view)")
            axes2[row,column].set(ylabel="Activities (%)")
    axes[0,0].legend();axes2[0,0].legend()
    fig.savefig(output/'incoming_distance_ecdf.png',dpi=160);plt.close(fig)
    fig2.savefig(output/'incoming_distance_histograms.png',dpi=160);plt.close(fig2)
    fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,purpose in zip(axes,('shop','leisure')):
        values=paired.loc[paired.purpose.eq(purpose),'relocation_km']
        ax.hist(values,bins=np.arange(0,51,1),histtype='step',linewidth=1.5)
        ax.set(title=purpose,xlabel='Destination relocation (km; view 0–50km)',ylabel='Activities')
    fig.savefig(output/'destination_relocations.png',dpi=160);plt.close(fig)


def plot_city_changes(variants, radius, output):
    fig,axes=plt.subplots(3,2,figsize=(13,16),constrained_layout=True)
    for row,(city,(x,y)) in enumerate(CENTRES.items()):
        bins=np.arange(-radius,radius+500,500)
        for column,purpose in enumerate(('shop','leisure')):
            grids=[]
            for data in variants.values():
                selection=data.purpose.eq(purpose)&(np.hypot(data.x-x,data.y-y)<=radius)
                grids.append(np.histogram2d(data.loc[selection,'x']-x,data.loc[selection,'y']-y,bins=(bins,bins))[0])
            change=grids[1]-grids[0];bound=max(1.,np.abs(change).max())
            image=axes[row,column].imshow(change.T,origin='lower',extent=(-radius/1000,radius/1000,-radius/1000,radius/1000),
                 cmap='RdBu_r',vmin=-bound,vmax=bound)
            axes[row,column].set(title=f'{city} — {purpose}',xlabel='km east of centre',ylabel='km north of centre')
            fig.colorbar(image,ax=axes[row,column],label='With − without opening hours (activities / 500m cell)')
    fig.savefig(output/'city_destination_changes.png',dpi=160);plt.close(fig)
