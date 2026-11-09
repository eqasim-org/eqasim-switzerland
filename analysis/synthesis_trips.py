"""Compare unshuffled, legacy and half-hour synthesis trip schedules."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from synthesis.population import trips as trip_stage


METHODS = ("none", "legacy", "half_hour")
LABELS = {"none": "Original", "legacy": "Legacy ±30 min", "half_hour": "Within half-hour"}
COLORS = {"none": "#666666", "legacy": "#2878b5", "half_hour": "#e48726"}


def configure(context):
    trip_stage.configure(context)
    context.config("analysis_path")


def execute(context):

    # Run the actual trip stage on identical upstream inputs for each method.

    output = Path(context.config("analysis_path")) / "synthesis_trips"
    output.mkdir(parents=True, exist_ok=True)
    activities, summaries, changes = {}, [], {}
    reference = None
    for method in METHODS:
        folder = output / method
        folder.mkdir(exist_ok=True)
        trips = trip_stage.execute(MethodContext(context, method, folder))
        ordered = trips.sort_values(["person_id", "trip_index"])
        data = ordered[["person_id", "trip_index", "arrival_time", "departure_time",
                        "trip_duration", "following_purpose"]].reset_index(drop=True)
        activities[method] = activity_table(data)
        if reference is None:
            reference = data
        if not data[["person_id", "trip_index"]].equals(reference[["person_id", "trip_index"]]):
            raise ValueError("Shuffle comparisons produced different person/trip keys")
        changes[method] = pd.DataFrame({
            "activity_start": (data.arrival_time - reference.arrival_time) / 60.,
            "activity_duration": (activities[method].duration - activities["none"].duration) / 60.,
            "trip_duration": data.trip_duration - reference.trip_duration,
        })
        summaries.append(schedule_summary(method, data, reference, activities[method]))

    # Save distributions and diagnostics without exporting person-level records.

    plot_comparison(activities, changes, output)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps({
        "random_seed": int(context.config("random_seed")),
        "specific_day_scenario": context.config("specific_day_scenario"),
        "notes": "Activity starts are arrivals; duration plots use intermediate activities only. "
                 "Initial home and final open-ended activities are excluded from durations. "
                 "Border-event draws are identical; half_hour also leaves one minute before initial departures. "
                 "Its final activities extend past 30h when needed for the one-minute minimum. "
                 "Existing invalid source chains are retained.",
        "methods": summary.to_dict("records"),
    }, indent=2), encoding="utf-8")

    # Return generated figures and summaries for review.

    return {"output": str(output), "figures": [str(path) for path in sorted(output.glob("*.png"))],
            "summary": str(output / "summary.csv")}


class MethodContext:
    def __init__(self, parent, method, folder):
        self.parent, self.method, self.folder = parent, method, folder

    def config(self, key):
        return self.method if key == "trip_time_shuffle" else self.parent.config(key)

    def stage(self, name):
        return self.parent.stage(name)

    def path(self):
        return str(self.folder)


def activity_table(trips):
    result = trips[["person_id", "trip_index", "following_purpose", "arrival_time"]].rename(
        columns={"following_purpose": "purpose", "arrival_time": "start_time"}).copy()
    following_departure = trips.groupby("person_id", sort=False).departure_time.shift(-1)
    result["duration"] = following_departure - result.start_time
    return result


def schedule_summary(method, trips, reference, activities):
    starts_changed_bin = np.floor(trips.arrival_time / 1800).ne(np.floor(reference.arrival_time / 1800))
    start_delta = trips.arrival_time - reference.arrival_time
    preceding_arrival = trips.groupby("person_id", sort=False).arrival_time.shift()
    source_preceding = reference.groupby("person_id", sort=False).arrival_time.shift()
    overlaps = trips.departure_time.lt(preceding_arrival)
    initial = trips.groupby("person_id", sort=False).departure_time.first()
    terminal = trips.groupby("person_id", sort=False).arrival_time.last()
    final_duration = np.maximum(108000. - terminal, 60.) if method == "half_hour" else 108000. - terminal
    return {
        "method": method, "trips": len(trips), "persons": int(trips.person_id.nunique()),
        "intermediate_activities": int(activities.duration.notna().sum()),
        "starts_outside_original_half_hour": int(starts_changed_bin.sum()),
        "starts_outside_original_half_hour_percent": float(starts_changed_bin.mean() * 100),
        "negative_departures": int(trips.departure_time.lt(0).sum()),
        "negative_trip_durations": int(trips.trip_duration.lt(0).sum()),
        "overlapping_trips": int(overlaps.sum()),
        "new_overlapping_trips": int((overlaps & ~reference.departure_time.lt(source_preceding)).sum()),
        "zero_duration_activities": int(activities.duration.eq(0).sum()),
        "intermediate_activities_under_one_minute": int(activities.duration.lt(60).sum()),
        "initial_activities_under_one_minute": int(initial.lt(60).sum()),
        "final_activities_under_one_minute": int((final_duration < 60).sum()),
        "minimum_intermediate_duration_seconds": float(activities.duration.min()),
        "mean_activity_duration_minutes": float(activities.duration.mean() / 60),
        "median_activity_duration_minutes": float(activities.duration.median() / 60),
        "min_start_change_minutes": float(start_delta.min() / 60),
        "max_start_change_minutes": float(start_delta.max() / 60),
        "changed_trip_durations": int(trips.trip_duration.ne(reference.trip_duration).sum()),
    }


def histogram(ax, values, bins, method):
    values = np.asarray(values, dtype=float)
    ax.hist(values[np.isfinite(values)], bins=bins, histtype="step", linewidth=1.5,
            color=COLORS[method], label=LABELS[method])
    ax.grid(axis="y", alpha=.2)
    ax.set_ylabel("Activities")
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, output, name):
    fig.savefig(output / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(activities, changes, output):
    maximum_hour = max(30., np.ceil(activities["none"].start_time.max() / 3600))
    time_bins = np.arange(0, maximum_hour + 1 / 6, 1 / 6)
    duration_bins = np.arange(0, 24 * 60 + 10, 10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    for method, data in activities.items():
        histogram(axes[0], data.start_time / 3600, time_bins, method)
        histogram(axes[1], data.start_time.mod(1800) / 60, np.arange(0, 40, 10), method)
    axes[0].set(xlabel="Activity start (hours after midnight)", title="Activity starts — 10-minute bins")
    axes[1].set(xlabel="Minutes into half-hour interval", title="Position within the half-hour — 10-minute bins")
    axes[0].legend()
    save_figure(fig, output, "activity_start_times.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    for method, data in activities.items():
        histogram(axes[0], data.duration / 60, duration_bins, method)
        histogram(axes[1], data.duration / 60, np.arange(0, 130, 10), method)
    axes[0].set(xlabel="Duration (minutes)", title="Intermediate activity durations — 10-minute bins")
    axes[1].set(xlabel="Duration (minutes)", title="Activities up to 2 hours — 10-minute bins")
    axes[0].legend()
    save_figure(fig, output, "activity_durations.png")

    purposes = sorted(activities["none"].purpose.dropna().unique())
    for kind in ("starts", "durations"):
        fig, axes = plt.subplots(int(np.ceil(len(purposes) / 3)), 3,
                                 figsize=(16, 3.4 * np.ceil(len(purposes) / 3)), constrained_layout=True)
        for ax, purpose in zip(np.ravel(axes), purposes):
            for method, data in activities.items():
                values = data.loc[data.purpose.eq(purpose), "start_time" if kind == "starts" else "duration"]
                histogram(ax, values / (3600 if kind == "starts" else 60),
                          time_bins if kind == "starts" else duration_bins, method)
            ax.set(title=f"{purpose} — 10-minute bins", xlabel="Hours after midnight" if kind == "starts" else "Duration (minutes)")
            if kind == "durations":
                ax.set_xlim(0, 720)
        for ax in np.ravel(axes)[len(purposes):]:
            ax.set_visible(False)
        np.ravel(axes)[0].legend(fontsize=8)
        save_figure(fig, output, f"activity_{kind}_by_purpose.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for method in ("legacy", "half_hour"):
        histogram(axes[0], changes[method].activity_start, np.arange(-30, 41, 10), method)
        histogram(axes[1], changes[method].activity_duration, np.arange(-60, 71, 10), method)
        histogram(axes[2], changes[method].trip_duration, np.arange(-2.5, 3.5), method)
    axes[0].set(title="Activity-start changes — 10-minute bins", xlabel="Change from original (minutes)")
    axes[1].set(title="Activity-duration changes — 10-minute bins", xlabel="Change from original (minutes)")
    axes[2].set(title="Trip-duration changes", xlabel="Change from original (seconds)", ylabel="Trips")
    axes[0].legend()
    save_figure(fig, output, "time_changes.png")


def validate(context):
    return hashlib.sha256(Path(trip_stage.__file__).read_bytes()).hexdigest()


class CachedContext:
    """Explicit read-only reuse of unique existing caches for a focused analysis."""
    def __init__(self, settings):
        self.options = settings["config"].copy()
        self.root = Path(settings["working_directory"])
        from synpp import ConfigurationContext
        from synpp.pipeline import flatten
        configuration = ConfigurationContext(flatten(self.options))
        configure(configuration)
        self.options = configuration.required_config
        self.inputs = {}
        for dependency in configuration.required_stages:
            name = dependency["descriptor"]
            matches = list(self.root.glob(name + "__*.p"))
            if len(matches) != 1:
                raise ValueError(f"Expected one existing cache for {name}, found {len(matches)}")
            self.inputs[name] = pd.read_pickle(matches[0])

    def config(self, key):
        return self.options[key]

    def stage(self, name):
        return self.inputs[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument("--cached-inputs", action="store_true", required=True,
                        help="Use existing unique input caches read-only, without rebuilding upstream stages")
    args = parser.parse_args()
    with open(args.config) as stream:
        settings = yaml.safe_load(stream)
    print(json.dumps(execute(CachedContext(settings)), indent=2))


if __name__ == "__main__":
    main()
