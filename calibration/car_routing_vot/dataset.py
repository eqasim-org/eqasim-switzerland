import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("synpp:calibration.car_routing_vot.dataset")

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")
    zip_path = os.path.join(data_path, "GPS_data/MOBIS_Covid_version_3.zip")
    file_path = "tracking/legs.csv"
    with zipfile.ZipFile(zip_path) as z:
        with z.open(file_path) as f:
            df = pd.read_csv(f)

    #1. Filter by mode
    df = df[df["mode"].str.contains("car", case=False, na=False)]
    #2. Keep legs within Switzerland
    df = df[df["in_switzerland"]]
    #3. Get relevant columns
    df = df[["participant_id", "leg_id", "started_at", "finished_at", "start_x", "start_y", "end_x", "end_y", "length", "duration"]]
    #4. Compute average speed
    df["average_speed"] = df["length"] / df["duration"] * 3.6  # convert m/s to km/h
    #5. Transform dates to datetime
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])
    #6. Compute euclidean distance
    df["euclidean_distance"] = ((df["end_x"] - df["start_x"])**2 + (df["end_y"] - df["start_y"])**2)**0.5
    #7. Plot evolution of average speed over time
    retained_dates = ("2021-07-15", "2022-12-01")
    plot_evolution(context, df, retained_dates)
    #8. Limit the data to avoid covid and other impacts
    df = df[df["started_at"].between(*retained_dates)]
    #9. Remove tripsthat are too short
    df = df[df["euclidean_distance"] >= 1000].reset_index(drop=True)
    #9. log stats
    logger.info(f"\tTotal number of participants: {df['participant_id'].nunique()}")
    logger.info(f"\tTotal number of legs after filtering: {len(df)}")
    logger.info(f"\tAverage speed: {df['average_speed'].mean():.2f} km/h")
    logger.info(f"\tAverage leg length: {df['length'].mean():.2f} m")
    return df







def plot_evolution(context, df, retained_dates):
    df = df.copy()
    df["year"] = df["started_at"].dt.year
    df["week"] = df["started_at"].dt.isocalendar().week.astype(int)
    weekly_speed = df.groupby(["year", "week"])["average_speed"].mean().reset_index()
    weekly_speed = weekly_speed.sort_values(["year", "week"]).reset_index(drop=True)
    weekly_speed["abs_week"] = weekly_speed["year"] * 53 + weekly_speed["week"]

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(weekly_speed["abs_week"], weekly_speed["average_speed"],
            marker="o", markersize=3.5, linewidth=1.8, color="steelblue", alpha=0.85)

    # Place x-ticks at the first week of each year
    year_starts = weekly_speed.groupby("year")["abs_week"].min()
    ax.set_xticks(year_starts.values)
    ax.set_xticklabels([str(y) for y in year_starts.index], fontsize=10)
    ax.set_xlabel("Date (Year)", fontsize=12)
    ax.set_ylabel("Average Speed (km/h)", fontsize=12)
    ax.set_title("Weekly Average Car Speed Over Time", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.set_xlim(weekly_speed["abs_week"].min(), weekly_speed["abs_week"].max())

    # Grey area for COVID measures: 16 March 2020 – 19 June 2020
    covid_start = 2020 * 53 + pd.Timestamp("2020-03-16").isocalendar()[1]
    covid_end   = 2020 * 53 + pd.Timestamp("2020-06-19").isocalendar()[1]
    ax.axvspan(covid_start, covid_end, color="grey", alpha=0.3, label="COVID measures")

    # green area for retained_dates
    retained_start = pd.Timestamp(retained_dates[0])
    retained_end   = pd.Timestamp(retained_dates[1])
    retained_start_week = retained_start.year * 53 + retained_start.isocalendar()[1]
    retained_end_week   = retained_end.year * 53 + retained_end.isocalendar()[1]
    ax.axvspan(retained_start_week, retained_end_week, color="green", alpha=0.2, label="Retained data period")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(context.path(), "averagespeed_by_year.png"), dpi=200, bbox_inches="tight")