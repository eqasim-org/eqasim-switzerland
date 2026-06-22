import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def configure(context):
    context.stage("synthesis.population.models.subscriptions")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")

def execute(context):
    out_dir = context.config("output_path")
    mz_trips,filtered_out,_,_ = context.stage("data.microcensus.trips")
    mz_persons = context.stage("data.microcensus.persons")
    #mz_persons = mz_persons[mz_persons["age"]>15]
    mz_persons = mz_persons[(~mz_persons.person_id.isin(filtered_out)) & (mz_persons.weekend==False)]
    mz_trips = mz_trips[mz_trips.person_id.isin(mz_persons.person_id)].reset_index(drop=True)
    # merge to get weights
    mz_trips = mz_trips.merge(mz_persons[["person_id","household_weight","person_weight"]], on="person_id", how="left")
    mz_trips = mz_trips[mz_trips["mode"].isin(["car","car_passenger","pt","bike","walk"])]  
    mz_persons.loc[mz_persons["age"] > 15, "subscriptions_junior"] = False
    mz_ga_persons = mz_persons[(mz_persons.subscriptions_ga==True) | (mz_persons.subscriptions_junior==True)].person_id
    mz_trips = mz_trips[mz_trips.person_id.isin(mz_ga_persons)]

    mz_trips["euclidean_distance_km"] = mz_trips["crowfly_distance"] / 1000
    filepath = "%s/switzerland_trips.csv" % out_dir
    trips = pd.read_csv(filepath, encoding="latin1", sep=";")
    trips["crowfly_distance"] = trips["crowfly_distance"] / 1000
    trips = trips.rename(columns={"crowfly_distance" : "euclidean_distance_km"})

    persons = context.stage("synthesis.population.models.subscriptions")
    #persons = persons[persons["age"]>15]
    ga_persons = persons[persons.pt_subscription==1].person_id
    trips = trips[trips.person_id.isin(ga_persons)]

    mz_trips = mz_trips[(mz_trips.euclidean_distance_km<200)&(mz_trips.euclidean_distance_km>1e-3)]
    trips = trips[(trips.euclidean_distance_km<200)&(trips.euclidean_distance_km>1e-3)]

    mz_home = mz_trips[(mz_trips.origin_purpose=="home") & (mz_trips.purpose.isin(["work","work_secondary"]))].copy()

    df_home = trips[(trips.preceding_purpose=="home")&(trips.following_purpose.isin(["work","work_secondary"]))].copy()

    mzi = mz_home# [mz_home["mode"]=="car_passenger"]
    dfi = df_home# [df_home["main_mode"]=="car_passenger"]
    weight_col = "person_weight"

    # improved distance distribution comparison
    weight = mzi[weight_col] if weight_col is not None else None

    max_x = max(
        mzi.euclidean_distance_km.quantile(0.99),
        dfi.euclidean_distance_km.quantile(0.99)
    )
    max_x = 200
    bins = np.linspace(0, max_x, 61)

    fig, ax = plt.subplots(figsize=(10,5))

    # filled histograms
    ax.hist(mzi.euclidean_distance_km, bins=bins, density=True, weights=weight,
            alpha=0.35, color='C0', label='Microcensus', edgecolor='C0')
    ax.hist(dfi.euclidean_distance_km, bins=bins, density=True,
            alpha=0.35, color='C1', label='Simulated', edgecolor='C1')

    # clear step outlines for readability
    ax.hist(mzi.euclidean_distance_km, bins=bins, density=True, weights=weight,
            histtype='step', color='C0', linewidth=1.2)
    ax.hist(dfi.euclidean_distance_km, bins=bins, density=True,
            histtype='step', color='C1', linewidth=1.2, linestyle='--')

    # median and mean markers
    m_med, m_mean = np.quantile(mzi.euclidean_distance_km, 0.5), np.average(mzi.euclidean_distance_km, weights=weight)
    s_med, s_mean = dfi.euclidean_distance_km.median(), dfi.euclidean_distance_km.mean()
    ymax = ax.get_ylim()[1]
    ax.axvline(m_med, color='C0', linestyle=':', linewidth=1)
    ax.axvline(s_med, color='C1', linestyle=':', linewidth=1)
    ax.axvline(m_mean, color='C0', linestyle='-.', linewidth=1)
    ax.axvline(s_mean, color='C1', linestyle='-.', linewidth=1)

    ax.text(0.3*max_x, ymax*0.92, f"median {m_med:.2f} km", color='C0', ha='center', va='top', fontsize=11)
    ax.text(0.3*max_x, ymax*0.83, f"median {s_med:.2f} km", color='C1', ha='center', va='top', fontsize=11)
    ax.text(0.5*max_x, ymax*0.92, f"mean {m_mean:.2f} km", color='C0', ha='center', va='top', fontsize=11)
    ax.text(0.5*max_x, ymax*0.83, f"mean {s_mean:.2f} km", color='C1', ha='center', va='top', fontsize=11)
    # annotations and styling
    ax.set_xlim(0, max_x)
    ax.set_xlabel("Euclidean distance (km)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Distance distribution: Microcensus vs Simulated {' (' + weight_col + ')' if weight_col is not None else ''}", fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(frameon=False, fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribution.png"), dpi=200)
    plt.close()
