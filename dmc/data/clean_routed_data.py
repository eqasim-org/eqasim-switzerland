import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import logging
logger = logging.getLogger("synpp")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure(context):
    # Path to the csv file containing the dataset of routed trips  
    context.config("data_path")    
    context.config("routed_trips_file", "Routed_alternatives_v3_FINAL.txt")


def execute(context):
    path_to_data = os.path.join(context.config("data_path"),"dmc",context.config("routed_trips_file"))
    assert os.path.exists(path_to_data), f"The provided path for routed_trips_path ({path_to_data}) does not exist."

    df = pd.read_csv(path_to_data, sep=";", low_memory=False)

    modes = ["pt","car","walk","bike"]
    choice_cols = ["choice_"+m for m in modes]

    # Remove trips with less than two routed choices
    keep_row  = df[choice_cols].notna().sum(axis=1)>1    
    df = df[keep_row].reset_index(drop=True)
    logger.info("%d trips are removed because they have less than 2 routed modes.", (~keep_row).sum())

    # Remove trips with no mode selected
    keep_row  = df["w_verkehrsmittel"].notna()  
    df = df[keep_row].reset_index(drop=True)
    logger.info("%d trips are removed because their mode is nan.", (~keep_row).sum())

    # Sort by person and trip id
    df = df.sort_values(by=["HHNR","WEGNR"]).reset_index(drop=True)

    # only keep selected alternatives
    modes_cols = {mode: [col for col in df.columns if "_"+mode in col] for mode in modes}

    filtered_df = df[['HHNR', 'WEGNR', "w_verkehrsmittel"]].copy()
    filtered_df = filtered_df.rename(columns={"HHNR":"person_id","WEGNR":"trip_id","w_verkehrsmittel":"mode"})

    ### Set the right mode (pt, car, walk, bike, car_passenger)
    MOD_DICT = { # this is taken from Microcensus trips stage
        -99: "unknown",  # Pseudo stage
        1: "pt",         # Plane
        2: "pt",         # Train
        3: "pt",         # Postauto
        4: "pt",         # Ship
        5: "pt",         # Tram
        6: "pt",         # Bus
        7: "pt",         # Other PT
        8: "pt",         # Reisecar (coach)
        9: "car",        # Car
        10: "car",       # Truck
        11: "pt",        # Taxi
        12: "car",       # Motorbike
        13: "car",       # Mofa
        14: "bike",      # Bicycle / E-bike
        15: "walk",      # Walking
        16: "bike",      # Machines similar to a vehicle
        17: "unknown"    # Other / don't know
    }
    filtered_df["mode"] = filtered_df["mode"].map(MOD_DICT)

    for mode, cols in modes_cols.items():    
        selected_alternative = "choice_"+mode
        mode_new_cols = set([col.split('.')[0] for col in cols if col!=selected_alternative])
        
        for col in mode_new_cols:
            all_alternatives = [c for c in cols if col in c]
            
            if len(all_alternatives):
                df["sel_col"] = df[selected_alternative].map(lambda x: col+'.'+str(int(x)) if (not np.isnan(x)) else "noCol")
                get_col = lambda x: x[x["sel_col"]] if x["sel_col"]!="noCol" else np.nan
                filtered_df[col] = df[["sel_col",*all_alternatives]].apply(get_col, axis=1)
            else:
                logger.warning("No alternatives found for %s", mode_new_cols)

    df = df.drop(columns="sel_col")

    # Filtering
    ### Start with unknown modes
    to_remove = filtered_df["mode"] == "unknown"


    ### If mode is selected but its attributes are missings -> filter out
    for mode in modes:
        mode_is_selected = filtered_df["mode"] == mode
        
        mode_cols = [col for col in filtered_df.columns if f"_{mode}" in col]
        mode_routed  = ((filtered_df[mode_cols].isna()).sum(axis=1)==0) 
        mode_routed &= filtered_df[f"expectedModeUsed_{mode}"]
        
        to_remove |= (mode_is_selected & ~mode_routed)

    ### Apply filter
    filtered_df = filtered_df[~to_remove].reset_index(drop=True)
    logger.info("%d trips are removed because they have less than 2 routed modes or the selected mode is not routed.", to_remove.sum())

    return filtered_df
