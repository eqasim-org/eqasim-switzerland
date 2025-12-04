import pandas as pd
import os
import numpy as np
import matsim.runtime.eqasim as eqasim

def configure(context):
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("pt_preparation.pt_pricing.generate_config")
    context.stage("pt_preparation.pt_pricing.uspat_points")
    context.config("random_seed")


def execute(context):

    uspat_points = context.stage("pt_preparation.pt_pricing.uspat_points")
    N            = len(uspat_points)

    coord_x = uspat_points.geometry.x.to_numpy() 
    coord_y = uspat_points.geometry.y.to_numpy() 
    zone    = uspat_points["zone_id"].to_numpy() 

    # Generate all OD pairs
    orig_idx, dest_idx = np.meshgrid(np.arange(N), np.arange(N)) 
    mask = (orig_idx != dest_idx) #& (zone[orig_idx] != zone[dest_idx]) 
    orig_idx = orig_idx[mask] 
    dest_idx = dest_idx[mask] 

    od_df = pd.DataFrame({
        "origin_id": orig_idx, 
        "dest_id": dest_idx,
        "originX": coord_x[orig_idx],
        "originY": coord_y[orig_idx],
        "destinationX": coord_x[dest_idx],
        "destinationY": coord_y[dest_idx],
        "origin_zone": zone[orig_idx], 
        "destination_zone": zone[dest_idx] 
        })
    
    print(f"Size before sampling: {len(od_df)}")
    
    # Only keep one OD per origin zone, destination zone, origin point
    od_df = (
        od_df.sample(frac=1, random_state=context.config("random_seed"))  # shuffle rows randomly (fixed seed = reproducible)
            .drop_duplicates(subset=["origin_zone", "destination_zone", "origin_id"])
            .reset_index(drop=True)
    )
    
    print(f"Size after sampling: {len(od_df)}")

    od_df_expanded = pd.concat([
        od_df.assign(hasHalbtaxSubscription=False),
        od_df.assign(hasHalbtaxSubscription=True)
    ], ignore_index=True)
    
    print(f"Size after expanding (HT): {len(od_df)}")

    od_df = od_df_expanded

    # Transform in requests

    od_df["ID"]                     = range(len(od_df))
    od_df["homeX"]                  = od_df["originX"]
    od_df["homeY"]                  = od_df["originY"]
    od_df["departureTime_s"]        = np.random.randint(6*3600, 18*3600, size=len(od_df))
    od_df["hasGA"]                  = False
    #od_df["hasHalbtaxSubscription"] = False 
    od_df["hasVerbundAbo"]          = False
    od_df["hasStreckenAbo"]         = False
    od_df["hasGleis7Abo"]           = False
    od_df["hasJuniorAbo"]           = False
    od_df["age"]                    = 32

    trips = od_df[["ID", "originX", "originY", "destinationX", "destinationY",
        "homeX", "homeY", "departureTime_s",
        "hasGA", "hasHalbtaxSubscription", "hasVerbundAbo",
        "hasStreckenAbo", "hasGleis7Abo", "hasJuniorAbo",
        "age"]]
    
    #trips = trips[:10]
    
    for col in ["originX", "originY", "destinationX", "destinationY", "homeX", "homeY"]:
        trips[col] = trips[col].astype(int)
    
    print(f"Final size: {len(trips)}")
    config_path = context.stage("pt_preparation.pt_pricing.generate_config")
    results = []
    
    chunk_size = 200000
    
    num_chunks = len(trips) // chunk_size + 1
    
    for i in range(num_chunks):
        start_idx   = i * chunk_size
        end_idx     = min(len(trips), (i+1) * chunk_size)
        trips_chunk = trips.iloc[start_idx:end_idx] 
        print(f"\nProcessing chunk {i+1}/{num_chunks} ({len(trips_chunk)} rows)...")

        requests_path = os.path.join(context.path(), f"requests_chunk_{i}.csv")
        output_path = os.path.join(context.path(), f"requests_price_chunk_{i}.csv")
        
        trips_chunk.to_csv(requests_path, index=False)
        
        eqasim.run(context, "org.eqasim.switzerland.ch.utils.pricing.RunComputeTransitPrices", [
            "--config-path", config_path,
            "--requests-path", requests_path,
            "--output-path", output_path
        ])

        assert os.path.exists(output_path), f"Missing output: {output_path}"
        
        chunk_result = pd.read_csv(output_path)
        results.append(chunk_result)
    
    final_result = pd.concat(results, ignore_index=True)
    print(f"\n✅ Finished processing {len(final_result)} total rows across {num_chunks} chunks.")

    return od_df, final_result