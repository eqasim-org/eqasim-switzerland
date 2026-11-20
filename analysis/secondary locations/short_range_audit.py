"""Compare short replay errors with an optimistic facility-radius feasibility bound."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from cache_io import RunCache
from synthesis.population.spatial.secondary_nn.tools import destination_offer_masks,base_activity_purpose


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_dib_10_sNN.yml')
    parser.add_argument('--results',type=Path,default=Path(__file__).with_name('results'))
    args=parser.parse_args()
    cache=RunCache(args.config)
    destinations=cache.stage('synthesis.population.destinations')
    xy=np.column_stack([destinations.geometry.x,destinations.geometry.y])
    masks=destination_offer_masks(destinations)
    pools={purpose:xy[mask] for purpose,mask in masks.items()}
    trees={purpose:cKDTree(points) for purpose,points in pools.items() if len(points)}
    observed=pd.read_parquet(args.results/'replay_observed.parquet')
    observed=observed.loc[observed.crowfly_distance.between(10,1500)].copy()
    records=[]
    for row in observed.itertuples(index=False):
        purpose=base_activity_purpose(row.purpose)
        if purpose not in trees:continue
        origin=np.array([row.origin_x,row.origin_y])
        indices=trees[purpose].query_ball_point(origin,1800)
        if not indices:continue
        radii=np.linalg.norm(pools[purpose][indices]-origin,axis=1)
        records.append(dict(person_id=row.person_id,trip_id=row.trip_id,
            target_m=row.crowfly_distance,weight=row.weight,
            best_radius_error_m=np.abs(radii-row.crowfly_distance).min(),facilities=len(indices)))
    table=pd.DataFrame(records)
    output=args.results/'short_range_audit';output.mkdir(exist_ok=True)
    table.to_csv(output/'optimistic_radius_feasibility.csv',index=False)
    for low,high in [(10,100),(100,250),(250,500),(500,1000),(1000,1500)]:
        f=table.loc[table.target_m.between(low,high,inclusive='left')]
        if len(f):print(low,high,'trips',len(f),'optimistic mean error (m)',round(np.average(f.best_radius_error_m,weights=f.weight),2),flush=True)


if __name__=='__main__':main()
