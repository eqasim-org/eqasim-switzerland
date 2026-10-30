"""Person-level visits to fixed city-centre catchments, with radius sensitivity."""
import numpy as np
import pandas as pd
from pyproj import Transformer


def center_xy(region):
    return Transformer.from_crs(4326,2056,always_xy=True).transform(*region['center_lon_lat'])


def centre_visits(datasets, regions, purposes, cohort):
    records = []
    for city, region in regions.items():
        cx,cy = center_xy(region)
        x0,y0,x1,y1 = region['bounds']
        for name, frame in datasets.items():
            city_origin = frame.origin_x.between(x0,x1) & frame.origin_y.between(y0,y1)
            for purpose in ['all',*purposes]:
                chosen = city_origin if purpose=='all' else city_origin & frame.purpose.eq(purpose)
                columns = ['person_id','weight','destination_x','destination_y'] + (['draw'] if 'draw' in frame else [])
                f = frame.loc[chosen,columns].copy()
                if f.empty:
                    continue
                if 'draw' not in f:
                    f['draw'] = 0
                f['radius_km'] = np.hypot(f.destination_x-cx,f.destination_y-cy)/1000
                # One nearest visit per person and draw; replay weights already divide by draws.
                people = f.groupby(['person_id','draw'],sort=False).agg(radius_km=('radius_km','min'),weight=('weight','first'))
                order = np.argsort(people.radius_km.to_numpy())
                distance = people.radius_km.to_numpy()[order]
                mass = np.r_[0.,np.cumsum(people.weight.to_numpy()[order])]
                n_people = f.person_id.nunique()
                for radius in np.linspace(0,5,26):
                    share = mass[np.searchsorted(distance,radius,side='right')]/mass[-1]
                    records.append(dict(city=city,cohort=cohort,dataset=name,purpose=purpose,
                        radius_km=radius,share_people_visiting=share,persons=n_people,
                        person_draws=len(people),center_x=cx,center_y=cy))
    return pd.DataFrame(records)
