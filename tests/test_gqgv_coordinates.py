"""Freight coordinates must come from numeric named fields in mixed STATENT data."""
from contextlib import nullcontext
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from synthesis.freight.gqgv.trips import execute


class FreightCoordinatesTest(unittest.TestCase):
    def test_mixed_statent_columns_and_nonpositional_indices(self):
        zones = ['FR01', 'CH01']
        origins = pd.DataFrame([.5, .5], index=zones)
        destinations = pd.DataFrame([[.5, .5], [.5, .5]], index=zones, columns=zones)
        # Text comes first and labels are not positions: column order and the
        # dataframe's heterogeneous dtype must not affect facility coordinates.
        statent = pd.DataFrame({'enterprise_id':['one','two'], 'y':[1.,2.],
            'nuts_id_level_2':['CH01','CH01'], 'x':[3.,4.]}, index=[10,20])
        original = statent.copy(deep=True)
        stages = {
            'data.freight.gqgv.od': ({'truck':100}, {'truck':origins}, {'truck':destinations}),
            'data.spatial.nuts': gpd.GeoDataFrame({'nuts_id':zones},
                geometry=[box(20,20,30,30), box(0,0,5,5)], crs=2056),
            'data.spatial.swiss_border': gpd.GeoSeries([box(0,0,5,5)], crs=2056),
            'data.statent.statent': statent,
        }
        class Context:
            def stage(self, name): return stages[name]
            def config(self, name): return {'input_downsampling':1., 'random_seed':42}[name]
            def progress(self, **kwargs):
                class Progress:
                    def update(self, *args): pass
                return nullcontext(Progress())
        result = execute(Context())
        self.assertEqual(len(result),100)
        self.assertTrue(np.isfinite(result[['origin_x','origin_y','destination_x','destination_y']]).all().all())
        self.assertTrue(all(result[col].dtype == np.dtype('float64') for col in
                            ['origin_x','origin_y','destination_x','destination_y']))
        for prefix in ['origin','destination']:
            coords = set(result[[prefix+'_x',prefix+'_y']].itertuples(index=False, name=None))
            self.assertTrue({(3.,1.),(4.,2.)}.issubset(coords))
        pd.testing.assert_frame_equal(statent, original)


if __name__ == '__main__':
    unittest.main()
