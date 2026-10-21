"""Public OSM eligibility applies to destinations and every model feature."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from synthesis.population.spatial.secondary_nn.tools import (
    OSM_FEATURES, OSM_KINDS, aggregate_osm_features, candidate_points,
    derive_water_features, public_access, public_osm_features, osm_access_decision,
)


@pytest.mark.parametrize('restriction', ['private', 'no', 'customers', 'customer', 'permit',
                                        'members', 'residents', 'delivery', 'forestry', 'unknown'])
@pytest.mark.parametrize('key', ['access', 'foot'])
def test_restricted_access_is_not_public(key, restriction):
    assert not public_access({'leisure': 'park', key: restriction})


def test_gardens_require_explicit_permission_and_pedestrian_access_takes_precedence():
    assert not public_access({'leisure': 'garden'})
    assert not public_access({'leisure': 'garden', 'garden:type': 'residential'})
    assert public_access({'leisure': 'garden', 'access': 'yes'})
    assert public_access({'leisure': 'garden', 'access': 'permissive'})
    assert public_access({'highway': 'path', 'access': 'private', 'foot': 'yes'})
    assert not public_access({'leisure': 'park', 'access': 'yes', 'foot': 'private'})
    assert not public_access({'leisure': 'park', 'membership': 'yes'})
    assert not public_access({'leisure': 'park', 'access:conditional': 'no @ (20:00-06:00)'})
    assert public_access({'aerialway': 'station', 'fee': 'yes'})
    assert public_access({'aerialway': 'station', 'fee': 'yes', 'access': 'yes'})


def frame(rows):
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=2056)


def test_private_features_do_not_contribute_any_h3_input_or_destination():
    records = []
    for i, kind in enumerate(OSM_KINDS):
        geometry = (box(0, 0, 100, 100) if kind in ('park', 'garden', 'ski_area')
                    else LineString([(0, 0), (100, 100)]) if 'path' in kind or kind in ('aerialway', 'piste', 'shoreline')
                    else Point(50, 50))
        records.append(dict(osm_id=str(i), kind=kind, name='', public=False, geometry=geometry))
    private = public_osm_features(frame(records))
    cells = gpd.GeoDataFrame(geometry=[box(-10, -10, 110, 110)], crs=2056)
    assert aggregate_osm_features(cells, private)[OSM_FEATURES].eq(0).all().all()
    assert candidate_points(private).empty


def test_private_areas_remove_nested_pois_and_clip_paths_and_overlapping_parks():
    features = frame([
        dict(osm_id='private', kind='garden', name='', public=False, geometry=box(0, 0, 100, 100)),
        dict(osm_id='view', kind='viewpoint', name='', public=True, geometry=Point(50, 50)),
        dict(osm_id='path', kind='path', name='', public=True, geometry=LineString([(-50, 50), (150, 50)])),
        dict(osm_id='park', kind='park', name='', public=True, geometry=box(-50, 0, 150, 100)),
    ])
    public = public_osm_features(features)
    assert set(public.osm_id) == {'path', 'park'}
    assert public.loc[public.kind == 'path'].geometry.length.sum() == pytest.approx(100)
    assert public.loc[public.kind == 'park'].geometry.area.sum() == pytest.approx(10000)
    cells = gpd.GeoDataFrame(geometry=[box(-100, -100, 200, 200)], crs=2056)
    summary = aggregate_osm_features(cells, public)
    assert summary.osm_viewpoint_count.iloc[0] == 0
    assert summary.osm_park_area_m2.iloc[0] == pytest.approx(10000)


def test_shoreline_only_counts_publicly_accessible_sections():
    features = frame([
        dict(osm_id='lake', kind='water', name='', public=True, geometry=box(0, 0, 200, 100)),
        dict(osm_id='public_path', kind='path', name='', public=True, geometry=LineString([(0, -5), (200, -5)])),
        dict(osm_id='private_path', kind='path', name='', public=False, geometry=LineString([(0, 105), (200, 105)])),
    ])
    derived = derive_water_features(features, 10)
    assert derived.public.all()
    shoreline = derived.loc[derived.kind == 'shoreline']
    assert 200 <= shoreline.geometry.length.sum() < 220
    assert shoreline.geometry.bounds.maxy.max() < 10
    assert set(candidate_points(derived).kind) == {'shore_path'}


def test_private_water_does_not_generate_public_shoreline_features():
    features = frame([
        dict(osm_id='lake', kind='water', name='', public=False, geometry=box(0, 0, 100, 100)),
        dict(osm_id='path', kind='path', name='', public=True, geometry=LineString([(0, -5), (100, -5)])),
    ])
    derived = derive_water_features(features, 10)
    assert not derived.kind.isin(['shoreline', 'shore_path']).any()


def test_stage_excludes_private_gardens_and_untagged_features_inside_them(tmp_path):
    from data.osm.secondary_features import execute
    source = tmp_path / 'osm' / 'switzerland-test.osm'
    source.parent.mkdir()
    source.write_text('''<osm version="0.6">
<node id="1" lon="8.50" lat="47.30"/><node id="2" lon="8.51" lat="47.30"/>
<node id="3" lon="8.51" lat="47.31"/><node id="4" lon="8.50" lat="47.31"/>
<node id="5" lon="8.505" lat="47.305"><tag k="tourism" v="viewpoint"/></node>
<node id="6" lon="8.52" lat="47.32"><tag k="tourism" v="viewpoint"/><tag k="foot" v="yes"/></node>
<node id="8" lon="8.53" lat="47.33"><tag k="tourism" v="viewpoint"/></node>
<node id="7" lon="8.509" lat="47.309"/>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
<tag k="leisure" v="garden"/><tag k="access" v="private"/></way>
<way id="11"><nd ref="5"/><nd ref="7"/><tag k="highway" v="path"/></way>
</osm>''')
    class Context:
        def config(self, key):
            return {'data_path': str(tmp_path), 'osm_file': [source.name],
                    'secondary_nn_osm_water_path_buffer_m': 30}[key]
        def path(self):
            return str(tmp_path)
    result = execute(Context())
    assert result.public.all()
    assert set(result.osm_id) == {'n6', 'n8'}
    assert (tmp_path / 'secondary_features.parquet').is_file()


@pytest.mark.parametrize('tags', [
    {'leisure': 'park'}, {'leisure': 'winter_sports'},
    {'tourism': 'viewpoint'}, {'tourism': 'attraction', 'name': 'Named attraction'},
    {'amenity': 'place_of_worship'}, {'aerialway': 'station'},
    {'aerialway': 'cable_car'}, {'piste:type': 'downhill'},
    {'natural': 'water'}, {'waterway': 'river'}, {'highway': 'footway'},
])
def test_visitor_features_do_not_require_explicit_public_tags(tags):
    assert public_access(tags)
    assert public_access(dict(tags, access=''))
    assert public_access(dict(tags, access='yes'))
    assert not public_access(dict(tags, access='private'))
    assert not public_access(dict(tags, foot='no'))


@pytest.mark.parametrize('permission', ['yes', 'public', 'permissive', 'designated', 'official'])
@pytest.mark.parametrize('key', ['access', 'foot'])
def test_only_explicit_public_permission_is_accepted(key, permission):
    assert public_access({'tourism': 'viewpoint', key: permission})


def test_stage_restores_untagged_visitor_features_and_accessible_shore(tmp_path):
    from data.osm.secondary_features import execute
    source = tmp_path / 'osm' / 'switzerland-unknown.osm'
    source.parent.mkdir()
    source.write_text('''<osm version="0.6">
<node id="1" lon="8.50" lat="47.30"/><node id="2" lon="8.51" lat="47.30"/>
<node id="3" lon="8.51" lat="47.31"/><node id="4" lon="8.50" lat="47.31"/>
<node id="5" lon="8.52" lat="47.32"><tag k="tourism" v="viewpoint"/></node>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
<tag k="leisure" v="park"/></way>
<way id="11"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
<tag k="natural" v="water"/></way>
<way id="12"><nd ref="1"/><nd ref="2"/><tag k="highway" v="footway"/></way>
</osm>''')
    class Context:
        def config(self, key):
            return {'data_path': str(tmp_path), 'osm_file': source.name,
                    'secondary_nn_osm_water_path_buffer_m': 30}[key]
        def path(self):
            return str(tmp_path)
    result = execute(Context())
    saved = gpd.read_parquet(tmp_path / 'secondary_features.parquet')
    assert result.public.all()
    assert set(result.kind) == {'park', 'viewpoint', 'path', 'shore_path', 'shoreline'}
    assert set(saved.kind) == set(result.kind)
    cells = gpd.GeoDataFrame(geometry=[box(2600000, 1200000, 2800000, 1400000)], crs=2056)
    assert aggregate_osm_features(cells, result).osm_park_area_m2.iloc[0] > 0
    assert not candidate_points(result).empty


@pytest.mark.parametrize('garden_type', ['residential', 'residental', 'private'])
@pytest.mark.parametrize('tag', ['garden:type', 'garden'])
def test_named_residential_gardens_remain_excluded(garden_type, tag):
    assert not public_access({'leisure': 'garden', tag: garden_type, 'name': 'Villa garden'})
    assert not public_access({'leisure': 'garden', tag: garden_type, 'access': 'yes'})


@pytest.mark.parametrize('garden_type', ['botanical', 'arboretum', 'community'])
def test_visitor_gardens_are_kept_without_access_tags(garden_type):
    assert public_access({'leisure': 'garden', 'garden:type': garden_type})
    assert not public_access({'leisure': 'garden', 'garden:type': garden_type, 'access': 'private'})


def test_named_gardens_need_supporting_evidence():
    assert not public_access({'leisure': 'garden', 'name': 'Villa garden'})
    assert public_access({'leisure': 'garden', 'name': 'Jardin public', 'operator:type': 'government'})
    assert public_access({'leisure': 'garden', 'name': 'Visitor garden', 'tourism': 'attraction'})
    assert not public_access({'leisure': 'garden', 'name': 'Villa garden', 'building': 'house'})


@pytest.mark.parametrize('tags', [
    {'aerialway': 'station'}, {'aerialway': 'gondola'},
    {'piste:type': 'downhill'}, {'leisure': 'winter_sports'},
    {'tourism': 'attraction', 'name': 'Museum garden'},
    {'leisure': 'garden', 'garden:type': 'botanical'},
])
def test_ticketed_visitor_facilities_are_kept(tags):
    assert public_access(dict(tags, access='customers', fee='yes', ownership='private'))
    assert not public_access(dict(tags, access='private'))
    assert not public_access(dict(tags, access='no'))
    assert not public_access(dict(tags, access='customers', membership='yes'))


@pytest.mark.parametrize('tags', [
    {'aerialway': 'goods'}, {'aerialway': 'station', 'usage': 'freight'},
    {'aerialway': 'station', 'aerialway:access': 'no'},
    {'aerialway': 'cable_car', 'usage': 'industrial'},
])
def test_non_visitor_aerialways_are_excluded(tags):
    assert not public_access(tags)


def garden_records():
    records = []
    for osm_id, tags, geometry in [
        ('garden', {'leisure': 'garden', 'name': 'Jardin'}, box(0, 0, 100, 100)),
        ('path', {'highway': 'footway'}, LineString([(-10, 50), (110, 50)])),
    ]:
        kind = 'garden' if osm_id == 'garden' else 'path'
        public, reason = osm_access_decision(tags, kind)
        records.append(dict(osm_id=osm_id, kind=kind, name=tags.get('name', ''),
                            public=public, access_reason=reason, geometry=geometry))
    return records


def test_named_garden_with_internal_visitor_path_is_recovered():
    result = derive_water_features(frame(garden_records()))
    garden = result.loc[result.kind.eq('garden')].iloc[0]
    assert garden.public
    assert garden.access_reason == 'named_garden_with_path'
    assert 'garden' in set(candidate_points(result).kind)
    cells = gpd.GeoDataFrame(geometry=[box(-20, -20, 120, 120)], crs=2056)
    assert aggregate_osm_features(cells, result).osm_garden_area_m2.iloc[0] == pytest.approx(10000)


@pytest.mark.parametrize('path', [
    LineString([(-10, 0), (110, 0)]),
    LineString([(-10, -5), (110, -5)]),
])
def test_boundary_or_nearby_path_does_not_confirm_garden_access(path):
    records = garden_records()
    records[1]['geometry'] = path
    result = derive_water_features(frame(records))
    assert not result.kind.eq('garden').any()


def test_private_path_does_not_confirm_garden_access():
    records = garden_records()
    records[1]['public'] = False
    records[1]['access_reason'] = 'restricted_access'
    assert derive_water_features(frame(records)).empty


def test_path_inside_residential_grounds_does_not_confirm_named_garden():
    records = garden_records()
    records.append(dict(osm_id='residential', kind='garden', public=False, name='Villa',
                        access_reason='residential_grounds', geometry=box(-10, -10, 110, 110)))
    assert derive_water_features(frame(records)).empty


@pytest.mark.parametrize('kind_tags', [
    {'aerialway': 'chair_lift'}, {'aerialway': 'station'}, {'piste:type': 'downhill'},
])
def test_explicit_ski_permission_can_override_pedestrian_prohibition(kind_tags):
    assert public_access(dict(kind_tags, foot='no', ski='yes'))
    assert not public_access(dict(kind_tags, foot='no', ski='no'))
    assert not public_access(dict(kind_tags, foot='no', ski='yes', access='private', membership='yes'))


def test_stage_recovers_named_garden_and_visitor_stations_with_provenance(tmp_path):
    from data.osm.secondary_features import execute
    source = tmp_path / 'osm' / 'switzerland-visitors.osm'
    source.parent.mkdir()
    source.write_text('''<osm version="0.6">
<node id="1" lon="8.50" lat="47.30"/><node id="2" lon="8.51" lat="47.30"/>
<node id="3" lon="8.51" lat="47.31"/><node id="4" lon="8.50" lat="47.31"/>
<node id="5" lon="8.501" lat="47.305"/><node id="6" lon="8.509" lat="47.305"/>
<node id="7" lon="8.52" lat="47.32"><tag k="aerialway" v="station"/><tag k="fee" v="yes"/></node>
<node id="8" lon="8.53" lat="47.33"><tag k="amenity" v="place_of_worship"/></node>
<node id="9" lon="8.54" lat="47.34"><tag k="aerialway" v="station"/><tag k="access" v="customers"/></node>
<node id="10" lon="8.55" lat="47.35"><tag k="aerialway" v="station"/><tag k="access" v="private"/></node>
<way id="20"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
<tag k="leisure" v="garden"/><tag k="name" v="Jardin de test"/></way>
<way id="21"><nd ref="5"/><nd ref="6"/><tag k="highway" v="footway"/></way>
</osm>''')
    class Context:
        def config(self, key):
            return {'data_path': str(tmp_path), 'osm_file': source.name,
                    'secondary_nn_osm_water_path_buffer_m': 30}[key]
        def path(self):
            return str(tmp_path)
    result = execute(Context())
    assert set(result.osm_id) == {'n7', 'n8', 'n9', 'w20', 'w21'}
    assert result.public.all()
    saved = gpd.read_parquet(tmp_path / 'secondary_features.parquet').set_index('osm_id')
    assert saved.loc['w20', 'access_reason'] == 'named_garden_with_path'
    assert saved.loc['n9', 'access'] == 'customers'
    assert saved.loc['n9', 'access_reason'] == 'visitor_customers'
    assert set(candidate_points(result).kind) == {'cable_station', 'worship', 'garden'}


@pytest.mark.parametrize('public_flag', [False, True, None])
def test_model_consumers_use_stage_eligibility_without_refiltering_public(public_flag):
    retained = frame([
        dict(osm_id='station', kind='cable_station', name='Station', public=public_flag,
             geometry=Point(20, 20)),
        dict(osm_id='church', kind='worship', name='Church', public=public_flag,
             geometry=Point(80, 80)),
    ])
    cells = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=2056)
    expected = aggregate_osm_features(cells, retained)
    assert expected.osm_cable_station_count.iloc[0] == 1
    assert expected.osm_worship_count.iloc[0] == 1
    assert set(candidate_points(retained).kind) == {'cable_station', 'worship'}
    without_flag = retained.drop(columns='public')
    pd.testing.assert_frame_equal(aggregate_osm_features(cells, without_flag), expected)
    assert set(candidate_points(without_flag).kind) == {'cable_station', 'worship'}
