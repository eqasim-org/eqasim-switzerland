"""Reviewed ASTRA measurement branches, anchored to OSM ways rather than MATSim IDs.

Positions are EPSG:2056 sensor-group centres from ASTRA's SVZ sensor inventory.
Angles are local travel directions, counter-clockwise from east. One entry
represents one measured carriageway/branch, regardless of its number of lanes.
Source: https://services3.arcgis.com/IXAVBLfeIsfhE7s9/arcgis/rest/services/SVZ_2022/FeatureServer/0
Inventory reviewed against the 2025 station list on 2026-11-18.

The full downloaded inventory and CSV/GeoPackage group exports are stored under
counts_path/Switzerland/astra_sensors. Sensor IDs document lane membership; they
never multiply the simulated flow of a group. The Python dictionary is the
runtime registry; changing the exported review files alone does not update it.
"""

# Reintroduced from the original junction blacklist using hourly GEH < 6 on
# rda_not_finished iteration 100 (weekdays, 10% sample), reviewed 2026-11-18.
# The other reviewed anchors remain below for diagnostics, but their stations
# stay excluded: 68, 83, 112, 277, 283, 296, 577, 675, 799, 847.
# This fixed selection also applies before simulation in counts.target.
REINSTATED_STATION_IDS = frozenset({
    "6", "35", "123", "157", "285", "297", "507", "598", "618", "760", "852",
})

STATION_OSM_OVERRIDES = {
    225: [
        dict(group="bern", osm_id="290361969", x=2627767.399, y=1240563.993, angle=-91.8, direction=1),
        dict(group="luzern", osm_id="31240849", x=2627775.967, y=1240605.395, angle=-87.4, direction=1),
        dict(group="augst_a", osm_id="36793094", x=2627791.011, y=1240643.491, angle=95.4, direction=2),
        dict(group="augst_b", osm_id="202019938", x=2627799.701, y=1240645.677, angle=99.1, direction=2),
    ],
    525: [
        dict(group="minusio_main", osm_id="186489037", x=2708479.861, y=1115057.965, angle=148.8, direction=1),
        dict(group="minusio_exit", osm_id="23606710", x=2708500.676, y=1115060.529, angle=135.9, direction=1),
        dict(group="bellinzona_main", osm_id="25803307", x=2708477.503, y=1115053.659, angle=-35.0, direction=2),
        dict(group="bellinzona_entry", osm_id="36920756", x=2708475.965, y=1115042.483, angle=-30.7, direction=2),
    ],
    547: [
        dict(group="brissago_main", osm_id="246315226", x=2703104.889, y=1113157.645, angle=159.1, direction=1),
        dict(group="brissago_entry", osm_id="19881539", x=2703108.182, y=1113167.403, angle=-179.5, direction=1),
        dict(group="ascona_main", osm_id="186449621", x=2703103.303, y=1113152.156, angle=-19.9, direction=2),
        dict(group="ascona_exit", osm_id="29166639", x=2703100.498, y=1113143.130, angle=-21.1, direction=2),
    ],
    24: [
        # Lausanne's detector lies at a link boundary. Select this one OSM
        # segment, never both serial links on the same carriageway.
        dict(group="lausanne", osm_id="986058990", x=2511971.282, y=1144266.110, angle=28.9, direction=1),
        dict(group="geneve", osm_id="486357475", x=2511966.648, y=1144277.593, angle=-150.9, direction=2),
    ],

    # Additional sensor footprints reviewed against the simulation network.
    6: [
        dict(group='fr1_s1_2', osm_id='42704496', x=2613965.6, y=1265061.65,
             angle=-93.3, direction=1, sensor_ids=[1, 2]),
        dict(group='fr2_s3_4', osm_id='249164106', x=2613945.7, y=1264847.05,
             angle=77.7, direction=2, sensor_ids=[3, 4]),
        dict(group='fr2_s5', osm_id='334187943', x=2613951.3, y=1264844.9,
             angle=75.8, direction=2, sensor_ids=[5]),
    ],
    35: [
        dict(group='fr1_s1', osm_id='35006808', x=2753789.396, y=1211735.591,
             angle=-139.4, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2_3', osm_id='282383634', x=2753818.334, y=1211744.897,
             angle=-120.3, direction=1, sensor_ids=[2, 3]),
        dict(group='fr2_s4', osm_id='319833425', x=2753828.181, y=1211740.754,
             angle=60.1, direction=2, sensor_ids=[4]),
        dict(group='fr2_s5', osm_id='367293011', x=2753840.909, y=1211720.101,
             angle=77.8, direction=2, sensor_ids=[5]),
    ],
    68: [
        dict(group='fr1_s1_2', osm_id='75795026', x=2499832.399, y=1123325.689,
             angle=120.2, direction=1, sensor_ids=[1, 2]),
        dict(group='fr1_s3_4', osm_id='1351300781', x=2499793.598, y=1123305.859,
             angle=128.5, direction=1, sensor_ids=[3, 4]),
    ],
    83: [
        dict(group='fr1_s1_2', osm_id='186499465', x=2560465.07, y=1136989.917,
             angle=-118.3, direction=1, sensor_ids=[1, 2]),
        dict(group='fr2_s3_4', osm_id='486858469', x=2560476.463, y=1136986.739,
             angle=61.7, direction=2, sensor_ids=[3, 4]),
    ],
    112: [
        dict(group='fr1_s1_2', osm_id='25648976', x=2569018.643, y=1207849.233,
             angle=0.8, direction=1, sensor_ids=[1, 2]),
        dict(group='fr2_s3_4', osm_id='1338123495', x=2569018.673, y=1207855.733,
             angle=179.3, direction=2, sensor_ids=[3, 4]),
    ],
    123: [
        dict(group='fr1_s1', osm_id='1262774081', x=2690864.902, y=1285505.948,
             angle=-142.3, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2', osm_id='254701827', x=2690886.239, y=1285492.54,
             angle=-139.6, direction=1, sensor_ids=[2]),
        dict(group='fr2_s4', osm_id='46699597', x=2690916.67, y=1285508.63,
             angle=45.7, direction=2, sensor_ids=[4]),
        dict(group='fr2_s6', osm_id='1262774079', x=2690933.284, y=1285506.181,
             angle=27.7, direction=2, sensor_ids=[6]),
    ],
    157: [
        dict(group='fr1_s1', osm_id='40588769', x=2759637.724, y=1172126.446,
             angle=-2.8, direction=1, sensor_ids=[1]),
        dict(group='fr2_s2', osm_id='40588769', x=2759637.882, y=1172129.991,
             angle=177.2, direction=2, sensor_ids=[2]),
    ],
    241: [
        dict(group='fr1_s4_5', osm_id='43169923', x=2685506.317, y=1253834.337,
             angle=98.0, direction=1, sensor_ids=[4, 5]),
        dict(group='fr1_s6', osm_id='43169924', x=2685514.765, y=1253847.15,
             angle=106.6, direction=1, sensor_ids=[6]),
        dict(group='fr2_s1', osm_id='43169920', x=2685497.388, y=1253753.859,
             angle=-90.4, direction=2, sensor_ids=[1]),
        dict(group='fr2_s2_3', osm_id='23800819', x=2685513.003, y=1253725.992,
             angle=-80.4, direction=2, sensor_ids=[2, 3]),
    ],
    277: [
        dict(group='fr1_s1_2', osm_id='25263418', x=2613228.377, y=1266358.65,
             angle=-112.0, direction=1, sensor_ids=[1, 2]),
        dict(group='fr1_s3_4', osm_id='25124306', x=2613244.027, y=1266361.419,
             angle=-96.3, direction=1, sensor_ids=[3, 4]),
        dict(group='fr2_s5', osm_id='411825448', x=2613276.171, y=1266389.048,
             angle=62.6, direction=2, sensor_ids=[5]),
        dict(group='fr2_s6_7', osm_id='22948918', x=2613289.173, y=1266384.353,
             angle=75.7, direction=2, sensor_ids=[6, 7]),
    ],
    283: [
        dict(group='fr1_s1', osm_id='127376074', x=2610844.767, y=1268889.707,
             angle=11.7, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2_3', osm_id='185448041', x=2610844.362, y=1268898.572,
             angle=-0.6, direction=1, sensor_ids=[2, 3]),
        dict(group='fr2_s4_5', osm_id='26496237', x=2610844.159, y=1268907.437,
             angle=179.9, direction=2, sensor_ids=[4, 5]),
        dict(group='fr2_s6', osm_id='127376072', x=2610844.159, y=1268917.569,
             angle=175.7, direction=2, sensor_ids=[6]),
    ],
    285: [
        dict(group='fr1_s1', osm_id='224927551', x=2598440.994, y=1200536.451,
             angle=53.7, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2_3_4', osm_id='31240910', x=2598433.375, y=1200543.223,
             angle=48.5, direction=1, sensor_ids=[2, 3, 4]),
        dict(group='fr2_s5_6', osm_id='5119912', x=2598391.83, y=1200515.222,
             angle=-129.7, direction=2, sensor_ids=[5, 6]),
        dict(group='fr2_s7_8', osm_id='4223413', x=2598386.396, y=1200520.25,
             angle=-135.8, direction=2, sensor_ids=[7, 8]),
    ],
    296: [
        dict(group='fr1_s1', osm_id='39465579', x=2665203.755, y=1207874.28,
             angle=-41.4, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2_3', osm_id='23982827', x=2665179.007, y=1207952.969,
             angle=-64.3, direction=1, sensor_ids=[2, 3]),
        dict(group='fr2_s4_5', osm_id='1298035280', x=2665237.666, y=1207865.399,
             angle=121.2, direction=2, sensor_ids=[4, 5]),
        dict(group='fr2_s6', osm_id='298350588', x=2665224.799, y=1207917.171,
             angle=127.5, direction=2, sensor_ids=[6]),
    ],
    297: [
        dict(group='fr1_s1_2', osm_id='410357297', x=2614511.555, y=1265850.915,
             angle=-109.3, direction=1, sensor_ids=[1, 2]),
        dict(group='fr2_s3_4', osm_id='185428778', x=2614519.911, y=1265848.826,
             angle=70.6, direction=2, sensor_ids=[3, 4]),
    ],
    507: [
        dict(group='fr1_s1', osm_id='551847462', x=2715618.526, y=1094050.316,
             angle=-54.4, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2_3', osm_id='551852383', x=2715622.702, y=1094054.384,
             angle=-49.2, direction=1, sensor_ids=[2, 3]),
        dict(group='fr2_s4_5', osm_id='983284034', x=2715722.54, y=1093947.363,
             angle=119.6, direction=2, sensor_ids=[4, 5]),
        dict(group='fr2_s6', osm_id='983284036', x=2715727.77, y=1093950.505,
             angle=120.5, direction=2, sensor_ids=[6]),
    ],
    577: [
        dict(group='fr1_s4_5_6', osm_id='137477230', x=2677950.762, y=1244833.642,
             angle=8.4, direction=1, sensor_ids=[4, 5, 6]),
        dict(group='fr2_s1_2_3', osm_id='137477232', x=2677949.358, y=1244864.186,
             angle=-170.7, direction=2, sensor_ids=[1, 2, 3]),
    ],
    598: [
        dict(group='fr1_s1_2', osm_id='26551667', x=2751069.178, y=1259715.736,
             angle=-81.7, direction=1, sensor_ids=[1, 2]),
        dict(group='fr1_s3_4', osm_id='495021994', x=2751077.074, y=1259720.161,
             angle=-77.3, direction=1, sensor_ids=[3, 4]),
        dict(group='fr2_s5_6', osm_id='26560377', x=2751103.731, y=1259764.754,
             angle=129.5, direction=2, sensor_ids=[5, 6]),
        dict(group='fr2_s7', osm_id='225593197', x=2751110.376, y=1259770.42,
             angle=133.1, direction=2, sensor_ids=[7]),
    ],
    618: [
        dict(group='fr1_s1', osm_id='43022983', x=2677527.935, y=1228338.12,
             angle=-139.1, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2_3', osm_id='23749643', x=2677557.169, y=1228324.236,
             angle=-132.5, direction=1, sensor_ids=[2, 3]),
        dict(group='fr2_s4_5', osm_id='24933385', x=2677564.881, y=1228315.801,
             angle=47.3, direction=2, sensor_ids=[4, 5]),
        dict(group='fr2_s6', osm_id='554480402', x=2677580.889, y=1228294.892,
             angle=71.7, direction=2, sensor_ids=[6]),
    ],
    675: [
        dict(group='fr1_s1_2', osm_id='37323361', x=2559776.928, y=1203740.754,
             angle=51.8, direction=1, sensor_ids=[1, 2]),
        dict(group='fr2_s3_4', osm_id='35534914', x=2559766.455, y=1203746.061,
             angle=-127.6, direction=2, sensor_ids=[3, 4]),
    ],
    760: [
        dict(group='fr1_s1', osm_id='127514824', x=2698599.5, y=1244710.3,
             angle=-87.4, direction=1, sensor_ids=[1]),
        dict(group='fr2_s2_3', osm_id='139671080', x=2698604.95, y=1244710.8,
             angle=94.1, direction=2, sensor_ids=[2, 3]),
    ],
    799: [
        dict(group='fr1_s1_2', osm_id='139016175', x=2558302.791, y=1203218.866,
             angle=18.7, direction=1, sensor_ids=[1, 2]),
        dict(group='fr2_s3_4', osm_id='139020383', x=2558298.873, y=1203230.484,
             angle=-158.6, direction=2, sensor_ids=[3, 4]),
    ],
    847: [
        dict(group='fr1_s1', osm_id='749146263', x=2687839.3, y=1282955.1,
             angle=-42.3, direction=1, sensor_ids=[1]),
        dict(group='fr2_s2', osm_id='749146262', x=2687843.4, y=1282957.9,
             angle=109.6, direction=2, sensor_ids=[2]),
    ],
    852: [
        dict(group='fr1_s1', osm_id='26578737', x=2573715.0, y=1107131.7,
             angle=29.7, direction=1, sensor_ids=[1]),
        dict(group='fr1_s2_3', osm_id='1146816825', x=2573711.95, y=1107136.5,
             angle=23.6, direction=1, sensor_ids=[2, 3]),
        dict(group='fr2_s4_5', osm_id='1023972692', x=2573706.15, y=1107146.75,
             angle=-156.4, direction=2, sensor_ids=[4, 5]),
        dict(group='fr2_s6', osm_id='1307429215', x=2573702.7, y=1107153.1,
             angle=-159.9, direction=2, sensor_ids=[6]),
    ],
}
