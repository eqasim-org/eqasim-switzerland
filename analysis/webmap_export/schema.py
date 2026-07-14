"""DuckDB schema for the webmap-postprocess output.

Schema-version bumps require a coordinated change in webmap-backend.
"""

from __future__ import annotations

SCHEMA_VERSION = "v2"

H3_RESOLUTIONS = (6, 9, 12)
H3_RES_COARSE, H3_RES_MID, H3_RES_FINE = H3_RESOLUTIONS


AGE_BUCKET_COLS = [
    "age_0_6        INTEGER",
    "age_6_15       INTEGER",
    "age_15_18      INTEGER",
    "age_18_24      INTEGER",
    "age_24_30      INTEGER",
    "age_30_45      INTEGER",
    "age_45_65      INTEGER",
    "age_65_80      INTEGER",
    "age_80_plus    INTEGER",
]

DEMO_BASE_COLS = [
    "n_persons             INTEGER NOT NULL",
    *AGE_BUCKET_COLS,
    "sex_male              INTEGER",
    "sex_female            INTEGER",
    "car_avail_always      INTEGER",
    "car_avail_sometimes   INTEGER",
    "car_avail_never       INTEGER",
    "has_driving_license   INTEGER",
    "employed              INTEGER",
    "subs_ga               INTEGER",
    "subs_halbtax          INTEGER",
    "subs_verbund          INTEGER",
    "subs_strecke          INTEGER",
    "subs_gleis7           INTEGER",
    "subs_junior           INTEGER",
    "subs_other            INTEGER",
    "cars_0                INTEGER",
    "cars_1                INTEGER",
    "cars_2                INTEGER",
    "cars_3_plus           INTEGER",
    "sum_activities        INTEGER",
]

TRIP_AGG_COLS = [
    "n_trips                INTEGER NOT NULL",
    "mode_car               INTEGER",
    "mode_pt                INTEGER",
    "mode_walk              INTEGER",
    "mode_bike              INTEGER",
    "mode_car_passenger     INTEGER",
    "purpose_home           INTEGER",
    "purpose_work           INTEGER",
    "purpose_education      INTEGER",
    "purpose_shop           INTEGER",
    "purpose_leisure        INTEGER",
    "purpose_other          INTEGER",
    "sum_network_distance   DOUBLE",
    "sum_crowfly_distance   DOUBLE",
    "dist_bucket_0_1km      INTEGER",
    "dist_bucket_1_3km      INTEGER",
    "dist_bucket_3_10km     INTEGER",
    "dist_bucket_10_30km    INTEGER",
    "dist_bucket_30_plus_km INTEGER",
    *[f"time_h{h:<2} INTEGER" for h in range(24)],
]

OUT_OF_HOME_AGG_COLS = [
    "n_persons INTEGER NOT NULL",
    *[f"away_h{h:<2} INTEGER" for h in range(24)],
]


DDL_METADATA = """
CREATE TABLE metadata (
    schema_version       VARCHAR NOT NULL,
    build_date           TIMESTAMP NOT NULL,
    source_type          VARCHAR NOT NULL,
    matsim_run_id        VARCHAR,
    eqasim_commit_hash   VARCHAR,
    person_count         BIGINT NOT NULL,
    trip_count           BIGINT NOT NULL,
    activity_count       BIGINT NOT NULL,
    grid_resolutions_m   INTEGER[] NOT NULL,
    bbox_lv95            DOUBLE[] NOT NULL,
    hot_polygon_types    VARCHAR[] NOT NULL,
    h3_resolutions       INTEGER[] NOT NULL,
    has_pt_static        BOOLEAN NOT NULL
);
"""

DDL_PERSONS = """
CREATE TABLE persons (
    person_id             BIGINT PRIMARY KEY,
    household_id          BIGINT,
    age                   INTEGER,
    sex                   INTEGER,
    car_availability      VARCHAR,
    has_driving_license   BOOLEAN,
    employed              BOOLEAN,
    subscriptions_ga      BOOLEAN,
    subscriptions_halbtax BOOLEAN,
    subscriptions_verbund BOOLEAN,
    subscriptions_strecke BOOLEAN,
    subscriptions_gleis7  BOOLEAN,
    subscriptions_junior  BOOLEAN,
    subscriptions_other   BOOLEAN,
    canton_id             INTEGER,
    n_activities          INTEGER,
    home_pt               GEOMETRY,
    hilbert_idx           UBIGINT,
    home_h3_res12         BIGINT,
    home_h3_res9          BIGINT,
    home_h3_res6          BIGINT
);
"""

DDL_HOUSEHOLDS = """
CREATE TABLE households (
    household_id          BIGINT PRIMARY KEY,
    income_class          VARCHAR,
    n_cars_class          VARCHAR,
    n_bikes_class         VARCHAR,
    ovgk                  VARCHAR
);
"""

DDL_ACTIVITIES = """
CREATE TABLE activities (
    person_id             BIGINT NOT NULL,
    activity_index        INTEGER NOT NULL,
    purpose               VARCHAR,
    start_time            DOUBLE,
    end_time              DOUBLE,
    is_first              BOOLEAN,
    is_last               BOOLEAN,
    location_pt           GEOMETRY,
    canton_id             INTEGER,
    PRIMARY KEY (person_id, activity_index)
);
"""

DDL_TRIPS = """
CREATE TABLE trips (
    person_id             BIGINT NOT NULL,
    trip_index            INTEGER NOT NULL,
    departure_time        DOUBLE,
    travel_time           DOUBLE,
    main_mode             VARCHAR,
    preceding_purpose     VARCHAR,
    following_purpose     VARCHAR,
    network_distance      DOUBLE,
    crowfly_distance      DOUBLE,
    origin_pt             GEOMETRY,
    dest_pt               GEOMETRY,
    hilbert_origin        UBIGINT,
    origin_h3_res9        BIGINT,
    dest_h3_res9          BIGINT,
    origin_h3_res6        BIGINT,
    dest_h3_res6          BIGINT,
    origin_canton_id      INTEGER,
    dest_canton_id        INTEGER,
    PRIMARY KEY (person_id, trip_index)
);
"""


def _demo_hex_ddl(res: int) -> str:
    """DDL for demo_hex_res<res>; the coarsest resolution has no parent columns."""
    cols = ",\n    ".join(DEMO_BASE_COLS)
    parent_cols = ""
    if res == H3_RES_FINE:
        parent_cols = (
            "    h3_parent_res9        BIGINT NOT NULL,\n"
            "    h3_parent_res6        BIGINT NOT NULL,\n"
        )
    elif res == H3_RES_MID:
        parent_cols = "    h3_parent_res6        BIGINT NOT NULL,\n"
    return f"""
CREATE TABLE demo_hex_res{res} (
    h3_index              BIGINT PRIMARY KEY,
{parent_cols}    cell_geom             GEOMETRY,
    cell_center           GEOMETRY,
    {cols}
);
"""


def _trip_hex_origin_ddl() -> str:
    cols = ",\n    ".join(TRIP_AGG_COLS)
    return f"""
CREATE TABLE trip_hex_origin_res{H3_RES_MID} (
    h3_index              BIGINT PRIMARY KEY,
    h3_parent_res6        BIGINT NOT NULL,
    cell_geom             GEOMETRY,
    {cols}
);
"""


def _flow_hex_ddl() -> str:
    return f"""
CREATE TABLE flow_hex_res{H3_RES_MID} (
    origin_h3_index       BIGINT NOT NULL,
    dest_h3_index         BIGINT NOT NULL,
    origin_cell_geom      GEOMETRY,
    dest_cell_geom        GEOMETRY,
    n_trips               INTEGER NOT NULL,
    mode_car              INTEGER,
    mode_pt               INTEGER,
    mode_walk             INTEGER,
    mode_bike             INTEGER,
    mode_car_passenger    INTEGER,
    PRIMARY KEY (origin_h3_index, dest_h3_index)
);
"""


def _oh_hex_ddl() -> str:
    cols = ",\n    ".join(OUT_OF_HOME_AGG_COLS)
    return f"""
CREATE TABLE oh_hex_res{H3_RES_MID} (
    h3_index              BIGINT PRIMARY KEY,
    h3_parent_res6        BIGINT NOT NULL,
    cell_geom             GEOMETRY,
    {cols}
);
"""


DDL_HOT_POLYGONS = """
CREATE TABLE hot_polygons (
    polygon_id            VARCHAR PRIMARY KEY,
    polygon_type          VARCHAR NOT NULL,
    polygon_name          VARCHAR NOT NULL,
    parent_id             VARCHAR,
    polygon_geom          GEOMETRY
);
"""


def _hot_polygon_demo_ddl() -> str:
    cols = ",\n    ".join(DEMO_BASE_COLS)
    return f"""
CREATE TABLE hot_polygon_demo (
    polygon_id            VARCHAR PRIMARY KEY,
    {cols}
);
"""


def _hot_polygon_trips_ddl() -> str:
    cols = ",\n    ".join(TRIP_AGG_COLS)
    return f"""
CREATE TABLE hot_polygon_trips (
    polygon_id            VARCHAR PRIMARY KEY,
    {cols}
);
"""


def _hot_polygon_out_of_home_ddl() -> str:
    cols = ",\n    ".join(OUT_OF_HOME_AGG_COLS)
    return f"""
CREATE TABLE hot_polygon_out_of_home (
    polygon_id            VARCHAR PRIMARY KEY,
    {cols}
);
"""


DDL_HOT_POLYGON_FLOWS = """
CREATE TABLE hot_polygon_flows (
    origin_polygon_id     VARCHAR NOT NULL,
    dest_polygon_id       VARCHAR NOT NULL,
    n_trips               INTEGER NOT NULL,
    mode_car              INTEGER,
    mode_pt               INTEGER,
    mode_walk             INTEGER,
    mode_bike             INTEGER,
    mode_car_passenger    INTEGER,
    PRIMARY KEY (origin_polygon_id, dest_polygon_id)
);
"""

DDL_SPIDER_ROUTES = """
CREATE TABLE spider_routes (
    person_id             BIGINT NOT NULL,
    trip_index            INTEGER NOT NULL,
    departure_time        DOUBLE,
    route_links           VARCHAR[],
    PRIMARY KEY (person_id, trip_index)
);
"""

DDL_SPIDER_LINK_INDEX = """
CREATE TABLE spider_link_index (
    link_id               VARCHAR NOT NULL,
    person_id             BIGINT NOT NULL,
    trip_index            INTEGER NOT NULL,
    departure_time        DOUBLE,
    position              INTEGER,
    route_length          INTEGER
);
"""

DDL_NETWORK_LINKS = """
CREATE TABLE network_links (
    link_id               VARCHAR PRIMARY KEY,
    from_node             VARCHAR,
    to_node               VARCHAR,
    length                DOUBLE,
    capacity              DOUBLE,
    freespeed             DOUBLE,
    permlanes             DOUBLE,
    modes                 VARCHAR,
    road_type             VARCHAR,
    canton_id             INTEGER,
    geom                  GEOMETRY
);
"""

DDL_NETWORK_NODES = """
CREATE TABLE network_nodes (
    node_id               VARCHAR PRIMARY KEY,
    canton_id             INTEGER,
    geom                  GEOMETRY
);
"""

# link_speeds: one row per (link_id, 15-min time bin); freespeed/road_type/canton_id
# are denormalized from network_links.
DDL_LINK_SPEEDS = """
CREATE TABLE link_speeds (
    link_id               VARCHAR NOT NULL,
    time_bin              INTEGER NOT NULL,
    avg_speed             DOUBLE,
    volume                INTEGER,
    freespeed             DOUBLE,
    road_type             VARCHAR,
    canton_id             INTEGER,
    PRIMARY KEY (link_id, time_bin)
);
"""

DDL_STATIC_ASSETS = """
CREATE TABLE static_assets (
    key                   VARCHAR PRIMARY KEY,
    content_type          VARCHAR NOT NULL,
    payload               BLOB
);
"""

DDL_SPIDER_LINK_VOLUMES_BY_HEX_R6 = """
CREATE TABLE spider_link_volumes_by_hex_res6 (
    home_h3_index   BIGINT NOT NULL,
    link_id         VARCHAR NOT NULL,
    n_traversals    INTEGER NOT NULL,
    PRIMARY KEY (home_h3_index, link_id)
);
"""

DDL_ZONE_FLOW_LINK_VOLUMES_HEX_R6 = """
CREATE TABLE zone_flow_link_volumes_hex_res6 (
    origin_h3_index BIGINT NOT NULL,
    dest_h3_index   BIGINT NOT NULL,
    link_id         VARCHAR NOT NULL,
    n_trips         INTEGER NOT NULL,
    PRIMARY KEY (origin_h3_index, dest_h3_index, link_id)
);
"""

DDL_NODE_FLOW_MATRIX = """
CREATE TABLE node_flow_matrix (
    node_id     VARCHAR NOT NULL,
    from_link   VARCHAR NOT NULL,
    to_link     VARCHAR NOT NULL,
    n_trips     INTEGER NOT NULL,
    PRIMARY KEY (node_id, from_link, to_link)
);
"""


INDEX_DDL_COMMON = [
    "CREATE INDEX rtree_persons_home ON persons USING RTREE (home_pt)",
    "CREATE INDEX rtree_activities_loc ON activities USING RTREE (location_pt)",
    "CREATE INDEX idx_activities_purpose ON activities (purpose)",
    "CREATE INDEX idx_activities_canton ON activities (canton_id)",
    "CREATE INDEX rtree_trips_origin ON trips USING RTREE (origin_pt)",
    "CREATE INDEX rtree_trips_dest ON trips USING RTREE (dest_pt)",
    "CREATE INDEX idx_trips_mode ON trips (main_mode)",
    "CREATE INDEX idx_hot_polygons_type ON hot_polygons (polygon_type)",
    "CREATE INDEX rtree_hot_polygons_geom ON hot_polygons USING RTREE (polygon_geom)",
]

INDEX_DDL_HEX = [
    f"CREATE INDEX rtree_demo_hex_res{H3_RES_FINE}    ON demo_hex_res{H3_RES_FINE} USING RTREE (cell_geom)",
    f"CREATE INDEX idx_demo_hex_res{H3_RES_FINE}_p9  ON demo_hex_res{H3_RES_FINE} (h3_parent_res9)",
    f"CREATE INDEX idx_demo_hex_res{H3_RES_FINE}_p6  ON demo_hex_res{H3_RES_FINE} (h3_parent_res6)",
    f"CREATE INDEX rtree_demo_hex_res{H3_RES_MID}     ON demo_hex_res{H3_RES_MID} USING RTREE (cell_geom)",
    f"CREATE INDEX idx_demo_hex_res{H3_RES_MID}_p6   ON demo_hex_res{H3_RES_MID} (h3_parent_res6)",
    f"CREATE INDEX rtree_demo_hex_res{H3_RES_COARSE}  ON demo_hex_res{H3_RES_COARSE} USING RTREE (cell_geom)",
    f"CREATE INDEX rtree_trip_hex_origin_res{H3_RES_MID} ON trip_hex_origin_res{H3_RES_MID} USING RTREE (cell_geom)",
    f"CREATE INDEX idx_trip_hex_origin_res{H3_RES_MID}_p6 ON trip_hex_origin_res{H3_RES_MID} (h3_parent_res6)",
    f"CREATE INDEX rtree_oh_hex_res{H3_RES_MID}        ON oh_hex_res{H3_RES_MID} USING RTREE (cell_geom)",
    f"CREATE INDEX idx_oh_hex_res{H3_RES_MID}_p6      ON oh_hex_res{H3_RES_MID} (h3_parent_res6)",
]

INDEX_DDL_FLOW_HEX = [
    f"CREATE INDEX rtree_flow_hex_origin ON flow_hex_res{H3_RES_MID} USING RTREE (origin_cell_geom)",
    f"CREATE INDEX rtree_flow_hex_dest   ON flow_hex_res{H3_RES_MID} USING RTREE (dest_cell_geom)",
]

INDEX_DDL_PERSONS_H3 = [
    f"CREATE INDEX idx_persons_h3_r{H3_RES_FINE} ON persons (home_h3_res{H3_RES_FINE})",
    f"CREATE INDEX idx_persons_h3_r{H3_RES_MID}  ON persons (home_h3_res{H3_RES_MID})",
    f"CREATE INDEX idx_persons_h3_r{H3_RES_COARSE}  ON persons (home_h3_res{H3_RES_COARSE})",
]

INDEX_DDL_TRIPS_H3 = [
    f"CREATE INDEX idx_trips_h3_origin       ON trips (origin_h3_res{H3_RES_MID})",
    f"CREATE INDEX idx_trips_h3_dest         ON trips (dest_h3_res{H3_RES_MID})",
    f"CREATE INDEX idx_trips_h3_origin_r6    ON trips (origin_h3_res{H3_RES_COARSE})",
    f"CREATE INDEX idx_trips_h3_dest_r6      ON trips (dest_h3_res{H3_RES_COARSE})",
    "CREATE INDEX idx_trips_origin_canton    ON trips (origin_canton_id)",
    "CREATE INDEX idx_trips_dest_canton      ON trips (dest_canton_id)",
]

INDEX_DDL_SYNTHETIC_ONLY = [
    "CREATE INDEX idx_spider_link      ON spider_link_index (link_id)",
    "CREATE INDEX idx_spider_link_trip ON spider_link_index (person_id, trip_index)",
    "CREATE INDEX rtree_network_links ON network_links USING RTREE (geom)",
    "CREATE INDEX idx_network_links_canton ON network_links (canton_id)",
    "CREATE INDEX idx_network_links_road_type ON network_links (road_type)",
    "CREATE INDEX rtree_network_nodes ON network_nodes USING RTREE (geom)",
    "CREATE INDEX idx_network_nodes_canton ON network_nodes (canton_id)",
    "CREATE INDEX idx_link_speeds_link ON link_speeds (link_id)",
    "CREATE INDEX idx_link_speeds_canton ON link_speeds (canton_id)",
    "CREATE INDEX idx_link_speeds_road_type ON link_speeds (road_type)",
    "CREATE INDEX idx_slvh6_home  ON spider_link_volumes_by_hex_res6 (home_h3_index)",
    "CREATE INDEX idx_slvh6_link  ON spider_link_volumes_by_hex_res6 (link_id)",
    "CREATE INDEX idx_zflvh_orig  ON zone_flow_link_volumes_hex_res6 (origin_h3_index)",
    "CREATE INDEX idx_zflvh_dest  ON zone_flow_link_volumes_hex_res6 (dest_h3_index)",
    "CREATE INDEX idx_nfm_node    ON node_flow_matrix (node_id)",
]


def create_all_tables(db, source_type: str) -> None:
    """Create every table for ``source_type``; requires a fresh DB (caller drops/recreates the file)."""
    if source_type not in {"synthetic", "microcensus"}:
        raise ValueError(f"Unknown source_type: {source_type!r}")

    db.execute(DDL_METADATA)
    db.execute(DDL_PERSONS)
    db.execute(DDL_HOUSEHOLDS)
    db.execute(DDL_ACTIVITIES)
    db.execute(DDL_TRIPS)

    for r in H3_RESOLUTIONS:
        db.execute(_demo_hex_ddl(r))

    db.execute(_trip_hex_origin_ddl())
    db.execute(_oh_hex_ddl())

    db.execute(DDL_HOT_POLYGONS)
    db.execute(_hot_polygon_demo_ddl())
    db.execute(_hot_polygon_trips_ddl())
    db.execute(_hot_polygon_out_of_home_ddl())

    db.execute(DDL_STATIC_ASSETS)

    if source_type == "synthetic":
        db.execute(_flow_hex_ddl())
        db.execute(DDL_HOT_POLYGON_FLOWS)
        db.execute(DDL_SPIDER_ROUTES)
        db.execute(DDL_SPIDER_LINK_INDEX)
        db.execute(DDL_NETWORK_LINKS)
        db.execute(DDL_NETWORK_NODES)
        db.execute(DDL_LINK_SPEEDS)
        db.execute(DDL_SPIDER_LINK_VOLUMES_BY_HEX_R6)
        db.execute(DDL_ZONE_FLOW_LINK_VOLUMES_HEX_R6)
        db.execute(DDL_NODE_FLOW_MATRIX)


def create_all_indexes(db, source_type: str) -> None:
    """Create all indexes; must run after all bulk inserts."""
    for stmt in INDEX_DDL_COMMON:
        db.execute(stmt)
    for stmt in INDEX_DDL_PERSONS_H3:
        db.execute(stmt)
    for stmt in INDEX_DDL_TRIPS_H3:
        db.execute(stmt)
    for stmt in INDEX_DDL_HEX:
        db.execute(stmt)
    if source_type == "synthetic":
        for stmt in INDEX_DDL_FLOW_HEX:
            db.execute(stmt)
        for stmt in INDEX_DDL_SYNTHETIC_ONLY:
            db.execute(stmt)


EXPECTED_TABLES_SYNTHETIC = {
    "metadata", "persons", "households", "activities", "trips",
    f"demo_hex_res{H3_RES_COARSE}",
    f"demo_hex_res{H3_RES_MID}",
    f"demo_hex_res{H3_RES_FINE}",
    f"trip_hex_origin_res{H3_RES_MID}",
    f"flow_hex_res{H3_RES_MID}",
    f"oh_hex_res{H3_RES_MID}",
    "hot_polygons", "hot_polygon_demo", "hot_polygon_trips",
    "hot_polygon_out_of_home", "hot_polygon_flows",
    "spider_routes", "spider_link_index",
    "network_links", "network_nodes",
    "static_assets",
    "spider_link_volumes_by_hex_res6",
    "zone_flow_link_volumes_hex_res6",
    "node_flow_matrix",
}

EXPECTED_TABLES_MICROCENSUS = EXPECTED_TABLES_SYNTHETIC - {
    f"flow_hex_res{H3_RES_MID}", "hot_polygon_flows",
    "spider_routes", "spider_link_index",
    "network_links", "network_nodes",
    "spider_link_volumes_by_hex_res6",
    "zone_flow_link_volumes_hex_res6",
    "node_flow_matrix",
}


def _col_names(ddl_cols: list[str]) -> set[str]:
    return {line.strip().split()[0] for line in ddl_cols}


_DEMO_HEX_BASE_COLS = _col_names(DEMO_BASE_COLS) | {"h3_index", "cell_geom", "cell_center"}
_DEMO_HEX_FINE_COLS = _DEMO_HEX_BASE_COLS | {"h3_parent_res9", "h3_parent_res6"}
_DEMO_HEX_MID_COLS = _DEMO_HEX_BASE_COLS | {"h3_parent_res6"}
_TRIP_HEX_COLS = _col_names(TRIP_AGG_COLS) | {"h3_index", "h3_parent_res6", "cell_geom"}
_OH_HEX_COLS = _col_names(OUT_OF_HOME_AGG_COLS) | {"h3_index", "h3_parent_res6", "cell_geom"}

EXPECTED_COLUMNS = {
    f"demo_hex_res{H3_RES_FINE}": _DEMO_HEX_FINE_COLS,
    f"demo_hex_res{H3_RES_MID}": _DEMO_HEX_MID_COLS,
    f"demo_hex_res{H3_RES_COARSE}": _DEMO_HEX_BASE_COLS,
    f"trip_hex_origin_res{H3_RES_MID}": _TRIP_HEX_COLS,
    f"oh_hex_res{H3_RES_MID}": _OH_HEX_COLS,
    "hot_polygon_demo": _col_names(DEMO_BASE_COLS) | {"polygon_id"},
    "hot_polygon_trips": _col_names(TRIP_AGG_COLS) | {"polygon_id"},
    "hot_polygon_out_of_home": _col_names(OUT_OF_HOME_AGG_COLS) | {"polygon_id"},
    "static_assets": {"key", "content_type", "payload"},
}
