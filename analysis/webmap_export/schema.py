"""DuckDB schema (v1) for the webmap-postprocess output.

Single source of truth for table definitions. Indexes are kept separate from
CREATE TABLE so they can be added AFTER bulk inserts (5-10x faster).

Schema-version bumps require a coordinated change in webmap-backend.
"""

from __future__ import annotations

SCHEMA_VERSION = "v1"


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
    hilbert_idx           UBIGINT
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
    PRIMARY KEY (person_id, trip_index)
);
"""


def _demo_grid_ddl(resolution_m: int) -> str:
    cols = ",\n    ".join(DEMO_BASE_COLS)
    return f"""
CREATE TABLE demo_grid_{resolution_m}m (
    cell_id               UBIGINT PRIMARY KEY,
    cell_geom             GEOMETRY,
    cell_center           GEOMETRY,
    {cols}
);
"""


def _trip_grid_origin_ddl(resolution_m: int) -> str:
    cols = ",\n    ".join(TRIP_AGG_COLS)
    return f"""
CREATE TABLE trip_grid_origin_{resolution_m}m (
    cell_id                UBIGINT PRIMARY KEY,
    cell_geom              GEOMETRY,
    {cols}
);
"""


def _flow_grid_ddl(resolution_m: int) -> str:
    return f"""
CREATE TABLE flow_grid_{resolution_m}m (
    origin_cell_id        UBIGINT NOT NULL,
    dest_cell_id          UBIGINT NOT NULL,
    origin_cell_geom      GEOMETRY,
    dest_cell_geom        GEOMETRY,
    n_trips               INTEGER NOT NULL,
    mode_car              INTEGER,
    mode_pt               INTEGER,
    mode_walk             INTEGER,
    mode_bike             INTEGER,
    mode_car_passenger    INTEGER,
    PRIMARY KEY (origin_cell_id, dest_cell_id)
);
"""


def _out_of_home_grid_ddl(resolution_m: int) -> str:
    cols = ",\n    ".join(OUT_OF_HOME_AGG_COLS)
    return f"""
CREATE TABLE out_of_home_grid_{resolution_m}m (
    cell_id               UBIGINT PRIMARY KEY,
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
    geom                  GEOMETRY
);
"""

DDL_NETWORK_NODES = """
CREATE TABLE network_nodes (
    node_id               VARCHAR PRIMARY KEY,
    geom                  GEOMETRY
);
"""

DDL_LINK_SPEEDS = """
CREATE TABLE link_speeds (
    link_id               VARCHAR NOT NULL,
    time_bucket           INTEGER NOT NULL,
    speed                 DOUBLE,
    PRIMARY KEY (link_id, time_bucket)
);
"""

DDL_STATIC_ASSETS = """
CREATE TABLE static_assets (
    key                   VARCHAR PRIMARY KEY,
    content_type          VARCHAR NOT NULL,
    payload               BLOB
);
"""


INDEX_DDL_COMMON = [
    "CREATE INDEX rtree_persons_home ON persons USING RTREE (home_pt)",
    "CREATE INDEX rtree_activities_loc ON activities USING RTREE (location_pt)",
    "CREATE INDEX idx_activities_purpose ON activities (purpose)",
    "CREATE INDEX rtree_trips_origin ON trips USING RTREE (origin_pt)",
    "CREATE INDEX rtree_trips_dest ON trips USING RTREE (dest_pt)",
    "CREATE INDEX idx_trips_mode ON trips (main_mode)",
    "CREATE INDEX idx_hot_polygons_type ON hot_polygons (polygon_type)",
    "CREATE INDEX rtree_hot_polygons_geom ON hot_polygons USING RTREE (polygon_geom)",
]

INDEX_DDL_DEMO_GRID_TPL = "CREATE INDEX rtree_demo_grid_{r}m ON demo_grid_{r}m USING RTREE (cell_geom)"
INDEX_DDL_TRIP_GRID_TPL = "CREATE INDEX rtree_trip_grid_origin_{r}m ON trip_grid_origin_{r}m USING RTREE (cell_geom)"
INDEX_DDL_OOH_GRID_TPL = "CREATE INDEX rtree_oh_grid_{r}m ON out_of_home_grid_{r}m USING RTREE (cell_geom)"

INDEX_DDL_FLOW_TPL = [
    "CREATE INDEX rtree_flow_origin_{r}m ON flow_grid_{r}m USING RTREE (origin_cell_geom)",
    "CREATE INDEX rtree_flow_dest_{r}m  ON flow_grid_{r}m USING RTREE (dest_cell_geom)",
]

INDEX_DDL_SYNTHETIC_ONLY = [
    "CREATE INDEX idx_spider_link      ON spider_link_index (link_id)",
    "CREATE INDEX idx_spider_link_trip ON spider_link_index (person_id, trip_index)",
    "CREATE INDEX rtree_network_links ON network_links USING RTREE (geom)",
    "CREATE INDEX rtree_network_nodes ON network_nodes USING RTREE (geom)",
]


GRID_RESOLUTIONS_M = [100, 500, 5000]
TRIP_GRID_RESOLUTION_M = 500
FLOW_GRID_RESOLUTION_M = 500
OUT_OF_HOME_GRID_RESOLUTION_M = 500


def create_all_tables(db, source_type: str) -> None:
    """Create every table required for ``source_type`` ('synthetic'|'microcensus').

    Idempotent only on a fresh DB — caller must drop/recreate the file first.
    """
    if source_type not in {"synthetic", "microcensus"}:
        raise ValueError(f"Unknown source_type: {source_type!r}")

    db.execute(DDL_METADATA)
    db.execute(DDL_PERSONS)
    db.execute(DDL_HOUSEHOLDS)
    db.execute(DDL_ACTIVITIES)
    db.execute(DDL_TRIPS)

    for r in GRID_RESOLUTIONS_M:
        db.execute(_demo_grid_ddl(r))

    db.execute(_trip_grid_origin_ddl(TRIP_GRID_RESOLUTION_M))
    db.execute(_out_of_home_grid_ddl(OUT_OF_HOME_GRID_RESOLUTION_M))

    db.execute(DDL_HOT_POLYGONS)
    db.execute(_hot_polygon_demo_ddl())
    db.execute(_hot_polygon_trips_ddl())
    db.execute(_hot_polygon_out_of_home_ddl())

    db.execute(DDL_STATIC_ASSETS)

    if source_type == "synthetic":
        db.execute(_flow_grid_ddl(FLOW_GRID_RESOLUTION_M))
        db.execute(DDL_HOT_POLYGON_FLOWS)
        db.execute(DDL_SPIDER_ROUTES)
        db.execute(DDL_SPIDER_LINK_INDEX)
        db.execute(DDL_NETWORK_LINKS)
        db.execute(DDL_NETWORK_NODES)
        db.execute(DDL_LINK_SPEEDS)


def create_all_indexes(db, source_type: str) -> None:
    """Run AFTER all bulk inserts. Splitting reduces build time 5-10x."""
    for stmt in INDEX_DDL_COMMON:
        db.execute(stmt)
    for r in GRID_RESOLUTIONS_M:
        db.execute(INDEX_DDL_DEMO_GRID_TPL.format(r=r))
    db.execute(INDEX_DDL_TRIP_GRID_TPL.format(r=TRIP_GRID_RESOLUTION_M))
    db.execute(INDEX_DDL_OOH_GRID_TPL.format(r=OUT_OF_HOME_GRID_RESOLUTION_M))
    if source_type == "synthetic":
        for tpl in INDEX_DDL_FLOW_TPL:
            db.execute(tpl.format(r=FLOW_GRID_RESOLUTION_M))
        for stmt in INDEX_DDL_SYNTHETIC_ONLY:
            db.execute(stmt)


EXPECTED_TABLES_SYNTHETIC = {
    "metadata", "persons", "households", "activities", "trips",
    "demo_grid_100m", "demo_grid_500m", "demo_grid_5000m",
    "trip_grid_origin_500m", "flow_grid_500m", "out_of_home_grid_500m",
    "hot_polygons", "hot_polygon_demo", "hot_polygon_trips",
    "hot_polygon_out_of_home", "hot_polygon_flows",
    "spider_routes", "spider_link_index",
    "network_links", "network_nodes",
    "static_assets",

}

EXPECTED_TABLES_MICROCENSUS = EXPECTED_TABLES_SYNTHETIC - {
    "flow_grid_500m", "hot_polygon_flows",
    "spider_routes", "spider_link_index",
    "network_links", "network_nodes",
}


def _col_names(ddl_cols: list[str]) -> set[str]:
    return {line.strip().split()[0] for line in ddl_cols}


_DEMO_COLS = _col_names(DEMO_BASE_COLS) | {"cell_id", "cell_geom", "cell_center"}
_TRIP_COLS = _col_names(TRIP_AGG_COLS) | {"cell_id", "cell_geom"}
_OOH_COLS = _col_names(OUT_OF_HOME_AGG_COLS) | {"cell_id", "cell_geom"}

EXPECTED_COLUMNS = {
    "demo_grid_100m": _DEMO_COLS,
    "demo_grid_500m": _DEMO_COLS,
    "demo_grid_5000m": _DEMO_COLS,
    "trip_grid_origin_500m": _TRIP_COLS,
    "out_of_home_grid_500m": _OOH_COLS,
    "hot_polygon_demo": _col_names(DEMO_BASE_COLS) | {"polygon_id"},
    "hot_polygon_trips": _col_names(TRIP_AGG_COLS) | {"polygon_id"},
    "hot_polygon_out_of_home": _col_names(OUT_OF_HOME_AGG_COLS) | {"polygon_id"},
    "static_assets": {"key", "content_type", "payload"},
}
