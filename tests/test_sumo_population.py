"""Native SUMO demand contracts and focused integration with installed tools."""

import gzip
import os
from pathlib import Path
import pickle
import subprocess

from lxml import etree
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pyproj import Transformer
from shapely.geometry import Point
from synpp.pipeline import ConfigurationContext

from sumo.runtime import sumo
from sumo.scenario.network import road_network
from sumo.scenario.population import plans, population, tools, trips
from sumo.scenario.population.writer import PopulationWriter

ROOT = Path(__file__).resolve().parents[1]
OSM = '''<osm version="0.6">
<node id="1" lat="47" lon="8"/><node id="2" lat="47" lon="8.001"/>
<node id="3" lat="47" lon="8.002"/><node id="4" lat="47.001" lon="8.001"/>
<way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/></way>
<way id="11"><nd ref="2"/><nd ref="3"/><tag k="highway" v="residential"/></way>
<way id="12"><nd ref="2"/><nd ref="4"/><tag k="highway" v="residential"/></way>
</osm>'''


class Context:
    def __init__(self, module, path, config=None, stages=None):
        configured = ConfigurationContext({"sumo_modes": None, **(config or {})})
        module.configure(configured)
        self.values = configured.required_config
        self.stages = stages or {}
        self.cache = Path(path)
        self.cache.mkdir(parents=True, exist_ok=True)

    def config(self, key):
        return self.values[key]

    def path(self):
        return str(self.cache)

    def stage(self, key):
        return self.stages[key]


@pytest.fixture
def fixture_population():
    transform = Transformer.from_crs(4326, 2056, always_xy=True)
    home = Point(*transform.transform(8.0001, 47))
    work = Point(*transform.transform(8.0019, 47))
    persons = pd.DataFrame([
        {"person_id": 11, "age": 41, "home_x": home.x, "home_y": home.y, "household_id": 8},
        {"person_id": 12, "age": 70, "home_x": home.x, "home_y": home.y, "household_id": 8},
    ])
    activities = pd.DataFrame([
        {"person_id": 11, "activity_index": 0, "start_time": 0., "end_time": 8., "purpose": "home", "following_mode": "pt"},
        {"person_id": 11, "activity_index": 1, "start_time": 90., "end_time": 120., "purpose": "work", "following_mode": "car"},
        {"person_id": 11, "activity_index": 2, "start_time": 200., "end_time": 300., "purpose": "home"},
        {"person_id": 12, "activity_index": 0, "purpose": "home"},
    ], index=[100, 103, 108, 112])
    locations = pd.DataFrame([
        {"person_id": 11, "activity_index": 2, "geometry": home},
        {"person_id": 12, "activity_index": 0, "geometry": home},
        {"person_id": 11, "activity_index": 1, "geometry": work},
        {"person_id": 11, "activity_index": 0, "geometry": home},
    ])
    final_trips = pd.DataFrame([
        {"person_id": 11, "trip_index": 1, "departure_time": 140., "arrival_time": 200., "mode": "car", "trip_id": "return"},
        {"person_id": 11, "trip_index": 0, "departure_time": 10., "arrival_time": 90., "mode": "car", "trip_id": "outbound"},
    ])
    return persons, activities, locations, final_trips


def prepared(fixture_population):
    persons, activities, locations, final_trips = fixture_population
    return tools.combine_populations([(persons, tools.prepare_activities(activities, locations, final_trips), "resident")])


def read_xml(path):
    with gzip.open(path, "rb") as stream:
        return etree.parse(stream)


@pytest.fixture
def runtime():
    candidates = list((ROOT / "cache_sumo").glob("sumo.runtime.sumo__*.p"))
    if not candidates:
        pytest.skip("Install the isolated SUMO runtime in cache_sumo first")
    settings = pickle.loads(candidates[0].read_bytes())
    if not (Path(settings["home"]) / "bin/netconvert").is_file():
        pytest.skip("SUMO binaries are not available")
    return settings


def test_final_modes_departures_and_nontravellers_are_preserved(fixture_population, tmp_path):
    originals = [frame.copy(deep=True) for frame in fixture_population]
    result = prepared(fixture_population)
    activity = result["activities"].query("person_id == '11' and activity_index == 0").iloc[0]
    assert activity.following_mode == "car"
    assert activity.end_time == 10
    assert activity.arrival_time == 90
    assert activity.geometry.equals(fixture_population[2].query("person_id == 11 and activity_index == 0").iloc[0].geometry)
    context = Context(plans, tmp_path, stages={"sumo.scenario.population.population": result})
    root = read_xml(plans.execute(context)).getroot()
    assert [person.get("id") for person in root.findall("person")] == ["11", "12"]
    traveller = root.find("person[@id='11']")
    assert [stage.tag for stage in traveller if stage.tag != "param"] == ["stop", "personTrip", "stop", "personTrip", "stop"]
    assert traveller.find("stop").get("until") == "10.0"
    assert traveller.find("personTrip/param[@key='trip_id']").get("value") == "outbound"
    assert traveller.find("personTrip/param[@key='trip_index']").get("value") == "0"
    assert len(root.findall("person[@id='12']/stop")) == 1
    for original, current in zip(originals, fixture_population):
        assert_frame_equal(original, current)


@pytest.mark.parametrize("mode,tag,selected", [
    ("walk", "walk", None), ("bike", "personTrip", "bicycle"),
    ("pt", "personTrip", "public"), ("car_passenger", "ride", None),
    ("car_loop", "personTrip", "car"),
])
def test_mode_mapping_preserves_original_mode(fixture_population, tmp_path, mode, tag, selected):
    source = prepared(fixture_population)
    source["activities"].loc[source["activities"]["activity_index"] == 0, "following_mode"] = mode
    context = Context(plans, tmp_path, stages={"sumo.scenario.population.population": source})
    tree = read_xml(plans.execute(context))
    stage = tree.find(f"person[@id='11']/{tag}")
    assert stage is not None
    assert stage.get("modes") == selected
    assert stage.find("param[@key='mode']").get("value") == mode
    if mode == "car_passenger":
        assert stage.get("lines") == "ANY"


def test_passenger_vehicle_reference_and_explicit_taxi_option(fixture_population, tmp_path):
    source = prepared(fixture_population)
    source["activities"].loc[source["activities"]["activity_index"] == 0, "following_mode"] = "car_passenger"
    source["activities"]["passenger_vehicle_id"] = "assigned_car"
    context = Context(plans, tmp_path, stages={"sumo.scenario.population.population": source})
    assert read_xml(plans.execute(context)).find("person/ride").get("lines") == "assigned_car"
    context.values["sumo_car_passenger_mode"] = "taxi"
    assert read_xml(plans.execute(context)).find("person/personTrip").get("modes") == "taxi"


@pytest.mark.parametrize("chosen", [True, False])
def test_conditional_population_dependencies(chosen):
    context = ConfigurationContext({"enable_mode_choice": chosen, "include_external_population": chosen,
                                    "include_cross_border": chosen})
    population.configure(context)
    stages = [stage["descriptor"] for stage in context.required_stages]
    assert ("mode_choice.run" in stages) == chosen
    assert ("synthesis.population.trips_with_departures" in stages) != chosen
    assert ("data.external_population.read_outputs" in stages) == chosen
    assert ("data.cross_border.generate_cross_border_traffic" in stages) == chosen
    assert not any(stage.startswith("matsim.") for stage in stages)


def test_population_stage_includes_external_and_crossborder(fixture_population, tmp_path):
    persons, activities, locations, final_trips = fixture_population
    external = persons.iloc[:1].copy().assign(person_id=30)
    crossing = persons.iloc[:1].copy().assign(person_id=40)
    nontraveller = activities.iloc[-1:].copy()
    stages = {"synthesis.population.models.subscriptions": persons,
              "synthesis.population.activities": activities,
              "synthesis.population.spatial.locations": locations, "population_trips": final_trips,
              "data.external_population.read_outputs": (external, nontraveller.assign(person_id=30), None),
              "data.cross_border.generate_cross_border_traffic": (crossing, nontraveller.assign(person_id=40), None)}
    context = Context(population, tmp_path, {"include_cross_border": True}, stages)
    result = population.execute(context)
    assert set(result["persons"]["person_id"]) == {"11", "12", "30", "40"}
    assert set(result["persons"]["population_source"]) == {"resident", "external", "crossborder"}


def test_invalid_keys_and_chronology_fail(fixture_population, tmp_path):
    persons, activities, _, _ = fixture_population
    with pytest.raises(ValueError, match="Duplicate person_id"):
        tools.combine_populations([(pd.concat([persons, persons]), activities, "resident")])
    with pytest.raises(ValueError, match="Duplicate person_id/activity_index"):
        tools.combine_populations([(persons, pd.concat([activities, activities.iloc[:1]]), "resident")])
    result = prepared(fixture_population)
    result["activities"].loc[result["activities"]["activity_index"] == 1, "start_time"] = 1
    context = Context(plans, tmp_path, stages={"sumo.scenario.population.population": result})
    with pytest.raises(ValueError, match="Inconsistent activity/trip times"):
        plans.execute(context)


def test_departure_sorting_and_missing_chain():
    persons = pd.DataFrame({"person_id": [1, 2]})
    activities = pd.DataFrame({"person_id": [1, 2], "activity_index": [0, 0], "purpose": ["home", "home"],
                               "start_time": [100, 0]})
    result = tools.combine_populations([(persons, activities, "resident")])
    assert list(result["persons"]["person_id"]) == ["2", "1"]
    assert list(result["activities"]["person_id"]) == ["2", "1"]
    with pytest.raises(ValueError, match="missing their activity chain"):
        tools.combine_populations([(persons, activities.iloc[:1], "resident")])


def test_scalar_ids_and_coordinate_conversion():
    assert tools.identifier(1234567890123456.0) == "1234567890123456"
    assert tools.identifier("0001") == "0001"
    assert tools.scalar_text(np.bool_(False)) == "false"
    assert tools.scalar_text(float("nan")) is None
    transform = tools.lonlat_transformer("EPSG:2056")
    projected = Transformer.from_crs(4326, 2056, always_xy=True).transform(8, 47)
    lonlat = [float(part) for part in tools.to_lonlat(transform, projected).split(",")]
    assert lonlat == pytest.approx([8, 47], abs=1e-7)
    with pytest.raises(ValueError, match="coordinates"):
        tools.finite_coordinates(None, 1)


def test_deterministic_gzip_and_xml_escaping(tmp_path):
    results = []
    for index in range(2):
        path = tmp_path / f"persons_{index}.xml.gz"
        with PopulationWriter(path) as writer:
            with writer.person('person&1', attributes={"a": '<&"'}):
                writer.stop("8,47", until=1, activity_type="home&work")
        assert read_xml(path).find("person/param").get("value") == '<&"'
        results.append(path.read_bytes())
    assert results[0] == results[1]


def test_freight_lcv_sorted_and_separate_from_persons(tmp_path):
    freight = pd.DataFrame({"agent_id": [7, 8], "origin_x": [2600000] * 2, "origin_y": [1200000] * 2,
                            "destination_x": [2600100] * 2, "destination_y": [1200010] * 2,
                            "departure_time": [20, 5], "vehicle_type": ["truck", "truck"]})
    lcv = freight.iloc[:1].rename(columns={"agent_id": "trip_id"}).assign(departure_time=10)
    before = freight.copy()
    context = Context(trips, tmp_path, stages={"synthesis.freight.trips": freight, "synthesis.lcv.trips": lcv})
    tree = read_xml(trips.execute(context))
    assert [trip.get("id") for trip in tree.findall("trip")] == ["freight_8", "lcv_7", "freight_7"]
    assert [trip.get("type") for trip in tree.findall("trip")] == ["sumo_truck", "sumo_delivery", "sumo_truck"]
    assert tree.find("person") is None
    assert_frame_equal(freight, before)
    context = Context(trips, tmp_path / "disabled", {"use_freight": False, "use_lcv": False})
    assert not read_xml(trips.execute(context)).findall("trip")


def test_runtime_bundled_projection_database_and_user_override(tmp_path, monkeypatch):
    projection = tmp_path / "data/proj"
    projection.mkdir(parents=True)
    (projection / "proj.db").touch()
    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.delenv("PROJ_LIB", raising=False)
    environment = sumo._environment({"home": str(tmp_path)})
    assert environment["PROJ_DATA"] == str(projection)
    assert "PROJ_DATA" not in os.environ
    monkeypatch.setenv("PROJ_DATA", "/custom/proj")
    assert sumo._environment({"home": str(tmp_path)})["PROJ_DATA"] == "/custom/proj"


def test_native_schema_routing_and_simulation(fixture_population, tmp_path, runtime):
    source = tmp_path / "input.osm"
    source.write_text(OSM)
    (tmp_path / "osm").mkdir()
    network_context = Context(road_network, tmp_path / "network", {"data_path": str(tmp_path),
        "random_seed": 42, "sumo_osm_file": str(source)}, {"sumo.runtime.sumo": runtime})
    network = road_network.execute(network_context)
    context = Context(plans, tmp_path / "population", stages={"sumo.scenario.population.population": prepared(fixture_population)})
    routes = plans.execute(context)
    schema = etree.XMLSchema(etree.parse(str(Path(runtime["home"]) / "data/xsd/routes_file.xsd")))
    schema.assertValid(read_xml(routes))
    routed = tmp_path / "routed.rou.xml"
    runtime_context = Context(sumo, tmp_path / "routing", stages={"sumo.runtime.sumo": runtime})
    sumo.run(runtime_context, ["-n", network, "-r", routes, "-o", routed, "--xml-validation", "never"], binary="duarouter")
    assert etree.parse(str(routed)).find("person") is not None
    # SUMO 1.28.0 handles repeated private-car personTrips directly. Its offline
    # duarouter currently duplicates the generated private vehicle ID per leg.
    tripinfo = tmp_path / "tripinfo.xml"
    sumo.run(runtime_context, ["-n", network, "-r", routes, "--end", "400", "--no-step-log", "true",
                              "--xml-validation", "never", "--tripinfo-output", tripinfo])
    person = etree.parse(str(tripinfo)).find("personinfo[@id='11']")
    assert person is not None
    assert len(person.findall("ride")) == 2


def test_writer_all_supported_plan_elements_validate(tmp_path, runtime):
    path = tmp_path / "schema.rou.xml.gz"
    with PopulationWriter(path) as writer:
        writer.vehicle_type("test_type", "passenger", maxSpeed=20)
        writer.trip("v1", 0, "8,47", "8.002,47", vehicle_type="test_type")
        with writer.person("p1"):
            writer.stop("8,47", until=1)
            writer.walk("8,47", "8.001,47")
            writer.ride("8.001,47", "8.002,47", lines="v1")
            writer.person_trip("8.002,47", "8,47", "public")
    schema = etree.XMLSchema(etree.parse(str(Path(runtime["home"]) / "data/xsd/routes_file.xsd")))
    schema.assertValid(read_xml(path))
