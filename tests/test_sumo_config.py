"""Configuration schema, urban presets and flow-sensitive assignment checks."""

import gzip
from pathlib import Path
import pickle

from lxml import etree
import pandas as pd
import pytest
from synpp.pipeline import ConfigurationContext
from shapely.geometry import Point

from sumo.config import loaded, macroscopic, mesoscopic, microscopic
from sumo.router import assignment, mapped
from sumo.scenario import demand, unrouted_demand
from sumo.scenario.network.tools import detect_network_format, resolve_input_path
from sumo.runtime import sumo
from sumo.simulation import run

ROOT = Path(__file__).resolve().parents[1]


class Context:
    def __init__(self, module, path, config=None, stages=None):
        self.base = {"sumo_assignment_all_as_car": False, "sumo_population_crs": "EPSG:2056", "enable_mode_choice": True, "sumo_native_xml": False, "clip_trips_to_border_in_sumo": False, **(config or {})}
        context = ConfigurationContext(self.base)
        module.configure(context)
        self.config_values = {**self.base, **context.required_config}
        self.cache = Path(path)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.stages = stages or {}
        network = (config or {}).get('sumo_network_file')
        if network and Path(network).is_file() and detect_network_format(network) == 'sumo':
            self.stages.setdefault('sumo.scenario.network.road_network', str(Path(network).resolve()))

    def config(self, option):
        return self.config_values[option]

    def path(self, stage=None):
        if stage is not None:
            result = self.stages.get(stage, {})
            if isinstance(result, dict) and "vehicle_trips" in result:
                return str(Path(result["vehicle_trips"]).parent)
        return str(self.cache)

    def stage(self, name):
        if name == "sumo.scenario.unrouted_demand" and name not in self.stages:
            self.stages[name] = unrouted_demand.execute(Context(unrouted_demand, self.cache / "raw", self.base, self.stages))
        if name == "sumo.scenario.network.routing" and name not in self.stages:
            from sumo.scenario.network import routing
            self.stages[name] = routing.execute(Context(routing, self.cache / "routing", self.base, self.stages))
        return self.stages[name]


@pytest.fixture(scope="module")
def runtime():
    paths = list((ROOT / "cache_sumo").glob("sumo.runtime.sumo__*.p"))
    if not paths:
        pytest.skip("Install the isolated SUMO runtime first")
    return pickle.loads(paths[0].read_bytes())


@pytest.fixture
def network(tmp_path, runtime):
    nodes = tmp_path / "nodes.xml"
    nodes.write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="100" y="0"/>'
                     '<node id="c" x="200" y="0"/><node id="d" x="300" y="0"/></nodes>')
    edges = tmp_path / "edges.xml"
    edges.write_text('<edges><edge id="in" from="a" to="b" length="100" speed="10"/>'
                     '<edge id="bottleneck" from="b" to="c" length="100" speed="10"/>'
                     '<edge id="out" from="c" to="d" length="100" speed="10"/></edges>')
    path = tmp_path / "network.net.xml"
    context = Context(sumo, tmp_path / "network", stages={"sumo.runtime.sumo": runtime})
    sumo.run(context, ["--node-files", nodes, "--edge-files", edges, "--output-file", path,
                       "--no-internal-links", "true"], binary="netconvert")
    return path


def trip_file(tmp_path, count=20):
    path = tmp_path / "source.trips.xml"
    path.write_text('<routes>' + ''.join(
        f'<trip id="t{i}" depart="{i / 4}" from="in" to="out">'
        '<param key="source_travel_time_s" value="32"/></trip>' for i in range(count)) + '</routes>')
    return path


def validate_schema(path, runtime, name):
    schema = etree.XMLSchema(etree.parse(str(Path(runtime["home"]) / "data/xsd" / (name + 'Configuration.xsd'))))
    schema.assertValid(etree.parse(str(path)))


@pytest.mark.parametrize("model,module", [("microscopic", microscopic), ("mesoscopic", mesoscopic)])
def test_simulation_configs_validate_run_and_keep_producers_readonly(tmp_path, runtime, network, model, module):
    routes = trip_file(tmp_path)
    context = Context(module, tmp_path / "config", {"sumo_network_file": str(network),
        "sumo_route_files": [str(routes)], "random_seed": 42, "sumo_end": 100})
    config = Path(module.execute(context))
    validate_schema(config, runtime, "sumo")
    before = {path.name: path.read_bytes() for path in config.parent.iterdir()}
    execution = Context(run, tmp_path / "simulation", {"sumo_simulation_model": model},
                         {"sumo.config.config": {model: str(config)}, "sumo.runtime.sumo": runtime})
    result = run.execute(execution)
    assert Path(result["tripinfo"]).is_file()
    assert Path(result["edge_data"]).is_file()
    assert before == {path.name: path.read_bytes() for path in config.parent.iterdir()}


def test_external_files_avoid_synthesis_dependencies(tmp_path):
    context = ConfigurationContext({"sumo_network_file": "/network.xml", "sumo_route_files": ["/trips.xml"], "random_seed": 42})
    microscopic.configure(context)
    assert [stage['descriptor'] for stage in context.required_stages] == ['sumo.scenario.network.road_network']
    context = ConfigurationContext({"enable_mode_choice": True, "sumo_assignment_trip_files": ["/trips.xml"], "sumo_native_xml": False, "clip_trips_to_border_in_sumo": False})
    demand.configure(context)
    assert [s["descriptor"] for s in context.required_stages] == ["sumo.scenario.unrouted_demand"]


def prepare_workflow(tmp_path, runtime, network, scale=1):
    routes = trip_file(tmp_path, count=200)
    original = routes.read_bytes()
    inputs = demand.execute(Context(demand, tmp_path / "demand", {"sumo_assignment_trip_files": [str(routes)]}))
    mapping = mapped.execute(Context(mapped, tmp_path / "mapping", {"sumo_network_file": str(network), "random_seed": 42},
        {"sumo.runtime.sumo": runtime, "sumo.router.demand": inputs}))
    macro = macroscopic.execute(Context(macroscopic, tmp_path / "macro", {
        "random_seed": 42, "sumo_end": 3600, "sumo_assignment_iterations": 10, "sumo_assignment_scale": scale},
        {"sumo.router.mapped": mapping}))
    validate_schema(macro, runtime, "marouter")
    workflow = loaded.execute(Context(loaded, tmp_path / "loaded", {"random_seed": 42},
        {"sumo.router.mapped": mapping, "sumo.config.macroscopic": macro}))
    validate_schema(workflow["routing_config"], runtime, "duarouter")
    assert routes.read_bytes() == original
    return workflow


def test_loaded_assignment_reports_every_trip_and_responds_to_flow(tmp_path, runtime, network):
    results = []
    for scale in (1, 2):
        root = tmp_path / str(scale)
        root.mkdir()
        config = prepare_workflow(root, runtime, network, scale)
        before = {path: Path(path).read_bytes() for path in (config["assignment_config"], config["routing_config"])}
        result = assignment.execute(Context(assignment, root / "execution", stages={
            "sumo.runtime.sumo": runtime, "sumo.config.loaded": config}))
        from sumo.tables import read_table
        metrics = read_table(result["metrics"]).to_pandas()
        assert len(metrics) == 200
        assert metrics["trip_id"].is_unique
        assert set(metrics["distance_m"]) == {300}
        assert (metrics["loaded_travel_time_s"] > 30).all()
        assert (metrics["reference_travel_time_s"] == 32).all()
        assert metrics["loads_network"].all()
        with gzip.open(result["weights"], "rb") as stream:
            weights = etree.parse(stream)
        assert float(weights.find("interval/edge[@id='bottleneck']").get("entered")) == pytest.approx(200 * scale)
        assert all(Path(path).read_bytes() == content for path, content in before.items())
        assert not Path(config["weights"]).exists()
        results.append(metrics["loaded_travel_time_s"].mean())
    assert results[1] > results[0]


def test_flow_changes_assignment_paths(tmp_path, runtime):
    nodes = tmp_path / "nodes.xml"
    nodes.write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="100" y="0"/>'
                     '<node id="c" x="200" y="0"/><node id="d" x="300" y="0"/>'
                     '<node id="e" x="150" y="50"/></nodes>')
    edges = tmp_path / "edges.xml"
    edges.write_text('<edges><edge id="in" from="a" to="b" length="100" speed="10"/>'
                     '<edge id="fast" from="b" to="c" length="100" speed="10"/>'
                     '<edge id="slow1" from="b" to="e" length="70" speed="8"/>'
                     '<edge id="slow2" from="e" to="c" length="70" speed="8"/>'
                     '<edge id="out" from="c" to="d" length="100" speed="10"/></edges>')
    network = tmp_path / "network.net.xml"
    context = Context(sumo, tmp_path / "net", stages={"sumo.runtime.sumo": runtime})
    sumo.run(context, ["--node-files", nodes, "--edge-files", edges, "--output-file", network,
                       "--no-internal-links", "true"], binary="netconvert")
    workflow = prepare_workflow(tmp_path, runtime, network, scale=10)
    result = assignment.execute(Context(assignment, tmp_path / "run", stages={
        "sumo.runtime.sumo": runtime, "sumo.config.loaded": workflow}))
    with gzip.open(result["weights"], "rb") as stream:
        weights = etree.parse(stream)
    flow = {edge.get("id"): float(edge.get("entered")) for edge in weights.findall("interval/edge")}
    assert flow["fast"] > 0
    assert flow["slow1"] > 0
    assert flow["slow1"] == pytest.approx(flow["slow2"])
    assert flow["fast"] + flow["slow1"] == pytest.approx(2000)


def test_native_ue_and_invalid_steps_rejected(tmp_path, network):
    with pytest.raises(ValueError, match="native UE"):
        macroscopic.execute(Context(macroscopic, tmp_path / "macro", {"random_seed": 42, "sumo_assignment_method": "UE"},
                                    {"sumo.router.mapped": {}}))
    source = trip_file(tmp_path)
    context = Context(microscopic, tmp_path / "micro", {"sumo_network_file": str(network),
        "sumo_route_files": [str(source)], "random_seed": 42, "sumo_micro_step_length": .3})
    with pytest.raises(ValueError, match="multiple"):
        microscopic.execute(context)


def test_assignment_rejects_non_trip_input(tmp_path):
    path = tmp_path / "persons.xml"
    path.write_text('<routes><person id="p" depart="0"><stop edge="in" until="10"/></person></routes>')
    context = Context(demand, tmp_path / "cache", {"sumo_assignment_trip_files": [str(path)]})
    with pytest.raises(ValueError, match="individual <trip>"):
        demand.execute(context)


def test_multiple_trip_files_share_equivalent_vehicle_types(tmp_path):
    first, second = tmp_path / "first.xml", tmp_path / "second.xml"
    first.write_text('<routes><vType id="car" vClass="passenger"><param key="x" value="1"/></vType>'
                     '<trip id="a" type="car" depart="0" from="in" to="out"/></routes>')
    second.write_text('<routes>\n<vType vClass="passenger" id="car">\n  <param value="1" key="x"/>\n</vType>\n'
                      '<trip id="b" type="car" depart="1" from="in" to="out"/></routes>')
    context = Context(demand, tmp_path / "cache", {"sumo_assignment_trip_files": [str(first), str(second)]})
    assert demand.execute(context)["count"] == 2


def test_demand_reads_coordinates_only_for_selected_trip_endpoints(tmp_path):
    persons = pd.DataFrame({'person_id': ['p', 'inactive'], 'home_x': [float('nan')] * 2,
                            'home_y': [float('nan')] * 2})
    activities = pd.DataFrame([
        {'person_id': 'p', 'activity_index': 0, 'purpose': 'work', 'end_time': 10,
         'geometry': Point(2600000, 1200000), 'following_mode': 'car'},
        {'person_id': 'p', 'activity_index': 1, 'purpose': 'work', 'start_time': 30, 'end_time': 50,
         'geometry': Point(2600100, 1200000), 'following_mode': 'pt'},
        {'person_id': 'p', 'activity_index': 2, 'purpose': 'other', 'geometry': None},
        {'person_id': 'inactive', 'activity_index': 0, 'purpose': 'international', 'geometry': None},
    ])
    original = activities.copy(deep=True)
    context = Context(demand, tmp_path, {'sumo_modes': ['car']},
        {'sumo.scenario.population.population': {'persons': persons, 'activities': activities}})
    trips = list(unrouted_demand._population_trips(context))
    assert len(trips) == 1
    assert trips[0]['attributes']['id'] == 'person_p_0'
    assert trips[0]['parameters']['source_travel_time_s'] == 20
    pd.testing.assert_frame_equal(activities, original)
    activities.loc[1, 'geometry'] = None
    with pytest.raises(ValueError, match='person p, activity 1, selected trip mode car'):
        list(unrouted_demand._population_trips(context))


@pytest.mark.parametrize('arrival', [-1, None, float('nan'), float('inf'), 5, 30])
def test_invalid_reference_arrival_does_not_remove_or_retime_trip(tmp_path, arrival):
    persons = pd.DataFrame({'person_id': ['p']})
    activities = pd.DataFrame([
        {'person_id': 'p', 'activity_index': 0, 'purpose': 'other', 'end_time': 10,
         'geometry': Point(2600000, 1200000), 'following_mode': 'car'},
        {'person_id': 'p', 'activity_index': 1, 'purpose': 'other', 'start_time': arrival,
         'geometry': Point(2600100, 1200000)},
    ])
    context = Context(demand, tmp_path, {'sumo_modes': ['car']},
        {'sumo.scenario.population.population': {'persons': persons, 'activities': activities}})
    trips = list(unrouted_demand._population_trips(context))
    assert len(trips) == 1
    assert trips[0]['attributes']['id'] == 'person_p_0'
    assert float(trips[0]['attributes']['depart']) == 10
    assert trips[0]['parameters']['source_travel_time_s'] == (20 if arrival == 30 else None)
    activities.loc[0, 'end_time'] = -1
    with pytest.raises(ValueError, match='nonnegative seconds'):
        list(unrouted_demand._population_trips(context))


@pytest.mark.parametrize("all_as_car,expected_load", [(False, 1), (True, 3)])
def test_all_person_trips_are_queries_but_passengers_do_not_add_cars(tmp_path, all_as_car, expected_load):
    commercial = tmp_path / "commercial.xml"
    commercial.write_text("<routes/>")
    persons = pd.DataFrame({"person_id": ["p"], "home_x": [2600000], "home_y": [1200000]})
    activities = pd.DataFrame([
        {"person_id": "p", "activity_index": index, "trip_index": index,
         "purpose": "other", "geometry": Point(2600000 + index * 100, 1200000),
         "start_time": index * 100, "end_time": index * 100 + 20,
         "following_mode": mode}
        for index, mode in enumerate(("car", "car_passenger", "pt", None))
    ])
    context = Context(demand, tmp_path / "demand", {"sumo_assignment_all_as_car": all_as_car, "sumo_modes": None}, {
        "sumo.scenario.population.population": {"persons": persons, "activities": activities},
        "sumo.scenario.population.trips": str(commercial)})
    result = demand.execute(context)
    assert result["count"] == 3
    assert result["loading_count"] == expected_load
    with gzip.open(result["queries"], "rb") as stream:
        tree = etree.parse(stream)
    assert [trip.get("id") for trip in tree.findall("trip")] == ["person_p_0", "person_p_1", "person_p_2"]
    assert tree.find("trip[@id='person_p_1']/param[@key='mode']").get("value") == "car_passenger"
