"""Focused SUMO checks; integration cases use the separately installed runtime."""

import gzip
import json
import os
from pathlib import Path
import pickle
import subprocess
import xml.etree.ElementTree as ET

import osmium
import pytest
import synpp
from synpp.pipeline import ConfigurationContext

from sumo.runtime import sumo
from sumo.scenario.network import road_network

ROOT = Path(__file__).resolve().parents[1]
OSM = '''<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="SUMO regression fixture">
  <node id="1" lat="47.0" lon="8.0"/>
  <node id="2" lat="47.0" lon="8.001"/>
  <node id="3" lat="47.0" lon="8.002"/>
  <node id="4" lat="47.001" lon="8.001"/>
  <node id="5" lat="47.01" lon="8.0"/>
  <node id="6" lat="47.01" lon="8.002"/>
  <way id="10"><nd ref="1"/><nd ref="2"/>
    <tag k="highway" v="residential"/><tag k="oneway" v="yes"/>
    <tag k="maxspeed" v="30"/><tag k="hgv" v="no"/>
  </way>
  <way id="11"><nd ref="2"/><nd ref="3"/>
    <tag k="highway" v="residential"/><tag k="oneway" v="yes"/>
  </way>
  <way id="12"><nd ref="2"/><nd ref="4"/>
    <tag k="highway" v="residential"/><tag k="oneway" v="yes"/>
  </way>
  <way id="13"><nd ref="5"/><nd ref="6"/><tag k="railway" v="rail"/></way>
  <relation id="100">
    <member type="way" ref="10" role="from"/>
    <member type="node" ref="2" role="via"/>
    <member type="way" ref="12" role="to"/>
    <tag k="type" v="restriction"/><tag k="restriction" v="no_left_turn"/>
  </relation>
</osm>
'''


class Context:
    def __init__(self, path, config=None, settings=None):
        self.cache = Path(path)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        configured = ConfigurationContext(config or {})
        if config is not None:
            road_network.configure(configured)
        self.values = configured.required_config

    def config(self, key):
        return self.values[key]

    def path(self):
        return str(self.cache)

    def stage(self, name):
        assert name == "sumo.runtime.sumo"
        return self.settings


@pytest.fixture
def runtime():
    candidates = sorted((ROOT / "cache_sumo").glob("sumo.runtime.sumo__*.p"))
    if not candidates:
        pytest.skip("Install the isolated SUMO runtime in cache_sumo to run integration checks")
    with candidates[-1].open("rb") as stream:
        return pickle.load(stream)


def test_default_selects_latest_single_extract():
    context = ConfigurationContext({"data_path": "/data", "random_seed": 42,
                                    "osm_file": ["france.osm.pbf", "germany.osm.pbf"]})
    road_network.configure(context)
    assert road_network._osm_path(context) == Path("/data/osm/switzerland-261004.osm.pbf")
    assert [stage["descriptor"] for stage in context.required_stages] == ["sumo.runtime.sumo"]


def test_explicit_extract_does_not_require_latest_setting():
    context = ConfigurationContext({"data_path": "/data", "random_seed": 42,
                                    "sumo_osm_file": "/different/input.osm"})
    road_network.configure(context)
    assert "latest_osm_for_correction" not in context.required_config
    assert road_network._osm_path(context) == Path("/different/input.osm")


def test_source_replacement_invalidates_cache_and_missing_file_is_clear(tmp_path):
    source = tmp_path / "input.osm"
    source.write_text(OSM)
    context = Context(tmp_path / "cache", {"data_path": str(tmp_path), "random_seed": 42,
                                          "sumo_osm_file": str(source)})
    before = road_network.validate(context)
    source.write_text(OSM + "\n")
    assert road_network.validate(context) != before
    source.unlink()
    with pytest.raises(FileNotFoundError, match="SUMO OSM input not found"):
        road_network.validate(context)


@pytest.mark.parametrize("filename", [["a.pbf", "b.pbf"], "", "   "])
def test_rejects_non_single_input(tmp_path, filename):
    context = Context(tmp_path, {"data_path": str(tmp_path), "random_seed": 42,
                                 "sumo_osm_file": filename})
    with pytest.raises(ValueError, match="single OSM file"):
        road_network._osm_path(context)


def test_helper_sets_child_environment_and_propagates_failure(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    binary = home / "bin" / "sumo"
    binary.write_text('#!/bin/sh\nprintf "%s" "$SUMO_HOME"\nexit "${1:-0}"\n')
    binary.chmod(0o755)
    monkeypatch.setenv("SUMO_HOME", "parent-value")
    context = Context(tmp_path / "cache", settings={"home": str(home)})
    assert sumo.run(context, catch_output=True) == str(home)
    assert os.environ["SUMO_HOME"] == "parent-value"
    with pytest.raises(subprocess.CalledProcessError) as error:
        sumo.run(context, ["9"], catch_output=True)
    assert error.value.returncode == 9
    with pytest.raises(ValueError, match="name of a SUMO tool"):
        sumo.run(context, binary="../sumo")


def test_pbf_conversion_preserves_tags_and_restrictions(tmp_path):
    source = tmp_path / "input.osm"
    source.write_text(OSM)
    pbf = tmp_path / "input.osm.pbf"
    with osmium.io.Reader(str(source)) as reader, osmium.SimpleWriter(str(pbf)) as writer:
        osmium.apply(reader, writer)
    before = pbf.read_bytes()
    converted = tmp_path / "converted.osm.xml.gz"
    road_network._convert_pbf(pbf, converted)
    with gzip.open(converted, "rb") as stream:
        tree = ET.parse(stream)
    assert tree.find("way[@id='10']/tag[@k='hgv']").get("v") == "no"
    assert tree.find("relation[@id='100']/tag[@k='restriction']").get("v") == "no_left_turn"
    assert len(tree.findall("relation/member")) == 3
    assert pbf.read_bytes() == before


def test_validation_tokens_survive_synpp_json_roundtrip(tmp_path):
    source = tmp_path / "input.osm"
    source.write_text(OSM)
    context = Context(tmp_path / "cache", {"data_path": str(tmp_path), "random_seed": 42,
                                          "sumo_osm_file": str(source)})
    (context.cache / "venv").mkdir()
    (context.cache / "venv" / "pyvenv.cfg").write_text("test fixture")
    for stage in (road_network, sumo):
        token = stage.validate(context)
        assert json.loads(json.dumps(token)) == token


@pytest.mark.parametrize("suffix", [".osm", ".osm.gz", ".osm.pbf"])
def test_real_netconvert_and_sumo(tmp_path, runtime, suffix):
    xml = tmp_path / "source.osm"
    xml.write_text(OSM)
    source = tmp_path / ("input" + suffix)
    if suffix == ".osm.pbf":
        with osmium.io.Reader(str(xml)) as reader, osmium.SimpleWriter(str(source)) as writer:
            osmium.apply(reader, writer)
    elif suffix == ".osm.gz":
        with gzip.open(source, "wt") as stream:
            stream.write(OSM)
    else:
        source.write_text(OSM)
    before = source.read_bytes()
    context = Context(tmp_path / "cache", {"data_path": str(tmp_path), "random_seed": 42,
                                          "sumo_osm_file": str(source)}, runtime)
    network = Path(road_network.execute(context))
    assert network.parent == context.cache
    assert source.read_bytes() == before
    assert not list(context.cache.glob("osm_*"))
    assert (context.cache / "netconvert.log").is_file()
    with gzip.open(network, "rb") as stream:
        root = ET.parse(stream).getroot()
    assert root.tag == "net"
    assert root.find("edge[@id='10']") is not None
    assert root.find("edge[@id='13']") is None
    assert root.find("connection[@from='10'][@to='11']") is not None
    assert root.find("connection[@from='10'][@to='12']") is None
    assert sumo.run(context, ["--net-file", str(network), "--begin", "0", "--end", "1",
                              "--no-step-log", "true", "--xml-validation", "never"]) == 0


def test_invalid_converter_option_fails_and_cleans_temporary_input(tmp_path, runtime):
    source = tmp_path / "input.osm"
    source.write_text(OSM)
    context = Context(tmp_path / "cache", {"data_path": str(tmp_path), "random_seed": 42,
                                          "sumo_osm_file": str(source),
                                          "sumo_netconvert_options": ["--invalid-sumo-option"]}, runtime)
    with pytest.raises(subprocess.CalledProcessError):
        road_network.execute(context)
    assert not list(context.cache.glob("osm_*"))


def test_synpp_reuses_runtime_and_network_and_tracks_source_changes(tmp_path, runtime):
    source = tmp_path / "input.osm"
    source.write_text(OSM)
    config = {"data_path": str(tmp_path), "random_seed": 42,
              "sumo_osm_file": str(source), "sumo_version": runtime["version"]}
    options = {"definitions": [{"descriptor": "sumo.scenario.network.road_network"}],
               "config": config, "working_directory": str(ROOT / "cache_sumo"),
               "verbose": True, "rerun_required": False}
    first = synpp.run(**options)
    assert any(name.startswith("sumo.scenario.network.road_network__") for name in first["stale"])
    assert not any(name.startswith("sumo.runtime.sumo__") for name in first["stale"])
    second = synpp.run(**options)
    assert not second["stale"]
    assert second["results"] == first["results"]
    source.write_text(OSM + "\n")
    third = synpp.run(**options)
    assert len(third["stale"]) == 1
    assert next(iter(third["stale"])).startswith("sumo.scenario.network.road_network__")
