"""Refresh OSM attributes by ID without importing the latest network topology.

Missing ways/tags are not evidence of deletion in a regional extract. Preserve
old data in that case. Existing access, oneway and construction status remain
unchanged. Explicit local speed corrections take precedence over latest OSM.
"""
import logging
from pathlib import Path

import osmium

from data.osm.utils import apply_speed_correction, is_restriction, members_exist, relation_members

logger = logging.getLogger("synpp")
CLOSED_HIGHWAYS = {"construction", "proposed", "abandoned", "disused", "razed"}


def _compatible(members, ways, nodes):
    """Require existing references and a connected turn in the old geometry."""
    if not members_exist(members, ways, nodes):
        return False
    from_ways = [ref for kind, ref, role in members if kind == "w" and role == "from"]
    to_ways = [ref for kind, ref, role in members if kind == "w" and role == "to"]
    via = [(kind, ref) for kind, ref, role in members if role == "via"]
    if not from_ways or not to_ways or not via:
        return False
    if len(via) == 1 and via[0][0] == "n":
        return all(via[0][1] in ways[wid] for wid in from_ways + to_ways)
    if any(kind != "w" for kind, ref in via):
        return False
    # Walk the ordered via ways in either direction using their endpoints.
    for start in from_ways:
        reachable = set(ways[start])
        for _, wid in via:
            refs = ways[wid]
            if not refs:
                return False
            reachable = ({refs[-1]} if refs[0] in reachable else set()) | (
                {refs[0]} if refs[-1] in reachable else set()
            )
        if not all(reachable.intersection(ways[end]) for end in to_ways):
            return False
    return True


def _updated_tags(old, latest, speed_corrections):
    tags = dict(old.tags)
    incoming = dict(latest.tags)
    if (tags.get("highway") not in CLOSED_HIGHWAYS
            and incoming.get("highway") not in CLOSED_HIGHWAYS
            and "highway" in incoming):
        tags["highway"] = incoming["highway"]

    # Directional attributes are relative to node order, which we retain.
    old_refs = {node.ref: i for i, node in enumerate(old.nodes)}
    common = [old_refs[node.ref] for node in latest.nodes if node.ref in old_refs]
    direction = None
    if len(common) >= 2:
        if all(a < b for a, b in zip(common, common[1:])):
            direction = 1
        elif all(a > b for a, b in zip(common, common[1:])):
            direction = -1
    for family in ("maxspeed", "lanes"):
        updates = {}
        for key, value in incoming.items():
            if key != family and not key.startswith(family + ":"):
                continue
            parts = key.split(":")
            if "forward" in parts or "backward" in parts:
                if direction is None:
                    continue
                if direction == -1:
                    parts = [{"forward": "backward", "backward": "forward"}.get(p, p) for p in parts]
            updates[":".join(parts)] = value
        if updates:
            for key in list(tags):
                if key == family or key.startswith(family + ":"):
                    if direction is None and {"forward", "backward"}.intersection(key.split(":")):
                        continue
                    del tags[key]
            tags.update(updates)
    return apply_speed_correction(tags, old.id, speed_corrections)


def update_osm_latest(context, old_file, latest_file, output_file, speed_corrections=None):
    """Update a sorted OSM file from a sorted latest extract (XML or PBF).

    Extract extents and object counts may differ. Matching uses OSM entity type
    and ID (not stream position or geographic proximity). Unmatched old objects
    survive; unmatched latest nodes/ways are ignored. Renumbered or split ways
    are not spatially matched to old ways.

    Keep old restrictions absent from the extract; replace/add latest restrictions
    only when they describe a connected turn in the retained topology. New ways
    and nodes are never imported. Input files are never modified. The latest
    file gets a relation-only pre-scan; the old file is read only once.
    """
    old_file, latest_file, output_file = map(str, (old_file, latest_file, output_file))
    if Path(output_file).resolve() in {Path(old_file).resolve(), Path(latest_file).resolve()}:
        raise ValueError("Updated OSM output must differ from both input files.")

    required_ways, required_nodes = set(), set()
    for relation in osmium.FileProcessor(latest_file, entities=osmium.osm.RELATION):
        if is_restriction(relation):
            members = relation_members(relation)
            required_ways.update(ref for kind, ref, _ in members if kind == "w")
            required_nodes.update(ref for kind, ref, _ in members if kind == "n")
    ways, nodes = {}, set()

    # OSM streams are sorted node/way/relation. Collect only restriction-relevant
    # topology while writing, so the old network needs just one pass.
    processors = (osmium.FileProcessor(old_file), osmium.FileProcessor(
        latest_file, entities=osmium.osm.WAY | osmium.osm.RELATION))
    updated_ways = updated_relations = added_relations = 0
    compatible_count = skipped_count = 0
    with osmium.SimpleWriter(output_file, overwrite=True) as writer:
        for old, latest in context.progress(osmium.zip_processors(*processors),
                                            label="Updating OSM attributes and turn restrictions ..."):
            if old is not None:
                if old.is_node() and old.id in required_nodes:
                    nodes.add(old.id)
                elif old.is_way() and old.id in required_ways:
                    ways[old.id] = tuple(node.ref for node in old.nodes)
            compatible = False
            if latest is not None and latest.is_relation() and is_restriction(latest):
                compatible = _compatible(relation_members(latest), ways, nodes)
                compatible_count += compatible
                skipped_count += not compatible
            if old is None:
                if latest.is_relation() and compatible:
                    writer.add_relation(latest)
                    added_relations += 1
                continue
            if latest is not None and old.is_way():
                tags = _updated_tags(old, latest, speed_corrections)
                writer.add_way(osmium.osm.mutable.Way(old, tags=tags))
                updated_ways += tags != dict(old.tags)
            elif (latest is not None and old.is_relation() and is_restriction(old)
                  and compatible):
                writer.add_relation(latest)
                updated_relations += 1
            else:
                writer.add(old)
    logger.info("Latest OSM: %d compatible restrictions; %d incompatible restrictions skipped.",
                compatible_count, skipped_count)
    logger.info("Latest OSM: updated %d ways and %d restrictions; added %d restrictions.",
                updated_ways, updated_relations, added_relations)
    return output_file
