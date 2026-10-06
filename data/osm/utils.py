"""Shared OSM relation checks and topology-preserving speed corrections."""
import osmium


def is_restriction(relation):
    kind = relation.tags.get("type", "")
    return kind == "restriction" or kind.startswith("restriction:")


def relation_members(relation):
    """Copy member values so they outlive the streaming OSM object."""
    return [(member.type, member.ref, member.role) for member in relation.members]


def members_exist(members, way_ids, node_ids):
    retained = {"w": way_ids, "n": node_ids}
    return bool(members) and all(ref in retained.get(kind, ()) for kind, ref, _ in members)


def apply_speed_correction(tags, way_id, speed_corrections):
    """Apply an explicit local correction to a mutable tag dictionary."""
    speed = None if speed_corrections is None else speed_corrections.get(way_id)
    if speed is not None:
        tags["maxspeed"] = str(speed)
    return tags


def write_with_speed_correction(writer, item, speed_corrections):
    if item.is_way() and speed_corrections is not None and speed_corrections.get(item.id) is not None:
        tags = apply_speed_correction(dict(item.tags), item.id, speed_corrections)
        writer.add_way(osmium.osm.mutable.Way(item, tags=tags))
    else:
        writer.add(item)
