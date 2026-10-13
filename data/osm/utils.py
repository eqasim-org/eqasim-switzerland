"""Shared OSM relation checks and topology-preserving speed corrections."""
import osmium


def is_restriction(relation):
    kind = relation.tags.get("type", "")
    return kind == "restriction" or kind.startswith("restriction:")


def is_transport_relation(relation):
    """Relations carrying routing rules, routes, or transport infrastructure.

    Keep all route modes (including ferry, bicycle and walking), rather than
    maintaining a mode whitelist which would discard useful feeder routes.
    Generic sites/areas qualify only when explicitly tagged for transport.
    """
    tags = relation.tags
    kind = tags.get("type", "")
    return (is_restriction(relation)
            or kind in {"route", "route_master", "public_transport", "connectivity", "destination_sign"}
            or (kind in {"site", "multipolygon"}
                and (any(key in tags for key in ("public_transport", "highway", "railway", "aeroway"))
                     or tags.get("amenity") in {"bus_station", "ferry_terminal", "parking"}
                     or tags.get("site") in {"public_transport", "railway"})))


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
