"""Calibrate foreign transit after routing, with whole journeys and gateway controls."""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from lxml import etree as ET
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

CHIASSO = "Chiasso Autostrada (421)"
# Stable OSM nodes on one interior A2 segment, shared by both carriageway directions.
NODES = ("5066257662", "6406319663")


def screenline(network):
    links = {}
    with gzip.open(network, "rb") as stream:
        for _, link in ET.iterparse(stream, tag="link", events=("end",)):
            pair = (link.get("from"), link.get("to"))
            if set(pair) == set(NODES) and "car" in link.get("modes", "").split(","):
                if pair in links:
                    raise ValueError(f"Ambiguous Gotthard car screenline at {pair}")
                links[pair] = link.get("id")
            clear(link)
    if len(links) != 2:
        raise ValueError(f"Expected two Gotthard car links at {NODES}, found {links}")
    return set(links.values())


def clear(element):
    parent = element.getparent()
    element.clear()
    while element.getprevious() is not None:
        del parent[0]


def records(population):
    # RunPopulationRouting writes explicit person elements. Keep unaffected plans
    # byte-for-byte, and parse only the cross-border plans needed for calibration.
    with gzip.open(population, "rb") as stream:
        tail = b""
        while chunk := stream.read(2 << 20):
            parts = (tail + chunk).split(b"</person>")
            tail = parts.pop()
            for part in parts:
                start = part.index(b"<person ")
                yield part[:start], part[start:] + b"</person>"
        if b"<person " in tail or b"</population>" not in tail:
            raise ValueError("Incomplete prepared population XML")
        yield tail, None


def inventory(population, metadata, links, chiasso_links):
    # Group the two directional links of a physical visitor gateway together.
    # Matched and network-inferred exits use the actual directional road link.
    from networkx.utils import UnionFind

    sites = UnionFind()
    for row in metadata.values():
        if (row["label"] == "From-To" and row["status"] == "patched"
                and row["entry_interview_point_id"].removesuffix("_entry")
                == row["exit_interview_point_id"].removesuffix("_exit")):
            entry, exit_ = terminals(row)
            sites.union(entry, exit_)
    if chiasso_links:
        sites.union(*sorted(chiasso_links))
    site = {link: min(group) for group in sites.to_sets() for link in group}
    chiasso = site[next(iter(chiasso_links))] if chiasso_links else None
    gateways = Counter()
    journeys = defaultdict(list)
    total = 0
    largest_id = 0
    seen = set()
    for _, raw in records(population):
        if raw is None:
            break
        pid = re.match(rb'<person\s+id="([^"]+)"', raw)[1].decode()
        if pid.isdecimal():
            largest_id = max(largest_id, int(pid))
        row = metadata.get(pid)
        if row is None and b'type="border"' not in raw:
            continue
        person = ET.fromstring(raw)
        plan = person.find("plan[@selected='yes']")
        if plan is None:
            raise ValueError(f"Missing selected prepared plan for {pid}")
        activities = [a for a in plan.findall("activity") if not a.get("type").endswith("interaction")]
        car_trips = []
        for origin in activities[:-1]:
            modes = []
            for element in origin.itersiblings():
                if element.tag == "activity" and not element.get("type").endswith("interaction"):
                    break
                if element.tag == "leg":
                    modes.append(element.get("mode"))
            car_trips.append("car" in modes)
        if row:
            seen.add(pid)
        if not any(car_trips):
            continue
        count = 0
        if row and row["label"] == "Through":
            for leg in plan.findall("leg[@mode='car']"):
                route = leg.find("route")
                if route is None or route.get("type") != "links":
                    raise ValueError(f"Unrouted foreign transit driver {pid}")
                sequence = (route.text or "").split()
                # MATSim normally includes the two end links in the route text.
                if not sequence or sequence[0] != route.get("start_link"):
                    sequence.insert(0, route.get("start_link"))
                if sequence[-1] != route.get("end_link"):
                    sequence.append(route.get("end_link"))
                count += sum(link in links for link in sequence)
            total += count
        if row and row["status"] == "patched":
            ends = terminals(row)
            if (activities[0].get("link"), activities[-1].get("link")) != (
                    row["entry_terminal"], row["exit_terminal"]) or not all(car_trips):
                raise ValueError(f"Assigned terminals disagree with routed driver {pid}")
        else:
            ends = [a.get("link") for i, a in enumerate(activities)
                    if a.get("type") == "border" and any(car_trips[max(0, i - 1):i + 1])]
            if row and row["label"] == "Through" and len(ends) == 1 and row["inferred_exit_link"]:
                ends.append(row["inferred_exit_link"])
        ends = tuple(sorted(site.get(link, link) for link in ends))
        gateways.update(ends)
        if row and row["status"] == "patched":
            if row["label"] == "Through" and count:
                journeys[("transit", count, ends)].append(pid)
            elif row["label"] == "From-To" and len(set(ends)) == 1:
                journeys[("visitor", 0, ends)].append(pid)
    missing = {pid for pid, row in metadata.items() if row["status"] == "patched"} - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} patched border agents")
    return gateways, journeys, total, largest_id, chiasso


def terminals(row):
    return tuple(row[field].split(":", 2)[2] for field in ("entry_terminal", "exit_terminal"))


def balance(before, journeys, total, goal, chiasso, chiasso_goal):
    # Change complete tours. Minimize gateway rounding errors first, then the
    # deviation from a common sampling rate for eligible transit OD pairs.
    visitors = {ends[0] for kind, _, ends in journeys if kind == "visitor"}
    keys = sorted(k for k in journeys if k[0] == "visitor" or set(k[2]) <= visitors)
    changes = {s: 0 for s in before}
    if chiasso_goal is not None:
        if chiasso not in visitors:
            raise ValueError("No local Chiasso visitor tours to balance the crossing target")
        changes[chiasso] = chiasso_goal - before[chiasso]
    delta = goal - total
    capacity = sum(len(journeys[k]) * k[1] for k in keys)
    if delta < -capacity or (delta and not capacity):
        raise ValueError(f"Gotthard adjustment {delta} exceeds eligible transit support {capacity}")
    affected = {s for k in keys if k[0] == "transit" for s in k[2]}
    if chiasso_goal is not None:
        affected.add(chiasso)
    keys = [k for k in keys if k[0] == "transit" or k[2][0] in affected]
    if not keys:
        return {}
    sites = sorted({s for k in keys for s in k[2]})
    n, g = len(keys), len(sites)
    # x = signed integer tour changes, d = absolute proportional deviations,
    # e = absolute gateway residuals (at most one sampled passage per site).
    matrix = lil_matrix((1 + 2 * g + 2 * n, 2 * n + g))
    low = np.full(matrix.shape[0], -np.inf)
    high = np.full(matrix.shape[0], np.inf)
    lo = np.zeros(matrix.shape[1])
    hi = np.full(matrix.shape[1], np.inf)
    hi[2 * n:] = 1
    low[0] = high[0] = delta
    for i, s in enumerate(sites):
        high[1 + 2 * i] = changes[s]
        high[2 + 2 * i] = -changes[s]
        matrix[1 + 2 * i, 2 * n + i] = matrix[2 + 2 * i, 2 * n + i] = -1
    for j, k in enumerate(keys):
        kind, count, ends = k
        size = len(journeys[k])
        lo[j] = -size
        if kind == "transit":
            if delta <= 0:
                hi[j] = 0
            else:
                lo[j] = 0
        matrix[0, j] = count
        for s in ends:
            r = 1 + 2 * sites.index(s)
            matrix[r, j] += 1
            matrix[r + 1, j] -= 1
        desired = size * delta / capacity if count else 0
        r = 1 + 2 * g + 2 * j
        matrix[r, j], matrix[r + 1, j] = 1, -1
        matrix[r, n + j] = matrix[r + 1, n + j] = -1
        high[r], high[r + 1] = desired, -desired
    integrality = np.r_[np.ones(n), np.zeros(n+g)]
    bounds = Bounds(lo, hi)
    constraints = LinearConstraint(matrix.tocsr(), low, high)
    # First establish the minimum physically attainable number of odd passages.
    cost = np.r_[np.zeros(2 * n), np.ones(g)]
    fit = milp(cost, integrality=integrality, bounds=bounds, constraints=constraints)
    if not fit.success:
        raise ValueError(f"Infeasible Gotthard/gateway targets: {fit.message}")
    # Give Chiasso priority among solutions with the same minimum residual.
    cost2 = np.r_[np.zeros(n), [1 if k[0] == "transit" else 0.001 for k in keys], np.zeros(g)]
    fixes = [constraints, LinearConstraint(cost, 0, round(fit.fun))]
    if chiasso_goal is not None:
        j = 2 * n + sites.index(chiasso)
        weight = np.zeros(len(cost))
        weight[j] = 1
        nearest = milp(weight, integrality=integrality, bounds=bounds, constraints=fixes)
        if not nearest.success:
            raise ValueError(nearest.message)
        fixes.append(LinearConstraint(weight, 0, round(nearest.fun)))
    fit = milp(cost2, integrality=integrality, bounds=bounds, constraints=fixes)
    if not fit.success:
        raise ValueError(f"Gotthard journey fit failed: {fit.message}")
    return {k: int(round(x)) for k, x in zip(keys, fit.x[:n]) if abs(x) > 0.5}


def person_changes(journeys, deltas, largest_id, seed):
    rng = np.random.default_rng(seed)
    copies = {}
    for key, delta in sorted(deltas.items()):
        ids = rng.permutation(sorted(journeys[key])).tolist()
        if delta < 0:
            copies.update({pid: [] for pid in ids[:-delta]})
        else:
            for i in range(delta):
                donor = ids[i % len(ids)]
                largest_id += 1
                copies.setdefault(donor, [donor]).append(str(largest_id))
    return copies


def rewrite_population(path, output, copies):
    seen = set()
    vehicles = {}
    with gzip.open(output, "wb", compresslevel=2) as target:
        for prefix, raw in records(path):
            target.write(prefix)
            if raw is None:
                continue
            pid = re.match(rb'<person\s+id="([^"]+)"', raw)[1].decode()
            if pid not in copies:
                target.write(raw)
                continue
            seen.add(pid)
            person = ET.fromstring(raw)
            attribute = person.find("attributes/attribute[@name='vehicles']")
            refs = json.loads(attribute.text) if attribute is not None else {}
            if not refs.get("car") or any(not v.startswith(pid + ":") for v in refs.values()):
                raise ValueError(f"Unexpected vehicle ownership for cross-border driver {pid}")
            for vehicle in refs.values():
                vehicles[vehicle] = [new + vehicle[len(pid):] for new in copies[pid]]
            for new in copies[pid]:
                if new == pid:
                    target.write(raw)
                    continue
                clone = deepcopy(person)
                clone.set("id", new)
                clone.find("attributes/attribute[@name='vehicles']").text = json.dumps(
                    {mode: new + vehicle[len(pid):] for mode, vehicle in refs.items()})
                for route in clone.iter("route"):
                    if route.get("vehicleRefId") in refs.values():
                        route.set("vehicleRefId", new + route.get("vehicleRefId")[len(pid):])
                for name in ("householdId", "household_id"):
                    hh = clone.find(f"attributes/attribute[@name='{name}']")
                    if hh is not None and hh.text == pid:
                        hh.text = new
                target.write(ET.tostring(clone))
    if seen != copies.keys():
        raise ValueError("Missing people while applying Gotthard weights")
    return vehicles


def rewrite_auxiliary(path, output, tag, copies):
    seen = set()
    new_ids = {v for key, values in copies.items() for v in values if v != key}
    with gzip.open(path, "rb") as source, gzip.open(output, "wb", compresslevel=2) as target:
        parser = ET.iterparse(source, events=("start", "end"), resolve_entities=False, no_network=True)
        _, root = next(parser)
        with ET.xmlfile(target, encoding="utf-8") as writer:
            writer.write_declaration()
            if doctype := root.getroottree().docinfo.doctype:
                writer.write_doctype(doctype)
            with writer.element(root.tag, dict(root.attrib), nsmap=root.nsmap):
                for event, element in parser:
                    if event != "end" or element.getparent() is not root:
                        continue
                    key = element.get("id")
                    if ET.QName(element).localname != tag:
                        writer.write(element)
                    elif key in new_ids:
                        raise ValueError(f"Duplicate generated {tag} ID {key}")
                    elif key not in copies:
                        writer.write(element)
                    else:
                        seen.add(key)
                        if tag == "household":
                            members = element.findall(".//{*}personId")
                            if [m.get("refId") for m in members] != [key]:
                                raise ValueError(f"Expected single-person foreign household {key}")
                            household_vehicles = [(v, v.get("refId")) for v in
                                                  element.findall(".//{*}vehicleDefinitionId")]
                            if any(not ref.startswith(key + ":") for _, ref in household_vehicles):
                                raise ValueError(f"Unexpected shared household vehicle for {key}")
                        for new in copies[key]:
                            element.set("id", new)
                            if tag == "household":
                                members[0].set("refId", new)
                                for vehicle, ref in household_vehicles:
                                    vehicle.set("refId", new + ref[len(key):])
                            writer.write(element)
                    clear(element)
    if seen != copies.keys():
        raise ValueError(f"Missing {tag} records for {len(copies.keys() - seen)} changed IDs")


def calibrate(directory, population, households, vehicles, network, chiasso_links,
              daily_target, chiasso_target, sample, seed):
    directory = Path(directory)
    population, households, vehicles = map(Path, (population, households, vehicles))
    if not np.isfinite(daily_target) or daily_target < 0 or not 0 < sample <= 1:
        raise ValueError("Expected a nonnegative Gotthard target and a sample fraction in (0, 1]")
    diagnostics = directory / "border_endpoints_persons.csv"
    with diagnostics.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        metadata = {row["person_id"]: row for row in reader}
    links = screenline(network)
    before, journeys, total, largest, chiasso = inventory(population, metadata, links, chiasso_links)
    goal = round(sample * daily_target)
    chi_goal = round(sample * chiasso_target) if chiasso_target is not None else None
    deltas = balance(before, journeys, total, goal, chiasso, chi_goal)
    copies = person_changes(journeys, deltas, largest, seed)
    after = before.copy()
    tunnel_after = total
    for (_, passes, ends), delta in deltas.items():
        tunnel_after += passes * delta
        for site in ends:
            after[site] += delta
    if tunnel_after != goal or any(abs(after[s] - (chi_goal if s == chiasso and chi_goal is not None
                                                    else before[s])) > 1 for s in after):
        raise ValueError("Integer Gotthard solution failed its passage controls")
    with TemporaryDirectory(dir=directory) as tmp:
        tmp = Path(tmp)
        outputs = []
        if copies:
            vehicle_copies = rewrite_population(population, tmp / population.name, copies)
            rewrite_auxiliary(households, tmp / households.name, "household", copies)
            rewrite_auxiliary(vehicles, tmp / vehicles.name, "vehicle", vehicle_copies)
            with (tmp / diagnostics.name).open("w") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                for pid, row in metadata.items():
                    for new in copies.get(pid, [pid]):
                        writer.writerow(dict(row, person_id=new))
            outputs.extend([population, households, vehicles, diagnostics])
        with (tmp / "gotthard_transit_gateways.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gateway_link", "is_chiasso", "sample_before", "sample_target", "sample_after"])
            for site in sorted(after):
                writer.writerow([site, site == chiasso, before[site],
                                 chi_goal if site == chiasso and chi_goal is not None else before[site], after[site]])
        with (tmp / "gotthard_transit_people.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["action", "person_id", "source_person_id", "label"])
            for pid, ids in sorted(copies.items()):
                if not ids:
                    writer.writerow(["removed", pid, pid, metadata[pid]["label"]])
                for new in ids[1:]:
                    writer.writerow(["added", new, pid, metadata[pid]["label"]])
        summary = dict(screenline_links=sorted(links), sample_fraction=sample, daily_target=daily_target,
                       sample_before=total, sample_target=goal, sample_after=tunnel_after,
                       chiasso_sample_before=before[chiasso] if chiasso else None, chiasso_sample_after=after[chiasso] if chiasso else None,
                       chiasso_sample_target=chi_goal)
        (tmp / "gotthard_transit.json").write_text(json.dumps(summary, indent=2) + "\n")
        outputs.extend(directory / name for name in ("gotthard_transit_gateways.csv", "gotthard_transit_people.csv",
                                                   "gotthard_transit.json"))
        for output in outputs:
            (tmp / output.name).replace(output)
    return summary
