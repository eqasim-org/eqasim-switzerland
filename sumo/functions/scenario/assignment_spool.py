"""Bounded disk-backed metadata and route-share allocation for national assignments."""

import hashlib
import json
import logging
import math
import random
import sqlite3
import time
from array import array
from collections import OrderedDict
from itertools import groupby

from sumo.functions.scenario.tools import elements

logger = logging.getLogger("synpp")
BATCH = 10000
EXCLUDED = {
    "from",
    "to",
    "via",
    "fromLonLat",
    "toLonLat",
    "viaLonLat",
    "fromXY",
    "toXY",
    "fromJunction",
    "toJunction",
    "fromTaz",
    "toTaz",
}


class AssignmentSpool:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-131072")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
            CREATE TABLE sources(id TEXT PRIMARY KEY, attrs TEXT NOT NULL, source_index INTEGER NOT NULL) WITHOUT ROWID;
            CREATE TABLE groups(gid INTEGER PRIMARY KEY, signature BLOB UNIQUE,
                                alternative_count INTEGER NOT NULL, probabilities TEXT NOT NULL,
                                choices TEXT NOT NULL);
            CREATE TABLE members(seq INTEGER PRIMARY KEY, gid INTEGER NOT NULL, depart REAL NOT NULL,
                                 id TEXT UNIQUE, attrs TEXT NOT NULL, choice INTEGER NOT NULL DEFAULT 0);
        """)
        self.count = 0
        self.phase = "indexing source metadata"
        self.last_heartbeat = time.monotonic()
        self.db.set_progress_handler(self._heartbeat, 1000000)

    def _heartbeat(self):
        now = time.monotonic()
        if now - self.last_heartbeat >= 30:
            logger.info(
                "SUMO realization: %s; %d assigned vehicles indexed",
                self.phase,
                self.count,
            )
            self.last_heartbeat = now
        return 0

    def close(self):
        self.db.close()

    def read_sources(self, path):
        types = {}
        rows = []
        count = 0
        started = time.monotonic()
        from lxml import etree

        for element in elements(path, ("trip", "vType")):
            if element.tag == "vType":
                types[element.get("id")] = etree.fromstring(etree.tostring(element))
                continue
            rows.append(
                (
                    element.get("id"),
                    json.dumps(
                        {
                            "attributes": {
                                k: v
                                for k, v in element.attrib.items()
                                if k not in EXCLUDED
                            },
                            "parameters": {
                                p.get("key"): p.get("value")
                                for p in element.findall("param")
                            },
                        }
                    ),
                    count,
                )
            )
            count += 1
            if len(rows) >= BATCH:
                self.db.executemany("INSERT INTO sources VALUES (?,?,?)", rows)
                rows.clear()
                if count % 100000 == 0:
                    logger.info(
                        "SUMO realization: indexed %d source trips; %.1fs",
                        count,
                        time.monotonic() - started,
                    )
        if rows:
            self.db.executemany("INSERT INTO sources VALUES (?,?,?)", rows)
        self.db.commit()
        self.covered = bytearray(count)
        logger.info(
            "SUMO realization: source indexing complete: %d trips; %.1fs",
            count,
            time.monotonic() - started,
        )
        return types

    def read_assignments(self, path, begin, interval):
        self.phase = "indexing assignments"
        pending = []
        cache = OrderedDict()
        started = time.monotonic()
        for element in elements(path, "vehicle"):
            attrs = dict(element.attrib)
            choices = []
            for route in element.findall("routeDistribution/route") or element.findall(
                "route"
            ):
                probability = float(route.get("probability", "1"))
                edges = tuple(route.get("edges", "").split())
                if not math.isfinite(probability) or probability < 0 or not edges:
                    raise ValueError(
                        f"Invalid assigned route distribution for {attrs['id']}"
                    )
                if probability:
                    choices.append((edges, probability))
            if not choices:
                raise ValueError(
                    f"No positive assigned route probability for {attrs['id']}"
                )
            pending.append((attrs, choices))
            if len(pending) >= BATCH:
                self._insert(pending, cache, begin, interval)
                pending.clear()
                if self.count % 100000 == 0:
                    logger.info(
                        "SUMO realization: indexed %d assigned vehicles; %.1fs",
                        self.count,
                        time.monotonic() - started,
                    )
        if pending:
            self._insert(pending, cache, begin, interval)
        self._finish_assignments(started)

    def read_sources_arrow(self, path, types_file):
        """Ingest native SAX output without constructing per-trip XML objects."""
        from pathlib import Path

        from lxml import etree

        from sumo.functions.scenario.route_io import batches

        started = time.monotonic()
        count = 0
        for batch in batches(path):
            rows = [
                (row["id"], row["payload"], count + index)
                for index, row in enumerate(batch.to_pylist())
            ]
            self.db.executemany("INSERT INTO sources VALUES (?,?,?)", rows)
            count += len(rows)
            if count % 100000 < len(rows):
                logger.info(
                    "SUMO realization: indexed %d native source records; %.1fs",
                    count,
                    time.monotonic() - started,
                )
        self.db.commit()
        self.covered = bytearray(count)
        root = etree.fromstring(
            b"<routes>" + Path(types_file).read_bytes() + b"</routes>"
        )
        logger.info(
            "SUMO realization: source indexing complete: %d trips; %.1fs",
            count,
            time.monotonic() - started,
        )
        return {element.get("id"): element for element in root}

    def read_assignments_arrow(self, path, begin, interval):
        """Index encoded paths in batches, leaving edge lists inside native JSON."""
        from sumo.functions.scenario.route_io import batches

        started = time.monotonic()
        self.phase = "indexing native assignments"
        cache = OrderedDict()
        for batch in batches(path):
            records = [
                (
                    json.loads(row["attributes"]),
                    row["choices"],
                    row["probabilities"],
                    row["alternative_count"],
                )
                for row in batch.to_pylist()
            ]
            self._insert_encoded(records, cache, begin, interval)
            if self.count % 100000 < len(records):
                logger.info(
                    "SUMO realization: indexed %d native assigned records; %.1fs",
                    self.count,
                    time.monotonic() - started,
                )
        self._finish_assignments(started)

    def _finish_assignments(self, started):
        self.db.commit()
        if not self.count:
            raise ValueError("Macroscopic assignment produced no individual vehicles")
        if self.covered.count(0):
            raise ValueError(
                f"Loading did not complete every source trip: {self.covered.count(0)} missing sources"
            )
        logger.info(
            "SUMO realization: indexing group members for %d vehicles", self.count
        )
        self.phase = "building group index on disk"
        self.db.execute("CREATE INDEX member_groups ON members(gid,seq)")
        self.db.commit()
        logger.info(
            "SUMO realization: assignment indexing complete; %.1fs",
            time.monotonic() - started,
        )

    def _insert(self, pending, cache, begin, interval):
        encoded = [
            (
                attrs,
                json.dumps(choices, separators=(",", ":")),
                json.dumps([p for _, p in choices], separators=(",", ":")),
                len(choices),
            )
            for attrs, choices in pending
        ]
        self._insert_encoded(encoded, cache, begin, interval)

    def _insert_encoded(self, pending, cache, begin, interval):
        # Resolve a batch of source IDs, including native scale suffixes, without a national Python dictionary.
        ids = set()
        for attrs, *_ in pending:
            key = attrs["id"]
            ids.add(key)
            parent, separator, clone = key.rpartition(".")
            if separator and clone.isdigit():
                ids.add(parent)
        sources = {}
        keys = list(ids)
        for start in range(0, len(keys), 400):
            part = keys[start : start + 400]
            sources.update(
                (key, (attrs, index))
                for key, attrs, index in self.db.execute(
                    "SELECT id,attrs,source_index FROM sources WHERE id IN ("
                    + ",".join("?" for _ in part)
                    + ")",
                    part,
                )
            )
        members = []
        missing_groups = {}
        for attrs, payload, probabilities, alternative_count in pending:
            key = attrs["id"]
            parent = key
            if key not in sources:
                parent, separator, clone = key.rpartition(".")
                if not separator or not clone.isdigit():
                    parent = key
            if parent not in sources:
                raise ValueError(
                    f"Cannot match loaded vehicle {key} to its source trip"
                )
            encoded, index = sources[parent]
            original = json.loads(encoded)
            self.covered[index] = 1
            attrs = {**original["attributes"], **attrs}
            attrs["_sumo_parameters"] = {
                **original["parameters"],
                "sumo_original_trip_id": parent,
                "sumo_requested_depart": original["attributes"]["depart"],
            }
            depart = float(attrs["depart"])
            if not math.isfinite(depart) or depart < 0:
                raise ValueError(f"Invalid assigned departure for {key}")
            # Keep the identical signature without serializing long edge lists twice.
            prefix = json.dumps(
                [math.floor((depart - begin) / interval), attrs.get("type")],
                separators=(",", ":"),
            )
            signature = hashlib.sha256(
                (prefix[:-1] + "," + payload + "]").encode()
            ).digest()
            if signature not in cache and signature not in missing_groups:
                missing_groups[signature] = (
                    signature,
                    alternative_count,
                    probabilities,
                    payload,
                )
            self.count += 1
            members.append((self.count, signature, depart, key, json.dumps(attrs)))

        # Resolve all new groups in batches, preserving first-seen group order.
        group_ids = dict(cache)
        if missing_groups:
            self.db.executemany(
                "INSERT OR IGNORE INTO groups(signature,alternative_count,probabilities,choices) VALUES (?,?,?,?)",
                missing_groups.values(),
            )
            signatures = list(missing_groups)
            for start in range(0, len(signatures), 400):
                part = signatures[start : start + 400]
                group_ids.update(
                    self.db.execute(
                        "SELECT signature,gid FROM groups WHERE signature IN ("
                        + ",".join("?" for _ in part)
                        + ")",
                        part,
                    )
                )
            for signature in signatures:
                cache[signature] = group_ids[signature]
                if len(cache) > 512:
                    cache.popitem(last=False)
        rows = (
            (seq, group_ids[signature], depart, key, attrs)
            for seq, signature, depart, key, attrs in members
        )
        self.db.executemany(
            "INSERT INTO members(seq,gid,depart,id,attrs) VALUES (?,?,?,?,?)", rows
        )
        self.db.commit()

    def allocate(self, seed):
        self.phase = "allocating route shares"
        rng = random.Random(seed)
        completed = 0
        started = time.monotonic()
        last_log = started
        group_count = self.db.execute("SELECT count(*) FROM groups").fetchone()[0]
        cursor = self.db.execute("""
            SELECT m.gid, g.alternative_count,
                   CASE WHEN g.alternative_count > 1 THEN g.probabilities END, m.seq
            FROM members AS m JOIN groups AS g ON g.gid=m.gid ORDER BY m.gid,m.seq
        """)
        for gid, rows in groupby(cursor, key=lambda row: row[0]):
            first = next(rows)
            members = array("q", [first[3]])
            members.extend(row[3] for row in rows)
            if first[1] == 1:
                # Consume the same random draws as the full allocator, with no path JSON parsing.
                rng.random()
                rng.shuffle(members)
            else:
                probabilities = json.loads(first[2])
                total = math.fsum(probabilities)
                targets = [len(members) * p / total for p in probabilities]
                counts = [math.floor(t) for t in targets]
                ranking = sorted(
                    range(len(probabilities)),
                    key=lambda i: (targets[i] - counts[i], rng.random()),
                    reverse=True,
                )
                for i in ranking[: len(members) - sum(counts)]:
                    counts[i] += 1
                rng.shuffle(members)
                offset = 0
                updates = []
                for choice, count in enumerate(counts):
                    if choice == 0:
                        offset += count
                        continue
                    for index in range(offset, offset + count):
                        updates.append((choice, members[index]))
                        if len(updates) >= BATCH:
                            self.db.executemany(
                                "UPDATE members SET choice=? WHERE seq=?", updates
                            )
                            updates.clear()
                    offset += count
                if updates:
                    self.db.executemany(
                        "UPDATE members SET choice=? WHERE seq=?", updates
                    )
            completed += 1
            if time.monotonic() - last_log >= 30:
                self.db.commit()
                logger.info(
                    "SUMO realization: allocated %d/%d distributions; %.1fs",
                    completed,
                    group_count,
                    time.monotonic() - started,
                )
                last_log = time.monotonic()
        self.db.commit()
        logger.info(
            "SUMO realization: sorting departure index on disk for %d vehicles",
            self.count,
        )
        self.phase = "building departure index on disk"
        self.db.execute("CREATE INDEX departure_order ON members(depart,id)")
        self.db.commit()
        logger.info(
            "SUMO realization: allocated %d distributions; %.1fs",
            group_count,
            time.monotonic() - started,
        )
        return group_count

    def selected(self):
        self.phase = "streaming selected routes"
        cached = OrderedDict()
        cursor = self.db.execute("""
            SELECT m.gid,m.choice,m.attrs,g.choices FROM members AS m
            JOIN groups AS g ON g.gid=m.gid ORDER BY m.depart,m.id
        """)
        for gid, choice, attrs, payload in cursor:
            choices = cached.get(gid)
            if choices is None:
                choices = json.loads(payload)
                cached[gid] = choices
                if len(cached) > 128:
                    cached.popitem(last=False)
            yield json.loads(attrs), choices[choice][0]

    def write_selection(self, selected_path, groups_path, batch_size, batch_bytes):
        """Write compact, ordered IPC batches for the native timing and cutting path."""
        import pyarrow as pa

        from sumo.functions.scenario.route_io import write_batches

        selected_schema = pa.schema(
            [
                ("id", pa.string()),
                ("depart", pa.float64()),
                ("gid", pa.int64()),
                ("choice", pa.int32()),
                ("attributes", pa.string()),
                ("parameters", pa.string()),
            ]
        )
        selected = self.db.execute("""
            SELECT id,depart,gid,choice,json_remove(attrs,'$._sumo_parameters'),
                   json_extract(attrs,'$._sumo_parameters')
            FROM members ORDER BY depart,id
        """)
        write_batches(selected_path, selected_schema, selected, batch_size, batch_bytes)
        groups_schema = pa.schema([("gid", pa.int64()), ("choices", pa.string())])
        write_batches(
            groups_path,
            groups_schema,
            self.db.execute("SELECT gid,choices FROM groups ORDER BY gid"),
            batch_size,
            batch_bytes,
        )
