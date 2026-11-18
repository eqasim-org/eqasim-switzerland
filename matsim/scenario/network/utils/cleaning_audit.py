"""Streaming, original-link provenance for each network-cleaning operation."""
from collections import Counter
import gzip
import json
from pathlib import Path
import re


def original_ids(identifier, attributes):
    value = attributes.get("old_link_id") if isinstance(attributes, dict) else None
    return list(dict.fromkeys(filter(None, re.split(r"[_,;\s]+", str(value))))) if value else [str(identifier)]


class CleaningAudit:
    def __init__(self, path):
        self.path = Path(path)
        self.counts = Counter()
        self.original_counts = Counter()
        self.stream = None

    def __enter__(self):
        self.stream = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=3)
        return self

    def __exit__(self, *args):
        self.stream.close()

    def record(self, identifier, attributes, reason, action="removed", mode=None, survivor=None):
        ids = original_ids(identifier, attributes)
        event = dict(link_id=str(identifier), original_link_ids=ids, reason=reason, action=action)
        if mode:
            event["mode"] = mode
        if survivor is not None:
            event["survivor"] = str(survivor)
        self.stream.write(json.dumps(event, separators=(",", ":"))+"\n")
        self.counts[reason] += 1
        self.original_counts[reason] += len(ids)

    def rows(self, rows, reason, action="removed", mode=None):
        for identifier, attributes in zip(rows.link_id, rows.attributes):
            self.record(identifier, attributes, reason, action, mode)

    def summary(self):
        return {"file": self.path.name, "schema": 1, "events": dict(self.counts),
                "original_links": dict(self.original_counts)}
