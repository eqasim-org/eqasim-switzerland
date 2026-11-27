"""Atomic JSON checkpoints for the latest measurement of each cached stage."""

import fcntl
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("synpp")


class MonitorStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "state.json"
        self.lock_path = self.directory / "state.lock"

    def load(self):
        """Read complete records; a broken diagnostic file never stops a model."""
        try:
            data = json.loads(self.path.read_text())
            if data.get("version") != 1 or not isinstance(data.get("stages"), dict):
                raise ValueError("unsupported monitoring cache schema")
            return {
                key: row
                for key, row in data["stages"].items()
                if isinstance(row, dict) and {"stage", "cache", "status"}.issubset(row)
            }
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError, AttributeError) as error:
            logger.warning("Cannot read SUMO monitoring cache %s: %s", self.path, error)
            return {}

    def save(self, record):
        """Merge one stage under a file lock, then atomically replace the JSON."""
        self._update({record["cache"]: record})

    def _update(self, updates):
        pending = None
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self.load()
            records.update(updates)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=self.directory,
                    prefix="state_",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    pending = Path(stream.name)
                    json.dump(
                        {"version": 1, "stages": records},
                        stream,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    stream.write("\n")
                pending.replace(self.path)
            finally:
                if pending is not None:
                    pending.unlink(missing_ok=True)

    def import_reports(self):
        """Seed a new cache from reports created before persistent monitoring."""
        records = {}
        for path in sorted(self.directory.glob("*/performance.json")):
            try:
                data = json.loads(path.read_text())
                for index, source in enumerate(data.get("stages", [])):
                    row = dict(source)
                    if not {"stage", "cache", "status"}.issubset(row):
                        continue
                    row.setdefault("run_id", path.parent.name)
                    for name in ("slurm_job_id", "allocated_threads"):
                        row.setdefault(name, data.get(name))
                    # New combined reports carry an explicit segment index;
                    # old reports used elapsed process time for all samples.
                    if "timeline_start_seconds" in row:
                        row["samples"] = [
                            dict(
                                seconds=sample["seconds"]
                                - row["timeline_start_seconds"],
                                rss_gib=sample["rss_gib"],
                            )
                            for sample in data.get("samples", [])
                            if sample.get("stage_index") == index
                        ]
                    else:
                        start = row.get("start", 0)
                        end = (
                            start + row["seconds"]
                            if row.get("seconds") is not None
                            else float("inf")
                        )
                        row["samples"] = [
                            dict(
                                seconds=sample["seconds"] - start,
                                rss_gib=sample["rss_gib"],
                            )
                            for sample in data.get("samples", [])
                            if start <= sample["seconds"] <= end
                        ]
                    records[row["cache"]] = row
            except (OSError, ValueError, TypeError, KeyError) as error:
                logger.warning(
                    "Cannot import SUMO monitoring report %s: %s", path, error
                )
        if records:
            self._update(records)
        return records
