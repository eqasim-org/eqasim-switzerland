"""Low-frequency process-tree sampling shared by pipeline stage decorators."""

import atexit
import json
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from scripts.monitoring.storage import MonitorStore

logger = logging.getLogger("synpp")
_sessions = {}
_registry_lock = threading.Lock()


def process_tree():
    """Read Linux process counters without requiring a monitoring dependency.

    RSS is summed over Python and its descendants. Shared pages may be counted
    more than once; this is not Slurm MaxRSS or proportional memory usage.
    """
    pending = [os.getpid()]
    seen = set()
    result = {}
    ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            root = Path("/proc") / str(pid)
            stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
            rss = int((root / "statm").read_text().split()[1]) * page_size
            io = dict(
                line.split(": ", 1) for line in (root / "io").read_text().splitlines()
            )
            # Include start ticks in identity so reused PIDs cannot corrupt totals.
            result[(pid, stat[19])] = (
                rss,
                (int(stat[11]) + int(stat[12])) / ticks,
                int(io["read_bytes"]),
                int(io["write_bytes"]),
            )
            # A subprocess may be launched by any worker thread, not only main.
            for task in (root / "task").iterdir():
                pending.extend(
                    int(value) for value in (task / "children").read_text().split()
                )
        except (OSError, ValueError, IndexError, KeyError):
            continue  # Processes can finish between reading their counters.
    return result


class RunMonitor:
    """One sampler/renderer thread per pipeline execution."""

    def __init__(self, working_directory, pipeline_token=None, module="sumo"):
        if not module.isidentifier():
            raise ValueError("Monitoring module must be a Python package name")
        self.module = module
        self.started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = f"{stamp}_{os.getpid()}"
        self.pipeline_token = pipeline_token
        self.store = MonitorStore(Path(working_directory) / f"{module}_monitoring")
        self.directory = self.store.directory / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        restored = (
            self.store.load()
            if self.store.path.exists()
            else self.store.import_reports()
        )
        self.records = []
        for row in restored.values():
            record = dict(row)
            record["_samples"] = record.pop("samples", [])
            if record["status"] == "running":
                record["status"] = "interrupted"
            if record.get("seconds") is None:
                record["seconds"] = max(
                    (point["seconds"] for point in record["_samples"]), default=0.0
                )
            self.records.append(record)
        self.exports = set()
        self.active = []
        self.lock = threading.RLock()
        self.changed = threading.Event()
        self.stopping = threading.Event()
        self.dirty = True
        self.thread = threading.Thread(
            target=self._work, name=f"{module}-monitor", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)
        logger.info("%s monitoring: %s", module, self.directory)

    def begin(self, name, cache):
        record = dict(
            stage=name,
            cache=str(Path(cache).resolve()),
            run_id=self.run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            allocated_threads=os.environ.get("SLURM_CPUS_PER_TASK"),
            start=time.monotonic() - self.started,
            seconds=None,
            peak_rss_gib=0.0,
            cpu_seconds=0.0,
            read_bytes=0,
            write_bytes=0,
            status="running",
            _samples=[],
            _counters={},
            _started=time.monotonic(),
        )
        with self.lock:
            # A repeated stage replaces its earlier attempt in the same row.
            index = next(
                (
                    i
                    for i, old in enumerate(self.records)
                    if old["cache"] == record["cache"]
                ),
                None,
            )
            if index is None:
                self.records.append(record)
            else:
                self.records[index] = record
            self.active.append(record)
            self._sample()
            self._save(record)
            self.dirty = True
        self.changed.set()
        return record

    def finish(self, record, status):
        with self.lock:
            self._sample()
            record.update(seconds=time.monotonic() - record["_started"], status=status)
            self.active.remove(record)
            self._save(record)
            self.dirty = True
        logger.info(
            "Pipeline stage %s: %s; %.1fs; peak sampled process-tree RSS %.2f GiB",
            record["stage"],
            status,
            record["seconds"],
            record["peak_rss_gib"],
        )
        self.changed.set()

    def export(self, destination):
        with self.lock:
            self.exports.add(Path(destination))
            self.dirty = True
        self.changed.set()

    def _sample(self):
        if not self.active:
            return
        counters = process_tree()
        rss = sum(value[0] for value in counters.values()) / 1024**3
        for record in self.active:
            record["_samples"].append(
                dict(seconds=time.monotonic() - record["_started"], rss_gib=rss)
            )
            record["peak_rss_gib"] = max(record["peak_rss_gib"], rss)
            previous = record["_counters"]
            initial = not previous
            for identity, (_, cpu, read, written) in counters.items():
                # New child counters start at zero; already running processes
                # establish their baseline at stage entry.
                baseline = previous.get(
                    identity, (cpu, read, written) if initial else (0, 0, 0)
                )
                record["cpu_seconds"] += max(0, cpu - baseline[0])
                record["read_bytes"] += max(0, read - baseline[1])
                record["write_bytes"] += max(0, written - baseline[2])
                previous[identity] = (cpu, read, written)

    def _record(self, record):
        result = {
            key: value for key, value in record.items() if not key.startswith("_")
        }
        if result.get("seconds") is None:
            result["seconds"] = time.monotonic() - record["_started"]
        return result

    def _save(self, record):
        # A small synchronous checkpoint at stage boundaries survives an immediate
        # job failure; sampling and figure rendering remain on the worker thread.
        try:
            result = self._record(record)
            result["samples"] = list(record["_samples"])
            self.store.save(result)
        except Exception:
            logger.exception(
                "Pipeline monitoring could not checkpoint %s", record["stage"]
            )

    def _checkpoint(self):
        with self.lock:
            for record in self.active:
                self._save(record)

    def snapshot(self):
        with self.lock:
            records, samples = [], []
            offset = 0.0
            for index, record in enumerate(self.records):
                row = self._record(record)
                row["current_run"] = row.get("run_id") == self.run_id
                row["timeline_start_seconds"] = offset
                records.append(row)
                samples.extend(
                    dict(
                        seconds=offset + point["seconds"],
                        rss_gib=point["rss_gib"],
                        stage_index=index,
                    )
                    for point in record["_samples"]
                )
                offset += row["seconds"] or 0.0
            return dict(
                stages=records,
                samples=samples,
                module=self.module,
                run_id=self.run_id,
                pid=os.getpid(),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                allocated_threads=os.environ.get("SLURM_CPUS_PER_TASK"),
                memory_measure="Sum of sampled RSS for Python and descendant processes (shared pages may repeat)",
                scope="Latest measured attempt per stage cache; includes retained measurements from earlier runs",
                timeline="Stage execution time combined across retained and current measurements; downtime and replaced attempts excluded",
                sample_interval_seconds=2,
            )

    def _publish(self):
        from scripts.monitoring.plots import performance_figure

        data = self.snapshot()
        if not data["stages"]:
            return
        temporary = self.directory / "performance.json.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(self.directory / "performance.json")
        performance_figure(data, self.directory / "performance.png")
        with self.lock:
            exports = list(self.exports)
        for folder in [self.store.directory, *exports]:
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("performance.json", "performance.png"):
                source = self.directory / name
                pending = folder / (name + f".{self.run_id}.tmp")
                shutil.copy2(source, pending)
                pending.replace(folder / name)

    def _work(self):
        last_plot = -float("inf")
        last_checkpoint = time.monotonic()
        while not self.stopping.is_set():
            self.changed.wait(2)
            self.changed.clear()
            try:
                with self.lock:
                    self._sample()
                    publish = self.dirty and time.monotonic() - last_plot >= 5
                    if publish:
                        self.dirty = False
                if time.monotonic() - last_checkpoint >= 30:
                    self._checkpoint()
                    last_checkpoint = time.monotonic()
                if publish:
                    self._publish()
                    last_plot = time.monotonic()
            except Exception:
                logger.exception("Pipeline monitoring could not update its diagnostics")
        try:
            self._checkpoint()
            self._publish()
        except Exception:
            logger.exception(
                "Pipeline monitoring could not write its final diagnostics"
            )

    def close(self):
        atexit.unregister(self.close)
        self.stopping.set()
        self.changed.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=20)


def session(context, module=None):
    working = getattr(context, "working_directory", None)
    if working is None:
        return None  # Standalone fixtures/calls need no persistent pipeline monitor.
    cache_name = Path(context.path()).name if hasattr(context, "path") else ""
    module = module or (cache_name.split(".", 1)[0] if "__" in cache_name else "sumo")
    key = (str(Path(working).resolve()), module)
    # Normal Synpp execution shares cache across stage contexts; debug execution
    # instead shares cache_hashes. This also detects notebook reruns in one PID.
    token = getattr(context, "cache_hashes", getattr(context, "cache", None))
    with _registry_lock:
        manager = _sessions.get(key)
        if manager is not None and (
            manager.stopping.is_set()
            or (token is not None and manager.pipeline_token is not token)
        ):
            manager.close()
            manager = None
        if manager is None:
            manager = _sessions[key] = RunMonitor(
                key[0], pipeline_token=token, module=module
            )
        return manager


@contextmanager
def monitor(context, name=None):
    """Measure a stage or a manually scoped operation; never mask its exception."""
    manager = None
    record = None
    try:
        name = name or Path(context.path()).name.split("__")[0]
        manager = session(context, name.split(".", 1)[0])
        if manager is not None:
            record = manager.begin(name, context.path())
    except Exception:
        logger.exception("Pipeline monitoring could not start")
    status = "failed"
    try:
        yield manager
        status = "completed"
    finally:
        if record is not None:
            try:
                manager.finish(record, status)
            except Exception:
                logger.exception("Pipeline monitoring could not finish")


def monitored(execute):
    """Decorate an execute(context) while preserving Synpp's function metadata."""

    @wraps(execute)
    def wrapped(context):
        # Backend adapters may be called inside another stage with its context.
        # Only measure the actual stage, avoiding duplicate/nested cache records.
        try:
            cache = (
                Path(context.path()).name
                if getattr(context, "working_directory", None)
                else ""
            )
        except Exception:
            cache = ""
        if "__" in cache and cache.split("__", 1)[0] != execute.__module__:
            return execute(context)
        with monitor(context, execute.__module__):
            return execute(context)

    return wrapped


def export_monitor(context, destination, module=None):
    try:
        manager = session(context, module)
        if manager is not None:
            manager.export(destination)
    except Exception:
        logger.exception(
            "Pipeline monitoring could not register its export destination"
        )
