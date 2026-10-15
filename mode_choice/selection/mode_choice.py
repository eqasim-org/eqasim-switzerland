"""Reusable file-based mode choice, shared by ordinary runs and calibration."""
from pathlib import Path
import numpy as np
import polars as pl

from mode_choice.defaults import MODES, CALIBRATION_CANTON_COLUMN, RUN_DEFAULTS
from mode_choice.tours.prepared import PreparedBatch
from mode_choice.selection.sampler import TourSampler


class ModeChoice:
    """Select modes from person-batch Parquet files.

    Each run reads the compact batch files in order. Optional in-memory caching
    retains grouping, split plans and random buffers between evaluations; no
    routing, costs or base utilities are recomputed here. ``delta`` adjusts the
    five ASCs in MODES order; ``canton`` restricts an independent canton fit.
    """

    def __init__(self, batch_paths, seed, sampler=None, *,
                 cache_prepared=RUN_DEFAULTS["mode_choice_cache_prepared_batches"]):
        self.paths = tuple(Path(path) for path in batch_paths)
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Duplicate tour batch paths")
        self.seed = seed
        self.sampler = sampler if sampler is not None else TourSampler()
        self.cache_prepared = cache_prepared
        self._prepared = {}
        self.last_counts = np.zeros(len(MODES), dtype=np.int64)

    def run(self, delta=None, *, canton=None, draw_index=0):
        """Return a Polars table with exactly trip_id and mode."""
        outputs = []
        self.last_counts = np.zeros(len(MODES), dtype=np.int64)
        try:
            for path in self.paths:
                stat = path.stat()
                token = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
                trips = pl.read_parquet(path)
                if canton is not None:
                    trips = trips.filter(pl.col(CALIBRATION_CANTON_COLUMN) == canton)
                if trips.is_empty():
                    continue
                key = (path, canton)
                cached = self._prepared.get(key)
                if cached is not None and cached[0] != token:
                    cached[1].close()
                    del self._prepared[key]
                    cached = None
                prepared = cached[1] if cached else PreparedBatch(trips, self.seed, self.sampler, assume_sorted=True)
                if self.cache_prepared and cached is None:
                    self._prepared[key] = (token, prepared)
                try:
                    codes = prepared.sample_codes(delta, draw_index=draw_index)
                    self.last_counts += prepared.counts(codes)
                    outputs.append(trips.select("trip_id").with_columns(pl.Series("mode", np.asarray(MODES)[codes])))
                finally:
                    if not self.cache_prepared:
                        prepared.close()
            result = pl.concat(outputs) if outputs else pl.DataFrame(schema={"trip_id": pl.String, "mode": pl.String})
            if result["trip_id"].n_unique() != result.height:
                raise ValueError("Trip IDs occur in multiple tour batches")
            return result
        except BaseException:
            self.clear_cache()
            raise

    def cantons(self):
        """Check residence consistency once before independent canton calibration."""
        cantons = set()
        for path in self.paths:
            frame = pl.read_parquet(path, columns=["person_id", CALIBRATION_CANTON_COLUMN])
            if frame[CALIBRATION_CANTON_COLUMN].null_count():
                raise ValueError("Canton calibration requires every person's residence canton")
            if frame.group_by("person_id").agg(pl.col(CALIBRATION_CANTON_COLUMN).n_unique()).filter(
                    pl.col(CALIBRATION_CANTON_COLUMN) != 1).height:
                raise ValueError("A person's trips have inconsistent residence cantons")
            cantons.update(frame[CALIBRATION_CANTON_COLUMN].unique().to_list())
        return sorted(cantons)

    def clear_cache(self):
        for _, batch in self._prepared.values():
            batch.close()
        self._prepared.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear_cache()
