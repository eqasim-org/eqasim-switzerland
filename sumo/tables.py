"""The external-table boundary and Arrow IPC cache contract for SUMO stages."""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import shapely


def as_polars(value):
    """Convert external tables once; geometry is portable WKB, never Python Object."""
    if isinstance(value, pl.DataFrame):
        frame = value.clone()
        for name, dtype in frame.schema.items():
            if name == "geometry" and dtype == pl.String:
                frame = frame.with_columns(pl.Series(name, shapely.to_wkb(shapely.from_wkt(frame[name].to_numpy())), dtype=pl.Binary))
            elif name == "geometry" and dtype == pl.Null:
                frame = frame.with_columns(pl.col(name).cast(pl.Binary))
            elif dtype == pl.Object:
                values = frame[name].to_list()
                if name == "geometry":
                    frame = frame.with_columns(pl.Series(name, shapely.to_wkb(values), dtype=pl.Binary))
                else:
                    from sumo.scenario.population.tools import scalar_text
                    frame = frame.with_columns(pl.Series(name, [scalar_text(item) for item in values], dtype=pl.String))
        return frame
    if isinstance(value, pl.LazyFrame):
        return as_polars(value.collect())
    if isinstance(value, (str, Path)):
        return as_polars(read_table(value))
    if isinstance(value, (pa.Table, pa.RecordBatch)):
        return as_polars(pl.from_arrow(value))
    # Pandas/GeoPandas are accepted only here, at the boundary to external stages.
    if hasattr(value, "columns") and hasattr(value, "iloc"):
        if len(set(value.columns)) != len(value.columns):
            raise ValueError("External SUMO table has duplicate column names")
        columns = []
        for name in value.columns:
            series = value[name]
            if name == "geometry" or str(series.dtype) == "geometry":
                geometry = series.to_numpy(copy=True)
                import pandas as pd
                geometry[pd.isna(geometry)] = None
                try:
                    wkb = shapely.to_wkb(geometry)
                except TypeError:
                    strings = np.asarray([isinstance(item, str) for item in geometry])
                    if strings.any():
                        geometry[strings] = shapely.from_wkt(geometry[strings])
                    binary = np.asarray([isinstance(item, (bytes, bytearray, memoryview)) for item in geometry])
                    if binary.any():
                        geometry[binary] = shapely.from_wkb(geometry[binary])
                    wkb = shapely.to_wkb(geometry)
                columns.append(pl.Series(name, wkb, dtype=pl.Binary))
            else:
                try:
                    columns.append(pl.from_pandas(series.rename(name), nan_to_null=True))
                except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
                    # Mixed scalar metadata must remain serializable in IPC/XML.
                    from sumo.scenario.population.tools import scalar_text
                    columns.append(pl.Series(name, [scalar_text(item) for item in series], dtype=pl.String))
        frame = pl.DataFrame(columns)
    else:
        frame = pl.DataFrame(value)
    return frame.with_columns(pl.col(name).cast(pl.String) for name, dtype in frame.schema.items()
                              if dtype == pl.Categorical or isinstance(dtype, pl.Enum))


def write_table(frame, path):
    """Uncompressed Arrow IPC trades disk space for fast memory-mapped Polars reads."""
    frame = as_polars(frame)
    if any(dtype == pl.Object for dtype in frame.schema.values()):
        raise TypeError("SUMO table caches cannot contain Python Object columns")
    path = Path(path).resolve()
    frame.write_ipc(path, compression="uncompressed")
    return str(path)


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_ipc(path, memory_map=True, rechunk=False)


def id_expression(name, dtype):
    value = pl.col(name)
    if dtype.is_float():
        return pl.when(value.is_finite() & (value == value.floor())).then(
            value.cast(pl.Int64, strict=False).cast(pl.String)).when(value.is_finite()).then(
            value.cast(pl.String)).otherwise(None).alias(name)
    return value.cast(pl.String).alias(name)
