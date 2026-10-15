"""Read the common JAX/Biogeme export without importing either estimator."""
from pathlib import Path
from functools import lru_cache
from copy import deepcopy
from mode_choice.defaults import PARAMETER_CACHE_SIZE
import math
import yaml


def load(context):
    mode_path, cost_path = context.stage("dmc.params")
    signatures = []
    for path in (mode_path, cost_path):
        path = Path(path).resolve()
        stat = path.stat()
        signatures.append((str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino))
    # Callers receive their own mapping; nobody can mutate another run's cache.
    return deepcopy(_read_cached(tuple(signatures)))


@lru_cache(maxsize=PARAMETER_CACHE_SIZE)
def _read_cached(signatures):
    result = {}
    for path, *_ in signatures:
        with Path(path).open() as stream:
            values = yaml.safe_load(stream)
        # it must be a dict
        if not isinstance(values, dict):
            raise ValueError(f"Expected a parameter mapping in {path}")
        # check that all values are finite numbers (or dicts of finite numbers)
        for key, value in values.items():
            if isinstance(value, dict):
                if not all(math.isfinite(float(v)) for v in value.values()):
                    raise ValueError(f"Non-finite parameter in {path}: {key}")
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid numeric parameter in {path}: {key}")

        result.update(values)

    return result
