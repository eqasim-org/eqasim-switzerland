"""Named dataframe contracts for final synthetic scenario tables.

Frames belong to their preparation-stage cache. Consumers must treat them as
read-only. Optional freight tables are None when that demand is disabled.
"""

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Facilities:
    """One row per facility: facility_id, x, y and activity_options (tuple)."""

    facilities: pd.DataFrame


@dataclass(frozen=True)
class Households:
    """One household-attribute row per household and a separate member-ID table."""

    households: pd.DataFrame
    members: pd.DataFrame
    maximum_car_class: int


@dataclass(frozen=True)
class Vehicles:
    """Vehicle type definitions and vehicles, including their export attributes."""

    types: pd.DataFrame
    vehicles: pd.DataFrame


@dataclass(frozen=True)
class Population:
    """Sorted persons, activities and owned vehicles, plus optional freight demand."""

    persons: pd.DataFrame
    activities: pd.DataFrame
    vehicles: pd.DataFrame
    freight_trips: pd.DataFrame | None = None
    freight_vehicles: pd.DataFrame | None = None
    lcv_trips: pd.DataFrame | None = None
    lcv_vehicles: pd.DataFrame | None = None


def implementation_token(stage_file, *helpers):
    """Include shared schema/contracts and imported helpers in preparation caches."""
    import hashlib
    from pathlib import Path

    digest = hashlib.sha256()
    paths = {
        Path(stage_file),
        Path(__file__),
        Path(__file__).parent / "schema.py",
    }
    paths.update(Path(helper) for helper in helpers)
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
