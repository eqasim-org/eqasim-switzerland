"""
Plain data holder passed around by stages.py / comparison.py / plotting.py /
interactive_map.py / lemanis.py, so those modules stay free of any direct
dependency on synpp's `context` object.

comparison_passenger_counts_geneva.py's execute(context) builds one of
these from context.config(...) values - see that file for the actual
config keys (analysis.pt.*) read from the pipeline config.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    gtfs_zip: str
    perimeter_shapefile: str
    tpg_data_path: str
    matsim_output_folder: str
    output_path: str

    # Folder holding tpg{year}_agg_workday_*.csv files, built ahead of time
    # by tpg_raw_stats.py (2024) / tpg_raw_stats_2025.py (2025) from TPG's
    # raw daily passenger counts - these are heavy, one-off jobs (~1GB+ of
    # input, several minutes) and are run by hand, not as pipeline stages.
    tpg_processed_counts_path: str

    # MATSim population sample rate (e.g. 0.1 for a 10% population). MATSim
    # boardings/alightings are divided by this before being compared
    # against TPG's full-population counts.
    input_downsampling: float = 1.0

    # Plot stages only look at "active" TPG stops: those averaging more than
    # min_stop_avg_events total passenger movements (boardings + alightings)
    # during at least min_stop_active_hours distinct hours of the day.
    min_stop_avg_events:   float = 10.0
    min_stop_active_hours: int   = 6

    # Hour-of-day window (inclusive) used by the maps: the hourly map's time
    # slider only covers these hours, and the full-day map's dots/popup
    # charts are the aggregate over just this window (not the full 24h).
    map_hour_start: int = 6
    map_hour_end:   int = 22

    # Whether to fold Leman Express (2022 counts, both LEX directions and
    # its matching MATSim boardings/alightings) into the comparison and
    # maps. Requires lemanis_csv_path when enabled.
    include_lemanis:  bool           = False
    lemanis_csv_path: Optional[str]  = None

