from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    gtfs_zip: str
    perimeter_shapefile: str
    tpg_data_path: str
    matsim_output_folder: str
    output_path: str

    # Folder holding tpg{year}_agg_workday_*.csv files
    tpg_processed_counts_path: str

    # MATSim population sample rate
    input_downsampling: float = 1.0

    min_stop_avg_events:   float = 10.0
    min_stop_active_hours: int   = 6

    map_hour_start: int = 6
    map_hour_end:   int = 22

    # Whether to fold Leman Express into the comparison and
    # maps. Requires lemanis_csv_path when enabled.
    include_lemanis:  bool           = False
    lemanis_csv_path: Optional[str]  = None


    # Stops of the simulated schedule (output_transitSchedule.xml.gz) that the GTFS does not
    # know, e.g. the French stops of the Leman Express, are matched to GTFS stops by name and
    # position or added, so that their passengers are not dropped.
    use_schedule_stops: bool = True
    stop_alias: dict = field(default_factory = dict)  # filled by stages: schedule stop id -> GTFS stop id
