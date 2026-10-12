from dataclasses import dataclass
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

    # The MATSim counts of the hours 0 to night_hours-1 are replaced by the ones of the hours 24 to 24+night_hours-1
    # (the runs after midnight), see tpg_data.read_matsim_counts. 0 keeps the hours as they are.
    night_hours: int = 0

    map_hour_start: int = 6
    map_hour_end:   int = 22

    # Where the MATSim passenger counts come from: the file pt_passenger_counts.csv.gz of the simulation ("file"),
    # the boardings and alightings of output_events.xml.gz ("events"), or the file if it exists ("auto")
    counts_source: str = "auto"

    # Whether to fold Leman Express into the comparison and
    # maps. Requires lemanis_path (the folder of the Lemanis counts of lemanis_year) when enabled.
    include_lemanis:  bool           = False
    lemanis_year:     int            = 2022
    lemanis_path:     Optional[str]  = None

    # SBB passenger frequency of the stations (passagierfrequenz.csv), shown on the stop map when set
    sbb_path: Optional[str] = None


    # Stops of the simulated schedule (output_transitSchedule.xml.gz) that the GTFS does not
    # know, e.g. the French stops of the Leman Express, are matched to GTFS stops by name and
    # position or added, so that their passengers are not dropped.
    use_schedule_stops: bool = True
