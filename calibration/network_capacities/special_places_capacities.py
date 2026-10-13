from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Segments:    
    street_name: str = None    
    maximum_capacity_veh_h_lane: Optional[float] = None
    osm_id: Optional[str] = None  
    maximum_speed_km_h: Optional[float] = None


CAPACITY_SEGMENTS = [
   Segments(street_name="Boulevard Helvétique", maximum_capacity_veh_h_lane=700.0),
   Segments(street_name="Boulevard du Pont-d'Arve", maximum_capacity_veh_h_lane=700.0),
   Segments(street_name="Route de Meyrin", maximum_capacity_veh_h_lane=700.0),
   Segments(street_name="Rue de la Servette", maximum_capacity_veh_h_lane=700.0),
   Segments(street_name="Quai Gustave-Ador", maximum_capacity_veh_h_lane=1000.0),
   Segments(street_name="Route de Malagnou", maximum_capacity_veh_h_lane=1100.0),
   Segments(street_name="Route Blanche", maximum_capacity_veh_h_lane=1250.0),
   Segments(street_name="Quai Charles-Page", maximum_capacity_veh_h_lane=500.0, maximum_speed_km_h=30.0),
   Segments(street_name="Quai Ernest-Ansermet", maximum_capacity_veh_h_lane=500.0, maximum_speed_km_h=30.0),
   Segments(street_name="Rue du Stand", maximum_capacity_veh_h_lane=500.0, maximum_speed_km_h=30.0),
   Segments(street_name="Rue du Mont-Blanc", maximum_capacity_veh_h_lane=800.0, maximum_speed_km_h=30.0),

]