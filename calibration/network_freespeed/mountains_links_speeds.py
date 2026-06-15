from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Segments:
    origin_x: float      # lon
    origin_y: float      # lat
    destination_x: float # lon
    destination_y: float # lat
    speed: float         # km/h
    name: Optional[str] = None


ROUTING_PENALTY_M_PER_S = 0.1
MIN_FREESPEED_M_S = 20.0 / 3.6 # 20 km/h in m/s
SPEED_TOLERANCE_KMH = 2.0


"""
IMPORANT:
1 - These are obtained from google map
2- if you want to include new values, you need to be sure that that way will be used, otherwise, if it is a very long road, divid it into pieces and assign the speed to one of the pieces, otherwise, if it is a very short road, assign the speed to the whole road.
3- The speed is in km/h, it will be converted to m/s in the code
4- The coordinates are in lat/lon, they will be converted to the same coordinate system
"""
SPEED_SEGMENTS = [
    Segments(origin_x=46.31875990699816, origin_y=6.977670373513515, destination_x=46.68740828906988, destination_y=7.616250694145221, speed=52.0),
    Segments(origin_x=46.34384662386212, origin_y=8.012117641928883, destination_x=46.58222637582872, destination_y=8.36368014102935, speed=49.0),
    Segments(origin_x=46.56721314077649, origin_y=8.3624963858487, destination_x=46.70121946160386, destination_y=8.231846190442948, speed=48.0),
    Segments(origin_x=46.56221672709718, origin_y=8.361458418531196, destination_x=46.61822698674738, destination_y=8.566212705353886, speed=48.0),
    Segments(origin_x=46.63483974936093, origin_y=8.596478752802598, destination_x=46.82839169613524, destination_y=9.416427771180434, speed=58.0),
    Segments(origin_x=46.74916448327385, origin_y=10.079064101675094, destination_x=46.8841097211709, destination_y=10.463875068984343, speed=57.0),
    Segments(origin_x=46.74916448327385, origin_y=10.079064101675094, destination_x=46.8841097211709, destination_y=10.463875068984343, speed=57.0),
    Segments(origin_x=46.75488363792933, origin_y=10.078982148159396, destination_x=46.81135794788439, destination_y=9.844220078289682, speed=51.0),
    Segments(origin_x=46.81903463801564, origin_y=9.84875337168069, destination_x=46.974303626418994, destination_y=9.580739876811732, speed=67.0),
    Segments(origin_x=46.63852353106998, origin_y=10.458587540546588, destination_x=46.696716280731366, destination_y=10.102218522903733, speed=52.0),
    Segments(origin_x=46.69551920211572, origin_y=10.091779100636042, destination_x=46.252807944342536, destination_y=10.142274396770771, speed=59.0),
    Segments(origin_x=46.455127839250174, origin_y=9.793727480626487, destination_x=46.34396269840964, destination_y=9.523771858983118, speed=48.0),
    Segments(origin_x=46.46852805746778, origin_y=9.80128023319728, destination_x=46.84000027260411, destination_y=9.53892714766013, speed=53.0),
    Segments(origin_x=46.7066097302365, origin_y=8.230991381500182, destination_x=46.73937100775862, destination_y=8.372347266163942, speed=51.0),
    Segments(origin_x=46.73921265149859, origin_y=8.371303593035595, destination_x=46.731594844542364, destination_y=8.427383348979117, speed=39.0),
    Segments(origin_x=46.706060823238374, origin_y=8.59948876876654, destination_x=46.738513605666, destination_y=8.528358375653672, speed=47.0),
    Segments(origin_x=46.750781122800426, origin_y=8.10906732131053, destination_x=46.77735617920117, destination_y=8.157304168110935, speed=43.0),
    Segments(origin_x=46.619171609146655, origin_y=7.377684729094028, destination_x=46.61145934259408, destination_y=7.111893237815448, speed=53.0),
    Segments(origin_x=46.458768179027956, origin_y=7.120832052518118, destination_x=46.601436998074526, destination_y=7.075072807332628, speed=61.0),
    Segments(origin_x=46.44076897946171, origin_y=7.273024984512517, destination_x=46.36531535323616, destination_y=7.075824845821957, speed=49.0),
    Segments(origin_x=46.34632545112809, origin_y=7.155401153177598, destination_x=46.29598532773282, destination_y=7.065438528523965, speed=33.0),
    Segments(origin_x=46.30212037986872, origin_y=7.04645461904084, destination_x=46.298088061541236, destination_y=6.999161909811261, speed=43.0),
    Segments(origin_x=46.289718041536645, origin_y=7.065416452740825, destination_x=46.253734945529324, destination_y=7.022750998303633, speed=38.0),
    Segments(origin_x=45.95694160364035, origin_y=7.2092839338067085, destination_x=46.08309173172344, destination_y=7.052969274420094, speed=64.0),
]