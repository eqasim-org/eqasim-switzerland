import gzip
import re

STOP_REF = re.compile(r'<stop\s[^>]*refId="([^"]*)"')
RELATION = re.compile(r'<relation\s[^>]*fromStop="([^"]*)"[^>]*toStop="([^"]*)"')


def _open(path, mode):
    return gzip.open(path, mode, encoding = "utf-8") if path.endswith(".gz") else open(path, mode, encoding = "utf-8")


def remove_unserved_transfers(schedule_path, output_path):
    """Writes a copy of the schedule without the minimal transfer times from or to a stop that no route serves.

    SwissRailRaptor adds every minimal transfer time to its transfers and then looks up the route stops of both of
    its stops: a stop facility that no route serves has none, and the router fails with a NullPointerException
    (SwissRailRaptorData.calculateRouteStopTransfers). Such transfers come from the GTFS transfers of stops served on
    other days than the one converted, or whose routes were dropped since. The schedule is read line by line, as
    MATSim writes one element per line.

    Returns (number of transfers removed, number of transfers).
    """
    served = set()
    with _open(schedule_path, "rt") as f:
        for line in f:
            match = STOP_REF.search(line)
            if match:
                served.add(match.group(1))

    removed = total = 0
    with _open(schedule_path, "rt") as f, _open(output_path, "wt") as out:
        for line in f:
            match = RELATION.search(line)
            if match:
                total += 1
                if match.group(1) not in served or match.group(2) not in served:
                    removed += 1
                    continue
            out.write(line)

    return removed, total
