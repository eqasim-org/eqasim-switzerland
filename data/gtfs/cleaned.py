import data.gtfs.utils as gtfs
import os, pathlib
import pandas as pd
import logging

logger = logging.getLogger("synpp")

"""
This file reads GTFS schedule.

The steps mirror the standalone merge script (merge_gtfs.py): every feed is cleaned, the trips that
a block_id splits between agencies are merged, the Swiss consolidated feed and the standalone SNCF
feed (if both are present) are made consistent with each other, then the feeds are merged and the
cross-border stations are bridged.
"""

def configure(context):
    context.config("data_path")
    context.config("gtfs_name", default = "")
    context.config("gtfs_folder", default = "gtfs")

    # Optional: keep only the services that are active on this date (YYYY-MM-DD) in every feed, e.g. to
    # build a "typical day" feed. Empty: keep all the services (pt2matsim then selects the day with gtfs_date).
    context.config("gtfs_filter_date", default = "")

    # Optional: the TPG own feed (one typical working day), which replaces the TPG lines (agency 881) of the Swiss feed.
    # A file name in gtfs_folder, or an absolute path; it is not read as one more feed. Empty: the Swiss feed is kept.
    context.config("gtfs_tpg_file", default = "")
    # TPG lines of the Swiss feed that the TPG feed does not have and that are removed rather than kept (the airport
    # lines, at the request of TPG). Only used with gtfs_tpg_file.
    context.config("gtfs_tpg_removed_lines", default = ["A1", "A2", "A3", "A4", "A5", "A6"])

    # Transfer times added between the stops closer than this without a rule (see gtfs.add_nearby_transfers). MATSim
    # lets the passengers walk between stops up to its beeline walk distance (transitRouter.maxBeelineWalkConnectionDistance,
    # 300 m): both should be the same. 0: no times added.
    context.config("gtfs_transfer_max_distance", default = 300.0)
    context.config("gtfs_transfer_walk_speed", default = 1.0) # m/s, with the detour factor below
    context.config("gtfs_transfer_detour_factor", default = 1.3)


def execute(context):
    gtfs_folder = context.config("gtfs_folder")
    # gtfs_folder is relative to data_path, or absolute
    all_input_files = get_input_files(os.path.join(context.config("data_path"), gtfs_folder))

    if context.config("gtfs_name") == "":
        # Keep all files
        input_files = all_input_files

    else:
        # Look for the specified file name
        input_files = []
        for file in all_input_files:
            if context.config("gtfs_name") in file:
                input_files = [file]

        # If the specified file name was not found
        if len(input_files) == 0:
            raise RuntimeError("Did not find any GTFS data matching the specified name.")

    filter_date = context.config("gtfs_filter_date")

    tpg_file = context.config("gtfs_tpg_file")
    if tpg_file:
        tpg_file = tpg_file if os.path.isabs(tpg_file) else os.path.join(context.config("data_path"), gtfs_folder, tpg_file)
        if not os.path.exists(tpg_file):
            raise RuntimeError(f"gtfs_tpg_file not found: {tpg_file}")
        input_files = [f for f in input_files if os.path.abspath(f) != os.path.abspath(tpg_file)]

    # Load feeds
    feeds = []
    for path in input_files:
        logger.info(path)
        feed = gtfs.read_feed(path)
        feed = gtfs.clean_feed(feed)

        # This was fixed in pt2matsim, so we can remove one a new release (> 20.7) is available.
        feed = gtfs.despace_stop_ids(feed) # Necessary as MATSim does not like stops/links with spaces

        if filter_date:
            feed = gtfs.filter_feed_to_date(feed, filter_date)

        feed = gtfs.merge_cross_agency_blocks(feed)
        feeds.append((path, feed))

    swiss_idx = next((i for i, (p, _) in enumerate(feeds) if "fp" in os.path.basename(p)), None)

    # The TPG lines of the Swiss feed are replaced by the ones of the TPG own feed
    if tpg_file:
        if swiss_idx is None:
            raise RuntimeError("gtfs_tpg_file is set but there is no Swiss feed (file name with 'fp') to replace the TPG lines in")
        logger.info(f"Reading the TPG feed {tpg_file}")
        tpg_feed = gtfs.despace_stop_ids(gtfs.clean_feed(gtfs.read_feed(tpg_file)))
        swiss_feed = gtfs.replace_agency_lines(feeds[swiss_idx][1], tpg_feed, agency_id = "881", service_prefix = "TPG_",
                                               removed_lines = context.config("gtfs_tpg_removed_lines"))
        if filter_date:
            swiss_feed = gtfs.filter_feed_to_date(swiss_feed, filter_date)
        feeds[swiss_idx] = (feeds[swiss_idx][0], swiss_feed)

    # The Swiss consolidated feed (filename hint "fp") and the standalone
    # SNCF feed (identified by agency name, to tell it apart from other
    # French regional/bus feeds) are the two sides of the cross-border
    # split that merge_cross_agency_blocks() can't see, since block_id is
    # only meaningful within a single feed — see extend_swiss_trips_with_sncf().
    sncf_idx = next(
        (i for i, (p, f) in enumerate(feeds)
         if i != swiss_idx and "agency" in f
         and f["agency"]["agency_name"].astype(str).str.contains("SNCF VOYAGEURS", case = False, na = False).any()),
        None,
    )

    if swiss_idx is not None and sncf_idx is not None:
        swiss_feed, sncf_feed = gtfs.drop_duplicate_sncf_trips(feeds[swiss_idx][1], feeds[sncf_idx][1])
        swiss_feed, sncf_feed = gtfs.extend_swiss_trips_with_sncf(swiss_feed, sncf_feed)
        feeds[swiss_idx] = (feeds[swiss_idx][0], swiss_feed)
        feeds[sncf_idx] = (feeds[sncf_idx][0], sncf_feed)
    else:
        logger.info("extend_swiss_trips_with_sncf: could not identify both a Swiss and a standalone SNCF feed, skipping.")

    sources = [os.path.basename(path) for path, _ in feeds]
    ids_by_feed = {slot: [set(f[slot][column].astype(str)) if slot in f else set() for _, f in feeds]
                   for slot, column in (("routes", "route_id"), ("stops", "stop_id"))}
    feeds = [f for _, f in feeds]

    # Merge feeds
    merged_feed = gtfs.merge_feeds(feeds) if len(feeds) > 1 else feeds[0]
    merged_feed = gtfs.rename_swiss_leman_routes(merged_feed)
    transfers_feeds = _transfer_pairs(merged_feed)
    merged_feed = gtfs.bridge_cross_source_stations(merged_feed)
    transfers_bridged = merged_feed["transfers"].copy() if "transfers" in merged_feed else None
    merged_feed = gtfs.filter_long_transfers(merged_feed)
    merged_feed, transfers_manual = add_manual_transfers(merged_feed)
    transfers_nearby = set()
    if context.config("gtfs_transfer_max_distance") > 0:
        merged_feed, transfers_nearby = gtfs.add_nearby_transfers(
            merged_feed, max_distance_m = context.config("gtfs_transfer_max_distance"),
            walk_speed_m_s = context.config("gtfs_transfer_walk_speed"),
            detour_factor = context.config("gtfs_transfer_detour_factor"),
        )
    merged_feed = gtfs.fill_calendar_from_dates(merged_feed)
    merged_feed = gtfs.drop_unused_shapes(merged_feed)

    # Not used by pt2matsim: what data.gtfs.visualize shows about the merge (the source feed of the routes and stops,
    # and where each transfer comes from)
    _write_diagnostics(context.path(), merged_feed, sources, ids_by_feed, transfers_feeds, transfers_bridged, transfers_manual, transfers_nearby)

    # Fix for pt2matsim (will be fixed after PR #173)
    # Order of week days must be fixed

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    columns = [c for c in merged_feed["calendar"].columns if c not in days] + days
    merged_feed["calendar"] = merged_feed["calendar"][columns]

    # Write feed (not as a ZIP, but as files, for pt2matsim)
    gtfs.write_feed(merged_feed, "%s/output" % context.path())

    return "gtfs"

# Transfer times set by hand, from all the stops (platforms) of a station to all the ones of another, by their base code
# (the stop id before ":"): (from code, to code, min_transfer_time in s, why). They replace the rules of the feeds, and
# gtfs.add_nearby_transfers does not change them. E.g. ("8501120", "8592052", 360, "Lausanne -> Lausanne, Grancy (m2)").
# The neighbouring stops without a rule get a time from add_nearby_transfers (Lausanne -> Grancy: the 300 s of the station
# to the m2 at Lausanne, gare, plus the walk to Grancy).
MANUAL_TRANSFERS = []


def add_manual_transfers(feed):
    """Adds the MANUAL_TRANSFERS (transfer_type 2), replacing the rules that already link the same stops."""
    if not MANUAL_TRANSFERS:
        return feed, set()

    stops = feed["stops"]
    stops = stops[pd.to_numeric(stops["location_type"], errors = "coerce").fillna(0) == 0] if "location_type" in stops.columns else stops
    stop_ids = stops["stop_id"].astype(str)
    base = stop_ids.str.split(":").str[0]

    rows = []
    for from_code, to_code, seconds, why in MANUAL_TRANSFERS:
        origins, destinations = stop_ids[base == from_code].tolist(), stop_ids[base == to_code].tolist()
        if not origins or not destinations:
            logger.warning(f"Manual transfer {why}: no stop with the code {from_code if not origins else to_code}, not added")
            continue
        rows += [{"from_stop_id": a, "to_stop_id": b, "transfer_type": 2, "min_transfer_time": seconds} for a in origins for b in destinations]
        logger.info(f"Manual transfer {why}: {seconds} s between {len(origins)} x {len(destinations)} stops")

    if not rows:
        return feed, set()

    manual = pd.DataFrame(rows)
    pairs = set(zip(manual["from_stop_id"], manual["to_stop_id"]))
    transfers = feed["transfers"] if "transfers" in feed else pd.DataFrame(columns = manual.columns)
    existing = list(zip(transfers["from_stop_id"].astype(str), transfers["to_stop_id"].astype(str)))
    replaced = sum(pair in pairs for pair in existing)
    if replaced:
        logger.info(f"  {replaced} existing rule(s) replaced")

    feed = dict(feed)
    feed["transfers"] = pd.concat([transfers[[pair not in pairs for pair in existing]], manual], ignore_index = True)
    return feed, pairs


def _transfer_pairs(feed):
    if "transfers" not in feed:
        return set()
    return set(zip(feed["transfers"]["from_stop_id"].astype(str), feed["transfers"]["to_stop_id"].astype(str)))


def _source_of_ids(final_ids, ids_by_feed, sources):
    """{final id: source file}. merge_feeds() appends _m<k> to the ids of the k-th feed (from 1) that an earlier feed
    already has."""
    result = {}
    for k, ids in enumerate(ids_by_feed):
        for id_ in ids:
            result.setdefault(id_, sources[k])
            result.setdefault(f"{id_}_m{k + 1}", sources[k])
    return {id_: result.get(id_, "") for id_ in final_ids}


def _write_diagnostics(path, feed, sources, ids_by_feed, transfers_feeds, transfers_bridged, transfers_manual = frozenset(),
                       transfers_nearby = frozenset()):
    folder = os.path.join(path, "diagnostics")
    os.makedirs(folder, exist_ok = True)

    for slot, column in (("routes", "route_id"), ("stops", "stop_id")):
        final_ids = feed[slot][column].astype(str).unique()
        source = _source_of_ids(final_ids, ids_by_feed[slot], sources)
        pd.DataFrame({column: list(source), "source": list(source.values())}).to_csv(
            os.path.join(folder, f"{slot}_sources.csv"), index = False)

    if transfers_bridged is not None:
        kept = _transfer_pairs(feed)
        columns = ["from_stop_id", "to_stop_id", "min_transfer_time"]
        transfers = transfers_bridged[columns].astype({"from_stop_id": str, "to_stop_id": str})
        added = set(transfers_manual) | set(transfers_nearby)
        if added:
            final = feed["transfers"][columns].astype({"from_stop_id": str, "to_stop_id": str})
            final = final[[pair in added for pair in zip(final["from_stop_id"], final["to_stop_id"])]]
            transfers = pd.concat([transfers, final]).drop_duplicates(["from_stop_id", "to_stop_id"], keep = "last")
        pairs = list(zip(transfers["from_stop_id"], transfers["to_stop_id"]))
        transfers = transfers.assign(
            rule = ["manual" if pair in transfers_manual else "nearby" if pair in transfers_nearby
                    else "feed" if pair in transfers_feeds else "bridge" for pair in pairs],
            kept = [pair in kept for pair in pairs],
        )
        transfers.to_csv(os.path.join(folder, "transfers.csv"), index = False)


def get_input_files(base_path):
    gtfs_paths = [
        str(child)
        for child in pathlib.Path(base_path).glob("*")
        if child.suffix.lower() == ".zip"
    ]

    if len(gtfs_paths) == 0:
        raise RuntimeError("Did not find any GTFS data (.zip) in {}".format(base_path))

    return gtfs_paths

def validate(context):
    input_files = get_input_files(os.path.join(context.config("data_path"), context.config("gtfs_folder")))
    total_size = 0
    for path in input_files:
        total_size += os.path.getsize(path)
    return total_size
