"""Zurich flight-board capture and sampling using the shared airport timing engine."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from data.airports import geneva_times as engine
from data.airports.common import atomic_write, file_digest

ENDPOINT = 'https://flightdata.flughafen-zuerich.ch/flights'
SNAPSHOT_RELATIVE_PATH = Path('airports/zurich/data_from_websites/zurich_flight_times.json')


def snapshot_file(snapshot_path=None, *, data_path=None):
    if snapshot_path is None:
        if data_path is None:
            raise ValueError('Provide data_path or snapshot_path for Zurich timing data')
        return Path(data_path) / SNAPSHOT_RELATIVE_PATH
    path = Path(snapshot_path)
    return Path(data_path) / path if data_path is not None and not path.is_absolute() else path


def get_time_distribution(day=None, *, direction=None, snapshot_path=None, data_path=None):
    path = snapshot_file(snapshot_path, data_path=data_path)
    return engine.get_time_distribution(day, direction=direction, snapshot_path=path)


def sample_departure_times(direction, size, *, seed, travel_time_seconds=None, day=None, weekday=None,
                           snapshot_path=None, data_path=None):
    path = snapshot_file(snapshot_path, data_path=data_path)
    return engine.sample_departure_times(direction, size, seed=seed, travel_time_seconds=travel_time_seconds,
                                         day=day, weekday=weekday, snapshot_path=path)


def refresh_snapshot(output):
    """Refresh an existing reviewed snapshot; preserve its chosen timing assumptions."""
    output = Path(output)
    snapshot = engine.load_snapshot(output)
    with urlopen(ENDPOINT, timeout=30) as stream:
        raw = json.load(stream)
    flights = []
    for row in raw:
        status = (row.get('statusTextEn') or '').strip()
        labels = ' '.join([status, row.get('airline', ''), row.get('model', '')]).lower()
        if not row.get('isCommercial') or any(word in labels for word in ['cargo', 'freight']):
            continue
        if row['flightType'] not in ('A', 'D'):
            raise ValueError('Unknown Zurich flight type')
        flights.append({'flight_id': row['id'], 'direction': 'to_airport' if row['flightType'] == 'D' else 'from_airport',
                        'scheduled_utc': row['STD'] if row['flightType'] == 'D' else row['STA'],
                        'status': status, 'weight': 1.})
    dates = sorted({datetime.fromisoformat(r['scheduled_utc'].replace('Z', '+00:00')).astimezone(ZoneInfo('Europe/Zurich')).date().isoformat() for r in flights})
    snapshot.update(flights=flights, start_date=dates[0], end_date=dates[-1],
                    retrieved_at=datetime.now(timezone.utc).isoformat(), raw_records=len(raw),
                    excluded_noncommercial_or_known_cargo=len(raw)-len(flights))
    # Coverage is recomputed from the refreshed rows, without hard-coded weekday names.
    frame = engine.flight_frame(snapshot)
    snapshot['covered_days'] = [engine.DAY_NAMES[d] for d in sorted(frame.weekday.unique())]
    raw_path = output.parent / 'flight_board_raw.json'
    atomic_write(raw_path, lambda p: p.write_text(json.dumps(raw, ensure_ascii=False) + '\n', encoding='utf-8'))
    atomic_write(output, lambda p: p.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8'))
    metadata_path = output.parent / 'zurich_airport.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        for source in metadata['sources']:
            if source['id'] == 'flight_board':
                source['sha256'] = file_digest(raw_path)
                source['retrieved_on'] = snapshot['retrieved_at'][:10]
        atomic_write(metadata_path, lambda p: p.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'))
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explicitly refresh the reviewed Zurich flight-board snapshot.')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = refresh_snapshot(args.output)
    print(f'Saved {len(result["flights"])} records; dates {result["start_date"]} to {result["end_date"]}')
