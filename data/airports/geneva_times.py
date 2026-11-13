"""Offline flight-time proxies and reproducible ground-trip departure sampling."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SNAPSHOT_RELATIVE_PATH = Path('airports/geneva/data_from_websites/geneva_flight_times.json')
DAY_NAMES = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
SOURCE_URL = 'https://www.gva.ch/fr/passagers/vols-et-destinations'
ENDPOINT = 'https://edge-graph.adobe.io/api/85d1a5c5-4d32-4b8b-9471-a94b4425466e/graphql'
QUERY = '''query Flights($dayOffset:Int=0){
GetDepartures(dayOffset:$dayOffset){... on Flight{
id departure{scheduled} flightNumber{airlineCode trackNumber} flightStatus}}
GetArrivals(dayOffset:$dayOffset){... on Flight3{
id arrival{scheduled} flightNumber{airlineCode trackNumber} flightStatus}}}'''


def refresh_snapshot(output):
    """Explicitly capture the public board's complete today-to-six-days window."""
    now = datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo('Europe/Zurich')).date()
    flights = []
    for offset in range(7):
        request = Request(ENDPOINT, data=json.dumps({'query': QUERY, 'variables': {'dayOffset': offset}}).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        if payload.get('errors'):
            raise ValueError(f'Flight-board errors: {payload["errors"]}')
        for key, direction, event in [('GetDepartures', 'to_airport', 'departure'),
                                      ('GetArrivals', 'from_airport', 'arrival')]:
            rows = payload['data'][key]
            if not rows or any(not row.get('id') or not row.get(event, {}).get('scheduled') for row in rows):
                raise ValueError(f'Incomplete board for offset {offset}, {direction}')
            for row in rows:
                flights.append({'flight_id': row['id'], 'direction': direction,
                                'scheduled_utc': row[event]['scheduled'], 'status': row['flightStatus'],
                                'weight': 1.0})
    if datetime.now(timezone.utc).astimezone(ZoneInfo('Europe/Zurich')).date() != today:
        raise ValueError('Local date changed during capture; repeat snapshot')
    snapshot = {'schema_version': 1, 'retrieved_at': now.isoformat(), 'timezone': 'Europe/Zurich',
                'start_date': str(today), 'end_date': str(today + timedelta(days=6)),
                'source_url': SOURCE_URL, 'endpoint': ENDPOINT, 'query': QUERY,
                'weighting': 'One per operating flight; codeshares are nested on the board and are not counted separately. No passenger loads available.',
                'excluded_statuses': ['cancelled', 'canceled', 'deleted', 'diverted', 'rerouted'],
                'checkin_lead_minutes': [90, 120, 180], 'arrival_clearance_minutes': [15, 30, 60],
                'terminal_opening_hour': 4,
                'assumptions': ['Seven-day November 2026 schedule is a seasonal proxy, not an annual or 2025 passenger-time observation.',
                                'Uniform flight weights; flight size, occupancy and connection shares by time are unknown.',
                                'Check-in lead triangular(90,120,180) minutes is a model assumption informed by airline advice, not measured behaviour.',
                                'Arrival clearance triangular(15,30,60) minutes is an uncalibrated model assumption.',
                                'Airport arrival targets before 04:00 are clipped to terminal opening; ground departures can be on the previous day.',
                                'Regions and resident/visitor profiles share the same time profile; no joint observations exist.'],
                'supporting_sources': [
                    {'url': 'https://www.swiss.com/fr/fr/fly/check-in/previous-day-check-in.html', 'use': 'Advice: about two hours before short-haul and three before long-haul; does not identify a statistical distribution.'},
                    {'url': 'https://www.gva.ch/en/passengers/travel-preparation/faq', 'use': 'Terminal reopens at 04:00.'}],
                'flights': flights}
    # Validate before replacing a previously usable snapshot.
    flight_frame(snapshot)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    return snapshot


def snapshot_file(path=None, *, data_path=None):
    if path is None:
        if data_path is None:
            raise ValueError('Provide data_path or snapshot_path for the Geneva timing data')
        return Path(data_path) / SNAPSHOT_RELATIVE_PATH
    path = Path(path)
    return Path(data_path) / path if data_path is not None and not path.is_absolute() else path


def load_snapshot(path=None, *, data_path=None):
    snapshot = json.loads(snapshot_file(path, data_path=data_path).read_text(encoding='utf-8'))
    flight_frame(snapshot)
    return snapshot


def flight_frame(snapshot):
    if snapshot.get('schema_version') != 1 or snapshot.get('timezone') != 'Europe/Zurich':
        raise ValueError('Unsupported flight snapshot schema or timezone')
    for field in ['checkin_lead_minutes', 'arrival_clearance_minutes']:
        triple = np.asarray(snapshot[field], dtype=float)
        if triple.shape != (3,) or not np.isfinite(triple).all() or not (0 <= triple[0] <= triple[1] <= triple[2]) or triple[0] == triple[2]:
            raise ValueError(f'Invalid triangular parameters: {field}')
    opening = snapshot['terminal_opening_hour']
    if isinstance(opening, bool) or not isinstance(opening, (float, int)) or not 0 <= opening < 24:
        raise ValueError('Invalid terminal opening hour')
    frame = pd.DataFrame(snapshot['flights'])
    required = {'direction', 'flight_id', 'scheduled_utc', 'status', 'weight'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Flight snapshot requires complete direction, flight_id, scheduled_utc, status and weight fields')
    if frame.empty or not set(frame.direction).issubset({'to_airport', 'from_airport'}):
        raise ValueError('Missing or unknown flight directions')
    if frame.duplicated(['direction', 'flight_id']).any():
        raise ValueError('Duplicate operating flights in snapshot')
    frame = frame.loc[~frame.status.str.strip().str.lower().isin(snapshot['excluded_statuses'])].copy()
    scheduled = pd.to_datetime(frame.scheduled_utc, utc=True, format="ISO8601").dt.tz_convert(snapshot['timezone'])
    frame['date'] = scheduled.dt.strftime('%Y-%m-%d')
    frame['weekday'] = scheduled.dt.weekday
    frame['flight_seconds'] = scheduled.dt.hour * 3600 + scheduled.dt.minute * 60 + scheduled.dt.second
    frame['weight'] = pd.to_numeric(frame.weight)
    if not np.isfinite(frame.weight).all() or (frame.weight <= 0).any():
        raise ValueError('Flight weights must be finite and positive')
    dates = pd.date_range(snapshot['start_date'], snapshot['end_date']).strftime('%Y-%m-%d')
    if not frame.date.isin(dates).all():
        raise ValueError('Flight outside declared snapshot dates')
    if any(not set(frame.loc[frame.direction.eq(d)].date).issuperset(dates) for d in ['to_airport', 'from_airport']):
        raise ValueError('Each direction needs flights on every snapshot date')
    return frame.reset_index(drop=True)


def sample_departure_times(direction, size, *, seed, travel_time_seconds=None, day=None, weekday=None,
                           snapshot_path=None, data_path=None):
    """Sample flight times, then infer ground departures in local wall-clock seconds.

    day accepts named days, weekday, weekend, all, or Monday=0..Sunday=6.
    weekday is a compatibility alias for day. Selected dates get equal weight. to_airport
    requires scalar or per-trip travel times. Negative or >86400 seconds retain
    previous/next-day departures; the accompanying day_offset makes this explicit.
    """
    if direction not in ('to_airport', 'from_airport'):
        raise ValueError('direction must be to_airport or from_airport')
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError('size must be a non-negative integer')
    if day is not None and weekday is not None:
        raise ValueError('Specify day or weekday, not both')
    selected_day = day_label(day if day is not None else weekday)
    if direction == 'to_airport' and travel_time_seconds is None:
        raise ValueError('to_airport requires travel_time_seconds to infer departure from the origin')
    travel = np.broadcast_to(np.asarray(0 if travel_time_seconds is None else travel_time_seconds, dtype=float), (size,))
    if not np.isfinite(travel).all() or (travel < 0).any():
        raise ValueError('Travel times must be finite and non-negative')
    if seed is None:
        raise ValueError('Provide a reproducible seed, normally context.config("random_seed")')
    snapshot = load_snapshot(snapshot_path, data_path=data_path)
    frame = flight_frame(snapshot)
    frame = frame.loc[frame.direction.eq(direction)]
    frame, weights = select_days(frame, selected_day, allow_partial=snapshot.get("allow_partial_week", False))
    rng = np.random.default_rng(seed)
    selected = frame.iloc[rng.choice(len(frame), size=size, p=weights)]
    flight = selected.flight_seconds.to_numpy(float)
    field = 'checkin_lead_minutes' if direction == 'to_airport' else 'arrival_clearance_minutes'
    buffer = rng.triangular(*snapshot[field], size=size) * 60
    event = (np.maximum(flight - buffer, snapshot['terminal_opening_hour'] * 3600)
             if direction == 'to_airport' else flight + buffer)
    departure = event - travel if direction == 'to_airport' else event
    return pd.DataFrame({'direction': direction, 'source_flight_id': selected.flight_id.to_numpy(),
                         'source_date': selected.date.to_numpy(), 'day': selected_day, 'flight_seconds': flight,
                         'airport_event_seconds': event, 'departure_seconds': departure,
                         'departure_day_offset': np.floor(departure / 86400).astype(int),
                         'departure_time_of_day': departure % 86400})


def day_label(day):
    if day is None:
        return 'all'
    if isinstance(day, str):
        label = day.strip().lower()
        if label in (*DAY_NAMES, 'weekday', 'weekend', 'all'):
            return label
    elif not isinstance(day, (bool, np.bool_)) and isinstance(day, (int, np.integer)) and 0 <= day <= 6:
        return DAY_NAMES[day]
    raise ValueError('day must be monday..sunday, weekday, weekend, all, or Monday=0..Sunday=6')


def select_days(frame, day, *, allow_partial=False):
    """Average normalized date-specific distributions, giving each date equal mass."""
    label = day_label(day)
    selected = (range(5) if label == 'weekday' else (5, 6) if label == 'weekend'
                else range(7) if label == 'all' else (DAY_NAMES.index(label),))
    frame = frame.loc[frame.weekday.isin(selected)].copy()
    if frame.empty or (not allow_partial and not set(selected).issubset(frame.weekday)):
        raise ValueError(f'Incomplete timing data for {label}')
    weights = frame.weight / frame.groupby('date').weight.transform('sum') / frame.date.nunique()
    return frame, weights.to_numpy()


def timing_coverage(snapshot, day, direction):
    """Describe exactly which calendar dates support a requested timing curve."""
    label = day_label(day)
    frame = flight_frame(snapshot)
    selected, _ = select_days(frame.loc[frame.direction.eq(direction)], label,
                             allow_partial=snapshot.get('allow_partial_week', False))
    required = (set(range(5)) if label == 'weekday' else {5, 6} if label == 'weekend'
                else set(range(7)) if label == 'all' else {DAY_NAMES.index(label)})
    covered = set(selected.weekday)
    return {'day': label, 'covered_days': [DAY_NAMES[i] for i in sorted(covered)],
            'missing_days': [DAY_NAMES[i] for i in sorted(required - covered)],
            'complete_day_group': required.issubset(covered),
            'dates': sorted(selected.date.unique().tolist()), 'flight_count': len(selected),
            'weighting': 'equal_dates_then_operating_flights',
            'retrieved_at': snapshot.get('retrieved_at')}


def get_time_distribution(day=None, *, direction=None, snapshot_path=None, data_path=None):
    """Return a normalized named-day or average weekday/weekend distribution."""
    label = day_label(day)
    if direction is not None and direction not in ('to_airport', 'from_airport'):
        raise ValueError('direction must be to_airport or from_airport')
    table = build_time_distribution(load_snapshot(snapshot_path, data_path=data_path))
    keep = table.day.eq(label)
    if direction is not None:
        keep &= table.direction.eq(direction)
    if not keep.any():
        raise ValueError(f"No timing data for {label} in this snapshot")
    return table.loc[keep].reset_index(drop=True)


def triangular_cdf(x, triple):
    low, mode, high = np.asarray(triple, dtype=float) * 60
    x = np.asarray(x, dtype=float)
    if mode == low:
        middle = 1 - ((high - x) / (high - low)) ** 2
    elif mode == high:
        middle = ((x - low) / (high - low)) ** 2
    else:
        middle = np.where(x <= mode, (x-low)**2 / ((high-low)*(mode-low)),
                          1-(high-x)**2 / ((high-low)*(high-mode)))
    return np.where(x <= low, 0., np.where(x >= high, 1., middle))


def build_time_distribution(snapshot):
    """Exact 15-minute bins by named day and equal-date weekday/weekend averages.

    to_airport ground-event bins are airport arrival targets: origin travel times
    must still be subtracted. from_airport ground-event bins are ground departures.
    """
    frame = flight_frame(snapshot)
    edges = np.arange(0, 28 * 3600 + 1, 900)
    output = []
    for direction in ('to_airport', 'from_airport'):
        for weekday, day in [(-1, 'all'), (-2, 'weekday'), (-3, 'weekend'), *enumerate(DAY_NAMES)]:
            available = frame.loc[frame.direction.eq(direction)]
            partial = snapshot.get("allow_partial_week", False)
            if partial and weekday >= 0 and weekday not in set(available.weekday):
                continue
            flights, weights = select_days(available, day, allow_partial=partial)
            required = set(range(5) if day == "weekday" else (5, 6) if day == "weekend"
                           else range(7) if day == "all" else (weekday,))
            covered = set(flights.weekday)
            covered_names = "|".join(DAY_NAMES[i] for i in sorted(covered))
            times = flights.flight_seconds.to_numpy(float)
            scheduled = np.histogram(times, bins=edges, weights=weights)[0]
            if direction == 'to_airport':
                cdf = 1 - triangular_cdf(times[:, None] - edges[None, :], snapshot['checkin_lead_minutes'])
                cdf[:, edges <= snapshot['terminal_opening_hour'] * 3600] = 0
                event_name = 'airport_arrival_target'
            else:
                cdf = triangular_cdf(edges[None, :] - times[:, None], snapshot['arrival_clearance_minutes'])
                event_name = 'ground_departure'
            ground = weights @ np.diff(cdf, axis=1)
            if not np.isclose(ground.sum(), 1):
                raise ValueError('Timing support exceeds export interval; widen bins')
            for event, probabilities in [('scheduled_flight', scheduled), (event_name, ground)]:
                for start, probability in zip(edges[:-1], probabilities):
                    output.append({'direction': direction, 'weekday': weekday, 'day': day, 'event': event,
                                   'start_seconds': int(start), 'end_seconds': int(start+900),
                                   'probability': float(probability), 'flight_count': len(flights),
                                   'covered_days': covered_names, 'complete_day_group': required.issubset(covered),
                                   'weighting': 'equal_dates_then_operating_flights', 'timezone': snapshot['timezone']})
    return pd.DataFrame(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explicitly refresh the official seven-day Geneva flight-board snapshot.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    captured = refresh_snapshot(args.output)
    print(f'Saved {len(captured["flights"])} operating-flight rows to {args.output}')
