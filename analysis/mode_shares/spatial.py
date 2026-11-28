"""Compact spatial trip exports and a standalone, client-side comparison report."""

import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from pyproj import Transformer
import requests
import shapely

MODES = ['car', 'pt', 'walk', 'bike', 'car_passenger']
MISSING = np.iinfo(np.int32).min
ASSETS = {
    'leaflet.js': 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
    'leaflet.css': 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
    'leaflet.draw.js': 'https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js',
    'leaflet.draw.css': 'https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css',
}
TILES = 'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg'


def pack_trips(frame):
    """Encode columnar arrays, retaining every trip and marking missing locations.

    Geographic coordinates use integer microdegrees (roughly 0.1 m). Person IDs
    are replaced by dataset-local integer codes for distinct-person counts.
    Byte offsets are aligned for direct browser TypedArray views.
    """
    weights = frame.weight.to_numpy(dtype=float)
    if not (np.isfinite(weights) & (weights > 0)).all():
        raise ValueError('Interactive mode shares require finite positive trip weights')
    mode = frame['mode'].map({name: i for i, name in enumerate(MODES)})
    if mode.isna().any() or frame.person_id.isna().any():
        raise ValueError('Spatial trips require known modes and person IDs')
    people, unique = pd.factorize(frame.person_id, sort=False)
    blocks, columns, offset = [], {}, 0
    def append(name, values, dtype):
        nonlocal offset
        values = np.asarray(values, dtype=dtype)
        padding = (-offset) % values.dtype.itemsize
        blocks.append(bytes(padding)); offset += padding
        columns[name] = dict(offset=offset, dtype=values.dtype.str)
        blocks.append(values.tobytes()); offset += values.nbytes
    append('mode', mode, 'u1')
    append('person', people, '<u4')
    append('weight', weights, '<f8')
    transform = Transformer.from_crs(2056, 4326, always_xy=True)
    missing = {}
    for anchor in ('origin', 'destination', 'home'):
        xy = frame[[anchor+'_x', anchor+'_y']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1)
        longitude, latitude = transform.transform(xy[:,0], xy[:,1])
        valid &= np.isfinite(longitude) & np.isfinite(latitude)
        valid &= (np.abs(longitude) <= 180) & (np.abs(latitude) <= 90)
        missing[anchor] = int((~valid).sum())
        for suffix, values in (('lon', longitude), ('lat', latitude)):
            encoded = np.full(len(frame), MISSING, dtype='<i4')
            encoded[valid] = np.rint(values[valid]*1e6).astype(np.int32)
            append(anchor+'_'+suffix, encoded, '<i4')
    metadata = dict(rows=len(frame), people=len(unique), columns=columns, missing_locations=missing,
                    raw_counts=np.bincount(mode, minlength=len(MODES)).tolist(),
                    weighted_counts=np.bincount(mode, weights=weights, minlength=len(MODES)).tolist())
    return gzip.compress(b''.join(blocks), compresslevel=6, mtime=0), metadata


def write_report(output, target, simulated, municipalities, cantons=None, *, dataset_label="MATSim"):
    """Generate one HTML with embedded trips, boundaries and pinned UI libraries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(modes=MODES, datasets={}, municipalities=None, cantons=None, tiles=TILES)
    summaries = []
    for name, frame in [('Microcensus', target), (dataset_label, simulated)]:
        compressed, metadata = pack_trips(frame)
        payload['datasets'][name] = dict(**metadata, data=base64.b64encode(compressed).decode())
        (output / (name.lower().replace(' ', '_')+'_spatial_trips.bin.gz')).write_bytes(compressed)
        total = sum(metadata['weighted_counts'])
        for i, mode in enumerate(MODES):
            summaries.append(dict(dataset=name, mode=mode, observations=metadata['raw_counts'][i],
                weighted_trips=metadata['weighted_counts'][i], share=metadata['weighted_counts'][i]/total if total else None))
    pd.DataFrame(summaries).to_csv(output/'spatial_mode_shares_national.csv', index=False)

    # Keep exact municipal boundaries; simplify only the displayed outlines in JS.

    geography = municipalities[['municipality_id','municipality_name','geometry']].copy()
    geography = geography.loc[geography.geometry.notna() & ~geography.geometry.is_empty]
    geography.geometry = shapely.force_2d(geography.geometry.array)
    geography = geography.to_crs(4326).sort_values('municipality_id')
    geography['municipality_id'] = geography.municipality_id.astype(int)
    encoded = gzip.compress(geography.to_json(drop_id=True).encode(), mtime=0)
    payload['municipalities'] = base64.b64encode(encoded).decode()

    # Canton choices are Swiss administrative polygons, not residence attributes.

    canton_count = 0
    canton_json = '{"type":"FeatureCollection","features":[]}'
    if cantons is not None:
        regions = cantons.loc[cantons.canton_id.between(1, 26),
                              ['canton_id', 'canton_name', 'geometry']].copy()
        regions = regions.loc[regions.geometry.notna() & ~regions.geometry.is_empty]
        regions.geometry = shapely.force_2d(regions.geometry.array)
        regions['canton_id'] = regions.canton_id.astype(int)
        regions['canton_name'] = regions.canton_name.map(_display_name)
        canton_json = regions.to_crs(4326).sort_values('canton_id').to_json(drop_id=True)
        canton_count = len(regions)
    payload['cantons'] = base64.b64encode(gzip.compress(canton_json.encode(), mtime=0)).decode()

    # Bundle assets beside and inside the report; data selection works via file://.

    assets = _assets(output/'assets')
    template = Path(__file__).with_name('spatial.html.template').read_text().replace('MATSim', dataset_label)
    script = Path(__file__).with_name('spatial.js').read_text().replace('MATSim', dataset_label)
    worker = Path(__file__).with_name('spatial_worker.js').read_text().replace('MATSim', dataset_label)
    for key, text in [('__LEAFLET_CSS__',assets['leaflet.css']), ('__DRAW_CSS__',assets['leaflet.draw.css']),
                      ('__LEAFLET_JS__',assets['leaflet.js']), ('__DRAW_JS__',assets['leaflet.draw.js']),
                      ('__WORKER__',worker), ('__APP__',script),
                      ('__PAYLOAD__',json.dumps(payload, separators=(',',':')).replace('<', '\\u003c'))]:
        template = template.replace(key, text)
    destination = output/'spatial_mode_shares.html'
    destination.write_text(template)
    manifest = {name:{key:value for key,value in dataset.items() if key!='data'}
                for name,dataset in payload['datasets'].items()}
    (output/'spatial_mode_shares_manifest.json').write_text(json.dumps(dict(datasets=manifest,
        coordinate_crs='EPSG:4326', coordinate_precision='Integer microdegrees; about 0.1 m',
        selection=f'Same domestic-resident trip cohort as global mode shares; survey person_weight per trip, {dataset_label} weight 1.',
        airport_filter=f'{dataset_label} trips with airport origin OR destination excluded before all analyzer summaries.',
        query_modes=['arrivals','departures','either','both','residents'], municipalities=len(geography),
        cantons=canton_count), indent=2))
    return str(destination)


def _display_name(value):
    """Repair legacy UTF-8 names read as Latin-1 without changing source caches."""
    text = str(value)
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _assets(directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, url in ASSETS.items():
        path = directory/name
        if not path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            text = response.text
            if name.endswith('.css'):
                def inline(match):
                    source = match.group(1).strip(' \"\'')
                    if source.startswith('data:'):
                        return match.group(0)
                    image = requests.get(urljoin(url, source), timeout=60)
                    image.raise_for_status()
                    mime = image.headers.get('Content-Type','image/png').split(';')[0]
                    return 'url(data:'+mime+';base64,'+base64.b64encode(image.content).decode()+')'
                text = re.sub(r'url\(([^)]+)\)', inline, text)
            path.write_text(text)
        result[name] = path.read_text()
    return result


def implementation_token():
    paths = [Path(__file__), Path(__file__).with_name('spatial.js'),
             Path(__file__).with_name('spatial_worker.js'), Path(__file__).with_name('spatial.html.template')]
    return hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()
