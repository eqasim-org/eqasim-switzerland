"""Build a small offline geographic backdrop from existing spatial inputs."""

import json
from pathlib import Path


def prepare_background(settings, work):
    root = Path(settings.get("data_path", ".")) / "spatial"
    sources = {}
    for kind, folder in (("country", "country"), ("cantons", "canton"), ("lakes", "lakes")):
        files = sorted((root / folder).glob("*.shp"))
        if files:
            sources[kind] = files[0]
    target = work / "background.json"
    manifest = work / "background-manifest.json"
    identity = {"schema": 1, "files": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for source in sources.values() for p in sorted(source.parent.glob(source.stem + ".*"))]}
    try:
        if target.exists() and json.loads(manifest.read_text()) == identity:
            return
    except (OSError, ValueError):
        pass
    layers = {}
    if sources:
        import geopandas as gpd
        import shapely
        for kind, source in sources.items():
            frame = gpd.read_file(source)
            frame.geometry = frame.geometry.map(shapely.force_2d)
            frame = frame.to_crs(3857)
            frame.geometry = frame.geometry.simplify(160, preserve_topology=True)
            frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty, ["geometry"]]
            layers[kind] = json.loads(frame.to_json(drop_id=True))
    target.write_text(json.dumps({"layers": layers, "attribution": "swisstopo / Swiss Federal Statistical Office · local spatial inputs"}, separators=(",", ":")))
    manifest.write_text(json.dumps(identity, indent=2))
