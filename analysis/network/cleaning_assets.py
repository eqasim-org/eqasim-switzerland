"""Local Leaflet libraries; background tiles are fetched by the viewer server."""
from pathlib import Path

import requests


def build_assets(output, center, radius):
    # Preserve these arguments for existing configurations; no fixed snapshot.
    assets = Path(output) / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("leaflet.js", "leaflet.css"):
        target = assets / name
        if not target.exists():
            response = requests.get("https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/" + name, timeout=40)
            response.raise_for_status()
            target.write_bytes(response.content)
    return {"live_url": "tiles/osm/{z}/{x}/{y}.png", "snapshot": None,
            "attribution": "© OpenStreetMap contributors"}
