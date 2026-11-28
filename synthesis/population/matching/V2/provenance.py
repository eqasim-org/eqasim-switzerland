"""Source identity for V2 stage caches, without eager model imports."""

import hashlib
from pathlib import Path


def implementation_token():
    digest = hashlib.sha256()
    # Imported V1 preparation and the shared residence mask affect inputs too.
    matching = Path(__file__).parent.parent
    paths = [
        *Path(__file__).parent.glob("*.py"),
        matching / "commute.py",
        matching / "opening_hours.py",
        matching / "matched_v1.py",
        matching / "workflow.py",
        matching / "spatial_feature.py",
        matching.parent / "models" / "non_private.py",
    ]
    # Imported density settings/helpers affect preparation outside V2's package.
    from data.statent import density
    from dmc import constants

    paths.extend([Path(density.__file__), Path(constants.__file__)])
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
