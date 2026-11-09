"""Source identity for V2 stage caches, without eager model imports."""

import hashlib
from pathlib import Path


def implementation_token():
    digest = hashlib.sha256()
    # Imported V1 preparation and the shared residence mask affect inputs too.
    matching = Path(__file__).parent.parent
    paths = [*Path(__file__).parent.glob("*.py"), matching / "commute.py", matching / "opening_hours.py",
             matching / "matched_v1.py", matching.parent / "models" / "non_private.py"]
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
