"""Source identity for V2 stage caches, without eager model imports."""

import hashlib
from pathlib import Path


def implementation_token():
    digest = hashlib.sha256()
    # Include the shared work/education predictor used by the input adapter.
    paths = [*Path(__file__).parent.glob("*.py"), Path(__file__).parent.parent / "commute.py"]
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
