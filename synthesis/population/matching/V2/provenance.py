"""Source identity for V2 stage caches, without eager model imports."""

import hashlib
from pathlib import Path


def implementation_token():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
