"""Track moved stage helpers without invalidating unrelated cached stages."""

import hashlib
from importlib.util import find_spec
from pathlib import Path


def source_digest(module):
    return hashlib.sha256(Path(find_spec(module).origin).read_bytes()).hexdigest()
