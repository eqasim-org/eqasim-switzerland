"""Invalidate fitted-stage caches when their shared implementation changes."""
import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SOURCES = ('constants.py', 'parameters.py', 'models/data.py', 'models/declarations.py',
            'models/specification.py', 'models/utility.py', 'models/validation.py',
            'vot/constraints.py', 'vot/functions.py')


def validate(context):
    """Synpp already hashes each estimator module; also track its shared helpers."""
    result = {}
    for name in _SOURCES:
        with (_ROOT / name).open('rb') as stream:
            result[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result
