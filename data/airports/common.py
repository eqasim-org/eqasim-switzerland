"""Shared deterministic allocation and file I/O for airport input stages."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def normalized(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Cannot normalize invalid or empty survey weights")
    return values / values.sum()



def apportion(expected):
    """Deterministic largest-remainder rounding, preserving the rounded total."""
    expected = np.asarray(expected, dtype=float)
    if expected.ndim != 1 or not np.isfinite(expected).all() or (expected < 0).any():
        raise ValueError('Expected passenger counts must be a finite non-negative vector')
    counts = np.floor(expected).astype(np.int64)
    remainder = int(np.floor(expected.sum() + .5)) - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def export_signature(paths):
    """Detect missing/changed exports even when Synpp can reuse a stage result."""
    state = {}
    for name, path in sorted(paths.items()):
        try:
            state[name] = file_digest(path)
        except OSError:
            state[name] = 'missing_or_unreadable'
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()



def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
