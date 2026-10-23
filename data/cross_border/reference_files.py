"""Select a consistent local crossing reference without changing survey inputs."""

from pathlib import Path


REFERENCE_FILES = (
    "border_interview_places_road_and_rail.gpkg",
    "crossings_1.gpkg",
    "road_exit_points_official.gpkg",
    "road_exit_points_reviewed.json",
)


def reference_directory(data_path, reference_path):
    """Default to the legacy directory; resolve relative selections under data_path."""
    if reference_path is None:
        return Path(data_path) / "crossborder"
    root = Path(reference_path).expanduser()
    if not root.is_absolute():
        root = Path(data_path) / root

    # An explicitly selected release must be complete. Falling back to the
    # legacy files would silently discard reviewed exits on another machine.
    missing = [name for name in REFERENCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Selected cross_border_reference_path %s is incomplete; missing: %s"
                                % (root, ", ".join(missing)))
    return root


def file_versions(paths):
    """Use JSON-stable cache tokens, including absence of optional legacy files."""
    versions = {}
    for path in paths:
        stat = path.stat() if path.exists() else None
        versions[str(path)] = [stat.st_mtime_ns, stat.st_size] if stat is not None else None
    return versions
