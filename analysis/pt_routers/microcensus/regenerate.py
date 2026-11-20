"""Rebuild the Microcensus PT HTML from exported files without running routers."""
import argparse
import json
from pathlib import Path
import polars as pl
from analysis.pt_routers.microcensus import report


def regenerate(output):
    output = Path(output)
    trips = pl.read_parquet(output / 'requests.parquet')
    routes = {name: pl.read_parquet(output / f'{name}_routes.parquet')
              for name in ['rust', 'raptor', 'flash', 'google']}
    timings = json.loads((output / 'timings.json').read_text())
    metadata = json.loads((output / 'metadata.json').read_text())
    return report.generate(trips, routes, timings, metadata, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Existing pt_microcensus output directory')
    print(regenerate(parser.parse_args().output))
