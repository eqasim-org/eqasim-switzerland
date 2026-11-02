"""Extend an existing PT comparison report without running any routing stages."""
from pathlib import Path
import argparse
import json

import polars as pl
import yaml
from synpp.pipeline import ConfigurationContext

from analysis.pt_routers import analyse, report, unmatched


def regenerate(config_path):
    settings = yaml.safe_load(Path(config_path).read_text())
    context = ConfigurationContext(settings['config'])
    analyse.configure(context)
    output = Path(context.config('output_path')) / context.config('pt_router_analysis_output_folder')
    summary = json.loads((output / 'summary.json').read_text())
    summary['flash_only_diagnostics'] = unmatched.analyse(
        output, pl.read_parquet(output / 'requests.parquet'), context.config('random_seed'),
        output / 'java_itineraries.csv', sorted(output.glob('flash_selected_legs_batch_*.parquet')),
        context.config('flash_tb_walk_speed_m_per_s'), context.config('flash_tb_walk_detour_factor'))
    report.write_index(output, summary, summary.get('warnings', []))
    return output / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Configuration of the completed PT comparison')
    print(regenerate(parser.parse_args().config))
