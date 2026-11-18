"""Run analysis-only loss-weight search or its downstream comparison."""
import argparse
import logging
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_dib_10.yml')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    from synpp.pipeline import Synpp
    logging.basicConfig(level=logging.INFO)
    name = 'optimized_report' if args.report else 'optimize_weights'
    pipeline = Synpp.build_from_yml(args.config, run=['analysis.matching.' + name])
    pipeline.debug = True
    pipeline.dryrun = False
    pipeline.flowchart_path = None
    print(pipeline.run_pipeline()[-1], flush=True)


if __name__ == '__main__':
    main()
