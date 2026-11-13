"""Run only the three-method predictive validation, using configured input caches."""

import argparse
import logging
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument("--debug", action="store_true", help="Reuse upstream caches without rebuilding the population")
    parser.add_argument("--shared-model", action="store_true", help="Train one V2 model; run 50 survey and 10 population repetitions")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", str(len(os.sched_getaffinity(0))))
    from synpp.pipeline import Synpp
    logging.basicConfig(level=logging.INFO)
    stages = (["synthesis.population.matching.V2.model", "analysis.matching.repeat_experiments"]
              if args.shared_model else ["analysis.matching.validate_algorithms"])
    pipeline = Synpp.build_from_yml(args.config, run=stages)
    pipeline.debug = args.debug
    pipeline.dryrun = False
    pipeline.flowchart_path = None
    print(pipeline.run_pipeline()[-1], flush=True)


if __name__ == "__main__":
    main()
