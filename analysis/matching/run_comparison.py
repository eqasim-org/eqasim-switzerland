"""Run the comparison using an existing Synpp configuration, optionally in debug mode."""

import argparse
import logging
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument("--debug", action="store_true", help="Reuse cached boundaries without normal validation")
    parser.add_argument("--reuse-matches", action="store_true", help="In debug mode, execute only the report stage")
    args = parser.parse_args()
    if args.reuse_matches and not args.debug:
        parser.error("--reuse-matches requires --debug; use the saved-table renderer for offline report rebuilding")
    # Small neural batches perform better with limited numerical-library threads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", str(len(os.sched_getaffinity(0))))
    from synpp.pipeline import Synpp
    logging.basicConfig(level=logging.INFO)
    stages = ["analysis.matching.compare_algorithms"]
    if not args.reuse_matches:
        stages = ["synthesis.population.matching.V2.model"] + [
            f"synthesis.population.matching.matched_v{i}" for i in (1, 2, 3)] + stages
    pipeline = Synpp.build_from_yml(args.config, run=stages)
    pipeline.debug = args.debug
    pipeline.dryrun = False
    pipeline.flowchart_path = None
    results = pipeline.run_pipeline()
    print(results[-1], flush=True)


if __name__ == "__main__":
    main()
