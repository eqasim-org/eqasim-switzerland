"""Run only the three-method predictive validation, using configured input caches."""

import argparse
import logging
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--debug", action="store_true", help="Reuse upstream caches without rebuilding the population")
    parser.add_argument("--shared-model", action="store_true", help="Train one V2 model; run the configured survey and population repetitions")
    parser.add_argument("--expanded", action="store_true", help="200 neighbours, temperature .1, 100 survey / 20 population draws; separate outputs")
    parser.add_argument("--report", action="store_true", help="Render the saved expanded experiment without matching")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", str(len(os.sched_getaffinity(0))))
    from synpp.pipeline import Synpp
    logging.basicConfig(level=logging.INFO)
    stages = (["synthesis.population.matching.V2.model", "analysis.matching.repeat_experiments"]
              if args.shared_model else ["analysis.matching.validate_algorithms"])
    if args.expanded:
        import tempfile
        import yaml
        from pathlib import Path
        values = yaml.safe_load(Path(args.config).read_text())
        values["config"].update(matching_embedding_top_k=200, matching_embedding_temperature=.1,
            matching_repeat_microcensus=100, matching_repeat_population=20,
            matching_repeat_output_name="repeated_k200_t01_mz100_pop20", matching_repeat_render=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", dir="logs", delete=False) as stream:
            yaml.safe_dump(values, stream)
            config_path = stream.name
        stages = (["analysis.matching.repeat_report"] if args.report else
                  ["synthesis.population.matching.V2.model", "analysis.matching.repeat_experiments"])
        try:
            pipeline = Synpp.build_from_yml(config_path, run=stages)
        finally:
            Path(config_path).unlink()
    else:
        pipeline = Synpp.build_from_yml(args.config, run=stages)
    pipeline.debug = args.debug
    pipeline.dryrun = False
    pipeline.flowchart_path = None
    print(pipeline.run_pipeline()[-1], flush=True)


if __name__ == "__main__":
    main()
