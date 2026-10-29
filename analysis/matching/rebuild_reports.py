"""Rebuild matching reports from completed validation files, with no upstream stages.

Convenient direct use (does not invoke Synpp or inspect training code)::

    uv run -m analysis.matching.rebuild_reports config_dib_10.yml
"""

import argparse
import hashlib
import html
import json
import logging
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from . import compare_matching, matching_embedding, explanations, geometry, metrics, reporting, report_diagnostics
from synthesis.population.matching.V2.features import source_token

LOGGER = logging.getLogger(__name__)


def configure(context):
    context.config("analysis_path")
    context.config("matching_report_source")
    context.config("random_seed")
    context.config("matching_validation_bootstrap", 500)
    context.config("matching_embedding_plot_method", "tsne")
    context.config("matching_embedding_plot_sample_size", 1500)
    # Deliberately no context.stage(): a report rebuild must never schedule validation.


def execute(context):

    # Load only explicitly selected completed results; no pipeline dependencies are registered.

    source = Path(context.config("matching_report_source"))
    output = Path(context.config("analysis_path")) / "matching" / "rebuilt"

    # Regenerate figures and explanations while preserving all upstream cache files.

    return build_reports(source, output, seed=int(context.config("random_seed")),
                         bootstrap=int(context.config("matching_validation_bootstrap")),
                         method=str(context.config("matching_embedding_plot_method")).lower(),
                         maximum=int(context.config("matching_embedding_plot_sample_size")))


def validate(context):
    source = Path(context.config("matching_report_source"))
    manifest = check_source(source)
    evidence = [(str(path.relative_to(source)), path.stat().st_size, path.stat().st_mtime_ns)
                for path in _evidence_files(source, manifest)]
    return hashlib.sha256((json.dumps(evidence) + source_token(
        compare_matching, matching_embedding, explanations, geometry, metrics, reporting,
        report_diagnostics)).encode()).hexdigest()


def check_source(source):
    source = Path(source)
    if not (source / "manifest.json").is_file():
        raise ValueError(f"No completed validation manifest in {source}. Select the validation .cache directory.")
    manifest = json.loads((source / "manifest.json").read_text())
    missing = [str(path.relative_to(source)) for path in _evidence_files(source, manifest) if not path.is_file()]
    if missing:
        raise ValueError("Incomplete saved validation results; missing: " + ", ".join(missing[:12]) +
                         ". Report rebuilding will not start training to fill gaps.")
    return manifest


def _evidence_files(source, manifest):
    paths = [source / name for name in ("manifest.json", "out_of_fold_scores.csv", "respondents.csv",
             "grouped_permutation.csv", "feature_inventory.csv", "population_support.csv", "covariate_balance.csv")]
    for repeat in range(manifest["model_repeats"]):
        for fold in range(manifest["outer_folds"]):
            directory = source / f"repeat_{repeat}_fold_{fold}"
            paths.extend(directory / name for name in ("donors_embeddings.csv", "heldout_embeddings.csv",
                "donors_outcomes.npy", "heldout_outcomes.npy", "learned_neighbors.npy", "raw_neighbors.npy"))
    return paths


def build_reports(source, output, *, seed=None, bootstrap=500, method="tsne", maximum=1500):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    manifest = check_source(source)
    if source == output or source in output.parents:
        raise ValueError("The report output must be outside the read-only validation cache.")
    if method not in ("tsne", "pca") or maximum < 5 or bootstrap < 1:
        raise ValueError("Use method tsne/pca, a sample of at least 5, and a positive bootstrap count.")
    seed = int(manifest["seed"] if seed is None else seed)
    start = time.monotonic()
    # Limit report numerics only; never modify model settings or any saved model artifact.
    with threadpool_limits(limits=1):
        LOGGER.info("Rebuilding comparison from saved predictions (no training or matching)")
        comparison = compare_matching.generate_report(source, output / "comparison", seed, bootstrap)
        LOGGER.info("Rebuilding embedding figures from saved representations")
        embedding = matching_embedding.generate_report(source, output / "embedding", seed, bootstrap, method, maximum)
    settings = dict(source=str(source), validation_manifest_sha256=hashlib.sha256(
        (source / "manifest.json").read_bytes()).hexdigest(), seed=seed, bootstrap=bootstrap,
        projection=method, projection_sample_size=maximum, numerical_threads=1,
        elapsed_seconds=time.monotonic()-start, refitted_models=0, rematched_persons=0,
        code_sha256=source_token(compare_matching, matching_embedding, explanations, geometry,
                                 metrics, reporting, report_diagnostics))
    for directory in (output, output / "comparison", output / "embedding"):
        (directory / "report_settings.json").write_text(json.dumps(settings, indent=2))
    index = output / "report.html"
    index.write_text("<!doctype html><meta charset='utf-8'><title>Matching reports</title>"
        "<style>body{font:17px system-ui;max-width:850px;margin:3rem;line-height:1.6}</style>"
        "<h1>Matching: reading the saved experiment</h1>"
        f"<p>These reports reuse the completed test on {manifest['respondents']:,} survey respondents, "
        f"with {manifest['outer_folds']} household folds, {manifest['model_repeats']} model repeats and "
        f"{manifest['draws']} sampled donor days per prediction. No model was retrained and nobody was rematched.</p>"
        "<p><a href='comparison/report.html'>V1 versus V2: hidden-day prediction, errors and calibration</a></p>"
        "<p><a href='embedding/report.html'>Learned representation: neighbourhoods, nonlinear views and input contributions</a></p>"
        "<p>The original donor draws and exact chain labels were not archived by this validation version. "
        "The reports use their saved prediction scores, means and interval diagnostics. They do not claim to "
        "reconstruct exact sampled diary distributions. Prediction means are averages over possible days, "
        "not individual sampled days.</p>"
        "<p>Survey-to-survey performance is evidence about hidden survey respondents. Transferring the conclusion "
        "to the synthetic population also requires adequate covariate overlap and comparable behaviour conditional "
        "on those covariates.</p><p>Read-only evidence source: <code>" + html.escape(str(source)) + "</code></p>")
    LOGGER.info("Reports rebuilt in %.1f seconds: %s", settings["elapsed_seconds"], index)
    return str(index)


def find_source(working_directory):
    candidates = [path for path in Path(working_directory).expanduser().glob("analysis.matching.validation__*.cache")
                  if (path / "manifest.json").is_file() and (path / "out_of_fold_scores.csv").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Found {len(candidates)} completed validation caches. Select one with --source "
                         "or config.matching_report_source; no cache is chosen by modification time.")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="Existing pipeline YAML (read only; its run list is ignored)")
    parser.add_argument("--source", type=Path, help="Completed analysis.matching.validation .cache directory")
    parser.add_argument("--output", type=Path, help="New report directory (default: analysis_path/matching/rebuilt)")
    parser.add_argument("--method", choices=("tsne", "pca"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--bootstrap", type=int)
    args = parser.parse_args()
    settings, base = {}, Path.cwd()
    if args.config:
        import yaml
        config_path = Path(args.config).resolve()
        settings = yaml.safe_load(config_path.read_text())
        base = config_path.parent
    config = settings.get("config", {})

    def absolute(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else base / path

    source = args.source or config.get("matching_report_source")
    if source is None:
        if "working_directory" not in settings:
            parser.error("Supply a pipeline config or --source with --output.")
        source = find_source(absolute(settings["working_directory"]))
    output = args.output
    if output is None:
        if "analysis_path" not in config:
            parser.error("Supply --output or config.analysis_path.")
        output = absolute(config["analysis_path"]) / "matching" / "rebuilt"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(build_reports(absolute(source), absolute(output), seed=config.get("random_seed"),
        bootstrap=args.bootstrap if args.bootstrap is not None else int(config.get("matching_validation_bootstrap", 500)),
        method=args.method or str(config.get("matching_embedding_plot_method", "tsne")).lower(),
        maximum=args.sample_size if args.sample_size is not None else int(config.get("matching_embedding_plot_sample_size", 1500))))


if __name__ == "__main__":
    main()
