"""Launch analysis-only architecture experiment or dependent report."""
import argparse
import logging
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_dib_10.yml')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--double-counts',action='store_true')
    parser.add_argument('--retrieval',action='store_true')
    parser.add_argument('--counts-retrieval',action='store_true')
    args=parser.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key,'1')
    from synpp.pipeline import Synpp
    logging.basicConfig(level=logging.INFO)
    stage='ablation_report' if args.report else 'ablation_experiment'
    if args.double_counts:
        stage='ablation_double_report' if args.report else 'ablation_double_counts'
    if args.retrieval:
        stage='retrieval_report' if args.report else 'retrieval_experiment'
    if args.counts_retrieval:
        stage='counts_retrieval_report' if args.report else 'counts_retrieval'
    pipeline=Synpp.build_from_yml(args.config,run=['analysis.matching.'+stage])
    pipeline.debug=True
    pipeline.dryrun=False
    pipeline.flowchart_path=None
    print(pipeline.run_pipeline()[-1],flush=True)


if __name__=='__main__':
    main()
