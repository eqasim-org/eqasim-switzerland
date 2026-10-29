#!/usr/bin/env python3
"""Sequential loss-weight experiment using frozen completed secondary-run inputs."""
import argparse
import fcntl
import json
import logging
from pathlib import Path
import subprocess
import sys

REPOSITORY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPOSITORY))
from loss_sweep_tools import create_plan, execute_variant, FrozenCache, preflight

LOG=logging.getLogger('secondary_loss_sweep')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPOSITORY/'config_dib_10_nn_distance.yml')
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results')/'loss_weight_sweep')
    parser.add_argument('--weights',type=float,nargs='+',default=[.2,.3,.5])
    parser.add_argument('--threads',type=int,default=12)
    parser.add_argument('--replay-people',type=int,default=1000)
    parser.add_argument('--draws',type=int,default=3)
    parser.add_argument('--phase',choices=['plan','all','variant','compare'],default='all')
    parser.add_argument('--weight',type=float)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    output=args.output.resolve()
    path=output/'plan.json'

    # Prepare a private snapshot once, then leave the user's source configuration untouched.

    if path.exists():
        plan=json.loads(path.read_text())
        expected=(args.weights,args.threads,args.replay_people,args.draws,str(args.config.resolve()))
        actual=(plan['weights'],plan['threads'],plan['replay_people'],plan['draws'],plan['source_config'])
        if expected!=actual:
            raise ValueError('Arguments differ from the saved plan; choose another output directory')
        preflight(plan)
    else:
        if args.phase in ['variant','compare']:
            raise FileNotFoundError('Prepare the sweep plan first')
        plan=create_plan(args.config,output,args.weights,args.threads,args.replay_people,args.draws)
    LOG.info('Frozen baseline: %s; sequential weights: %s',plan['baseline_root'],plan['weights'])
    if args.phase=='plan':
        return
    if args.phase=='variant':
        execute_variant(plan,args.weight)
        return
    if args.phase=='compare':
        from loss_sweep_report import compare
        compare(plan)
        return

    # Separate processes release training and assignment memory between trials.

    shared=[sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
            '--output',str(output),'--threads',str(args.threads),'--replay-people',str(args.replay_people),
            '--draws',str(args.draws),'--weights',*[str(w) for w in args.weights]]
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for weight in plan['weights']:
            subprocess.run([*shared,'--phase','variant','--weight',str(weight)],check=True,cwd=REPOSITORY)
        subprocess.run([*shared,'--phase','compare'],check=True,cwd=REPOSITORY)
        FrozenCache(plan).verify()
        LOG.info('All trials and comparison completed: %s',output/'report.html')


if __name__=='__main__':
    main()
