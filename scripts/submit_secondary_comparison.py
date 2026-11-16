"""Submit focused NN training/assignment and a dependent RDA comparison report."""
import argparse
import json
import pickle
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import networkx as nx
import yaml
from synpp.pipeline import process_stages

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PREFIX='synthesis.population.spatial.secondary_nn.'
EXPERTS=['regional_model','subregional_model','local_model','short_range_model']


def prepare(source, directory, distance):
    settings=yaml.safe_load(source.read_text())
    if settings['config'].get('locations_algorithm')!='nn' or settings['config'].get('secondary_nn_use_target_distance') is not distance:
        raise ValueError(f'Unexpected secondary algorithm/distance setting in {source.name}')
    settings.update(debug=True,dryrun=False,run=[PREFIX+name for name in EXPERTS]+['synthesis.population.spatial.secondary.locations'])
    settings['flowchart_path']=str(directory/(source.stem+'_flowchart.json'))
    settings['config'].update(secondary_nn_no_distance_routing='hierarchy',secondary_nn_hierarchy_short_threshold_m=1600.,
        secondary_nn_model_validation_strategy='rotating_holdout',secondary_nn_model_min_epochs=30)
    definitions=[dict(descriptor=name,config={},downstream=[]) for name in settings['run']]
    registry=process_stages(definitions,settings['config'])
    graph=nx.DiGraph()
    graph.add_nodes_from(registry)
    for key,stage in registry.items():
        graph.add_edges_from((dependency,key) for dependency in stage['dependencies'])
    required=[key for key,stage in registry.items() if 'required-index' in stage]
    executing=set(required)
    for key in required:executing.update(nx.descendants(graph,key))
    names=[registry[key]['wrapper'].name for key in executing]
    assert len(names)==6 and set(names)==set(settings['run']+[PREFIX+'locations']),names
    cache=Path(settings['working_directory'])
    meta=json.loads((cache/'pipeline.json').read_text())
    reused={dep for key in executing for dep in registry[key]['dependencies'] if dep not in executing}
    choices={}
    def complete(key):return (cache/(key+'.p')).is_file() and (cache/(key+'.cache')).is_dir()
    for key in reused:
        name=registry[key]['wrapper'].name
        candidates=[candidate for candidate in meta if candidate.startswith(name+'__') and candidate not in executing and complete(candidate)]
        if complete(key):chosen=key
        elif candidates:chosen=max(candidates,key=lambda candidate:(meta[candidate]['updated'],candidate))
        else:raise ValueError(f'{source.name}: missing cached prerequisite {name}')
        choices[name]=chosen
    # Reject incompatible cached feature schemas before submitting either job.
    from synthesis.population.spatial.secondary_nn.tools import MOBILITY_FEATURES, OVGK_MEAN_FEATURES
    from synthesis.population.spatial.secondary_nn.chain_state import CHAIN_STATE_FEATURES
    for stage in ['mobility','mz_chains','h3']:
        with (cache/(choices[PREFIX+stage]+'.p')).open('rb') as stream:
            data=pickle.load(stream)
        if stage=='mobility':
            assert set(MOBILITY_FEATURES+['person_id']) <= set(data.columns), 'Cached mobility schema is stale'
        elif stage=='mz_chains':
            assert 'chain_state' in data and all(len(state)==len(CHAIN_STATE_FEATURES) for state in data.chain_state), 'Cached chain features are stale'
        else:
            assert all(set(OVGK_MEAN_FEATURES) <= set(cells.columns) for cells in data[1].values()), 'Cached H3 accessibility features are stale'
        del data
    destination=directory/source.name
    destination.write_text(yaml.safe_dump(settings,sort_keys=False))
    return destination,settings,dict(executing=sorted(names),reused=choices)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory=ROOT/'logs'/('secondary_comparison_'+stamp)
    directory.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'analysis/secondary locations'))
    from cache_io import RunCache
    rda=RunCache(ROOT/'config_dib_10.yml')
    if rda.kind!='rda':raise ValueError('The reference config must have completed RDA results')
    plans=[prepare(ROOT/name,directory,distance) for name,distance in
           [('config_dib_10_sNN.yml',True),('config_dib_10_sNNnD.yml',False)]]
    manifest=dict(created_utc=stamp,rda_config=str(ROOT/'config_dib_10.yml'),rda_root=rda.root,debug=True,runs=[])
    for path,settings,plan in plans:
        threads=int(settings['config']['threads'])
        record=dict(config=str(path),source_config=str(ROOT/path.name),plan=plan,threads=threads)
        command=['sbatch','--parsable','--job-name='+path.stem.removeprefix('config_dib_')+'_secondary',
            '--chdir='+str(ROOT),'--cpus-per-task='+str(threads),'--mem-per-cpu=5G',
            '--output='+str(ROOT/'logs/%x_%j.log'),'--error='+str(ROOT/'logs/%x_%j.log'),
            str(ROOT/'scripts/run_secondary_locations.sh'),str(ROOT),str(path)]
        if not args.dry_run:
            record['job_id']=subprocess.check_output(command,text=True).strip().split(';')[0]
        manifest['runs'].append(record)
        (directory/'submissions.json').write_text(json.dumps(manifest,indent=2))
        print(path.name, '6 focused stages;',len(plan['reused']),'cached inputs;',record.get('job_id','dry run'),flush=True)
    if not args.dry_run:
        result=subprocess.check_output(['bash',str(ROOT/'analysis/secondary locations/generate_report'),'--slurm',
            '--after-job',':'.join(run['job_id'] for run in manifest['runs']),
            '--configs',str(ROOT/'config_dib_10.yml'),*[str(path) for path,_,_ in plans]],text=True).strip()
        manifest['report_submission']=result
        (directory/'submissions.json').write_text(json.dumps(manifest,indent=2))
        print(result,flush=True)
    print('Submission manifest:',directory/'submissions.json')


if __name__=='__main__':main()
