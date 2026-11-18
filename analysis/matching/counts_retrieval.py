"""Compare existing count-head and baseline encoders at 200 neighbours / 0.1."""
import json
from pathlib import Path
import pandas as pd
from . import retrieval_experiment


def configure(context):
    retrieval_experiment.configure(context)


def execute(context):

    # Reuse both model variants; match with equal settings, seeds and 100 donor draws.

    summaries=[]
    for label, variant in [('Baseline','Current'),('Count head','Counts')]:
        directory=Path(retrieval_experiment.execute(context,model_variant=variant,
            neighbours=[200],temperatures=[.1],draws=100,
            output_name='counts_retrieval_k200_t01/'+variant))
        frame=pd.read_csv(directory/'summary.csv')
        frame['model']=label
        summaries.append(frame)
    root=directory.parent
    pd.concat(summaries,ignore_index=True).to_csv(root/'summary.csv',index=False)
    (root/'manifest.json').write_text(json.dumps(dict(models={'Baseline':'Current','Count head':'Counts'},
        neighbours=200,temperature=.1,draws=100,seeds=3,
        interpretation='Fixed prior encoders; original count-only head; exploratory follow-up on the same household split'),indent=2))
    return str(root)
