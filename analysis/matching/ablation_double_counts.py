"""Test the combined architecture with twice the count weight; reuse prior controls."""
import json
from pathlib import Path
import pandas as pd
from . import ablation_experiment, ablation_report

PRIOR = 'analysis.matching.ablation_experiment'
NAME = 'Counts + features (2x counts)'


def configure(context):
    ablation_experiment.configure(context)
    context.stage(PRIOR)


def execute(context):

    # Fit only the new combined variant on the same households and fitting seeds.

    prior = Path(context.stage(PRIOR))
    root = Path(ablation_experiment.execute(context, variants={NAME: (True, True)},
        count_multiplier=2.0, output_name='architecture_ablation_double_counts'))

    # Include saved controls only after verifying the comparison's identity.

    membership = pd.read_csv(root / 'membership.csv')
    pd.testing.assert_frame_equal(membership, pd.read_csv(prior / 'membership.csv'))
    manifest = json.loads((root / 'manifest.json').read_text())
    old = json.loads((prior / 'manifest.json').read_text())
    for key in ('seed', 'seeds', 'draws'):
        if manifest[key] != old[key]:
            raise ValueError('Prior control experiment differs: ' + key)
    for key, value in old['weights'].items():
        expected = value * (2 if key == 'trip_count' else 1)
        if manifest['weights'][key] != expected:
            raise ValueError('Unexpected weight difference: ' + key)
    for name in ('scores.csv', 'runtime.csv'):
        new = pd.read_csv(root / name)
        new = new.loc[new['method'].eq(NAME)]
        pd.concat([pd.read_csv(prior / name), new], ignore_index=True).to_csv(root / name, index=False)
    manifest['variants'] = {**old['variants'], NAME: [True, True]}
    manifest['weights_by_method'] = {name: old['weights'] for name in old['variants']}
    manifest['weights_by_method'][NAME] = manifest['weights']
    manifest['interpretation'] = 'Follow-up on the previously examined split; exploratory comparison, not a fresh confirmation'
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return str(root)


def render(directory):
    path = Path(ablation_report.render(directory))
    text = path.read_text().replace('Four matching models', 'Count-weight follow-up: five matching models')
    text = text.replace('All variants share inputs, loss weights, retrieval rules and household splits.',
        'All variants share inputs, retrieval rules and household splits. The added combined variant doubles only the count-loss weight from 1.5 to 3.0; all other loss weights stay fixed. Earlier controls are loaded from the completed experiment.')
    text = text.replace('test scores evaluate all four', 'test scores evaluate all five')
    text = text.replace('<h2>Variant selection', '<p>This follow-up uses a previously examined test split. Results are exploratory, not a new independent confirmation.</p><h2>Variant selection')
    path.write_text(text)
    return str(path)
