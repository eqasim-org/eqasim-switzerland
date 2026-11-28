"""Mode-share summaries of population mode choices before MATSim simulation."""
from .utils import ModeShareAnalyzer
from .simulated import summarize


def configure(context):
    context.stage('synthesis.prepared.population', config={'enable_mode_choice': True}, alias='analysis_population')
    context.stage('synthesis.prepared.households')
    context.stage('data.spatial.swiss_border')
    context.stage('data.external_population.constants')
    context.config('include_external_population', True)
    context.config('group_income_classes_in_analysis', False)


def execute(context):

    # Read the chosen-mode population using the shared production analyzer.

    analyzer = ModeShareAnalyzer(context, source='eqasim',
                                 group_income_classes=context.config('group_income_classes_in_analysis'))

    # Return the MATSim summary contract and normalized trips for distributions.

    result = summarize(analyzer, context)
    result['trips'] = analyzer.trips
    result['source'] = 'eqasim'
    return result


def validate(context):
    from hashlib import sha256
    from pathlib import Path
    from . import inputs, simulated, utils
    return sha256(b''.join(Path(module.__file__).read_bytes()
                          for module in (inputs, simulated, utils))).hexdigest()
